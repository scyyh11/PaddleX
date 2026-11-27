# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and integration tests for multi-page document post-processing in PaddleOCR-VL-1.8."""

import concurrent.futures
import difflib
import gc
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pytest
import yaml
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
except (ImportError, OSError):
    LGBMClassifier = None

from paddlex.inference.pipelines.layout_parsing import merge_table
from paddlex.inference.pipelines.paddleocr_vl.doc_postprocess import (
    DocPostprocessConfig,
    DocPostprocessor,
    DocPostprocessResult,
    ModelBundleError,
    bundle,
    features,
    models,
    portable_trees,
)
from paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.engine import (
    native_similarity,
    postprocess,
    table_graph,
    table_merge_bridge,
    text_features,
    unified,
)
from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline


class _FakeBooster:
    def __init__(self, model_string, best_iteration=-1):
        self._text = model_string
        self.best_iteration = best_iteration

    def model_to_string(self, num_iteration=-1):
        return self._text


class _FakeLGBMClassifier:
    """Minimal stand-in matching LGBMClassifier attributes read by portable_trees."""

    def __init__(self, model_string, classes):
        self.booster_ = _FakeBooster(model_string)
        self.classes_ = np.asarray(classes)


_FakeLGBMClassifier.__name__ = "LGBMClassifier"


_SAMPLE_LGBM_BINARY_TEXT = """tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=3
objective=binary sigmoid:1

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1.0 0.5
threshold=0.5 -0.2
decision_type=2 10
left_child=-1 -2
right_child=1 -3
leaf_value=0.8 -0.6 -0.1
is_linear=0

end of trees
"""


_SAMPLE_LGBM_MULTICLASS_TEXT = """tree
version=v4
num_class=3
num_tree_per_iteration=3
label_index=0
max_feature_idx=3
objective=multiclass num_class:3

Tree=0
num_leaves=2
num_cat=0
split_feature=0
split_gain=1.0
threshold=0.0
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.4 -0.3
is_linear=0

Tree=1
num_leaves=1
num_cat=0
leaf_value=0.1
is_linear=0

Tree=2
num_leaves=2
num_cat=0
split_feature=1
split_gain=1.0
threshold=0.25
decision_type=2
left_child=-1
right_child=-2
leaf_value=-0.2 0.5
is_linear=0

end of trees
"""


def _create_dummy_joblib_bundle(root: Path, include_oof: bool = False):
    rng = np.random.default_rng(42)
    x_bin = rng.normal(size=(40, 6)).astype(np.float32)
    y_bin = (x_bin[:, 0] + x_bin[:, 1] > 0).astype(int)
    x_mc = rng.normal(size=(45, 6)).astype(np.float32)
    y_mc = np.array([1, 2, 3] * 15)
    x_delta = rng.normal(size=(45, 6)).astype(np.float32)
    y_delta = np.array([-1, 0, 1] * 15)

    et_bin = ExtraTreesClassifier(n_estimators=5, random_state=2).fit(x_bin, y_bin)
    hgb_bin = HistGradientBoostingClassifier(max_iter=4, random_state=3).fit(
        x_bin, y_bin
    )
    hgb_depth = HistGradientBoostingClassifier(max_iter=4, random_state=4).fit(
        x_mc, y_mc
    )
    hgb_delta = HistGradientBoostingClassifier(max_iter=4, random_state=5).fit(
        x_delta, y_delta
    )

    specs = {
        "headings": (
            "fold-0.joblib",
            {"binary": hgb_bin, "depth": hgb_depth, "pair": hgb_delta},
            False,
        ),
        "parents": ("parent-fold-0.joblib", {"parent": et_bin}, False),
        "spans": (
            "span-fold-0.joblib",
            {"span": _FakeLGBMClassifier(_SAMPLE_LGBM_BINARY_TEXT, [0, 1])},
            False,
        ),
        "graph": ("graph-fold-0.joblib", {"relation": hgb_delta}, False),
        "tables": ("fold-0.joblib", {"model": hgb_bin, "feature_version": 2}, True),
        "paragraph-runtime": (
            "fold-0.joblib",
            {"model": et_bin, "feature_version": 2},
            True,
        ),
        "graphic": ("fold-0.joblib", {"model": et_bin}, True),
    }
    for family, (filename, pkg, need_manifest) in specs.items():
        d = root / family
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(pkg, d / filename)
        if need_manifest:
            (d / "manifest.json").write_text(
                json.dumps({"folds": {}}), encoding="utf-8"
            )
    if include_oof:
        oof = root / "headings" / "oof"
        oof.mkdir(parents=True, exist_ok=True)
        (oof / "docA.json").write_text(json.dumps({"fold": 0}), encoding="utf-8")
    (root / "bundle.json").write_text(json.dumps({"default_fold": 0}), encoding="utf-8")
    return root


class TestArchitectureHygiene:
    def test_no_bare_modules_env_writes_gc_or_monkey_patches(self):
        env_before = dict(os.environ)
        gc_before = gc.get_threshold()

        importlib.import_module(
            "paddlex.inference.pipelines.paddleocr_vl.doc_postprocess"
        )
        importlib.import_module(
            "paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.engine.unified"
        )

        for bare in (
            "unified",
            "postprocess",
            "table_graph",
            "text_stage",
            "fast_html",
            "runtime_context",
            "doc_resources",
            "cli",
        ):
            assert bare not in sys.modules

        assert dict(os.environ) == env_before
        assert gc.get_threshold() == gc_before

        assert not getattr(merge_table.can_merge_tables, "_doc_postprocess", False)
        assert not getattr(merge_table.perform_table_merge, "_doc_postprocess", False)
        assert not getattr(text_features.merge_nodes, "_runtime_paragraph_gate", False)
        assert not getattr(text_features.merge_nodes, "_identity_guard", False)
        assert not getattr(table_graph.identity_probability, "_structure_guard", False)
        assert not getattr(
            table_graph.identity_probability, "_semantic_identity_guard", False
        )
        assert not getattr(
            postprocess.process_in_memory, "_source_text_structure", False
        )
        assert unified.__name__.endswith(".engine.unified")

    def test_multiple_doc_postprocessor_instances_coexist(self, tmp_path):
        dir_a = _create_dummy_joblib_bundle(tmp_path / "bundle_a")
        dir_b = _create_dummy_joblib_bundle(tmp_path / "bundle_b")
        port_a = bundle.export_portable(dir_a, tmp_path / "port_a")
        port_b = bundle.export_portable(dir_b, tmp_path / "port_b")

        cfg_a = DocPostprocessConfig(heading_keep_old=0.45)
        cfg_b = DocPostprocessConfig(heading_keep_old=0.65)

        proc_a = DocPostprocessor(models_dir=port_a, device="cpu", config=cfg_a)
        proc_b = DocPostprocessor(models_dir=port_b, device="cpu", config=cfg_b)
        try:
            assert proc_a.available() and proc_b.available()
            proc_a.warmup()
            proc_b.warmup()
            assert proc_a._model_store.root == port_a.resolve()
            assert proc_b._model_store.root == port_b.resolve()
            assert proc_a.config.heading_keep_old == 0.45
            assert proc_b.config.heading_keep_old == 0.65
        finally:
            proc_a.close()
            proc_b.close()


class TestNativeSimilarity:
    def test_ratio_matches_difflib(self):
        rng = random.Random(0)
        for _ in range(300):
            a, b = (
                "".join(rng.choice("ab c表格一二") for _ in range(rng.randint(0, 40)))
                for _ in range(2)
            )
            expected = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            assert native_similarity.ratio(a, b) == expected


def _reference_ink(crop):
    _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, dtype=np.uint8)
    for i, (_, _, cw, ch, area) in enumerate(stats):
        if i and area >= 3 and ch >= 2 and cw < max(30, ch * 16):
            keep[i] = 1
    return keep[labels]


def _synthetic_page(seed=0):
    rng = np.random.default_rng(seed)
    gray = np.full((600, 500), 245, np.uint8)
    for row in range(40, 560, 22):
        x = 30
        while x < 460:
            w = int(rng.integers(4, 14))
            cv2.rectangle(
                gray, (x, row), (x + w, row + 12), int(rng.integers(0, 80)), -1
            )
            x += w + int(rng.integers(2, 9))
    cv2.line(gray, (20, 300), (480, 300), 0, 1)
    return gray


class TestFeatures:
    def test_component_filter_matches_reference(self):
        gray = _synthetic_page()
        rng = np.random.default_rng(7)
        clean = gray.copy()
        clean[300] = 245
        specks = gray.copy()
        specks[rng.integers(0, 600, 3000), rng.integers(0, 500, 3000)] = 0
        for image in (gray, clean, specks):
            crop = image[20:580, 10:490]
            assert np.array_equal(features._ink(crop), _reference_ink(crop))

    def test_gray_conversion_matches_png_files(self, tmp_path):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        image[:16] = image[:16, :, :1]
        path = str(tmp_path / "page.png")
        cv2.imwrite(path, image)
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        expected = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(features.to_gray(color), expected)
        bgra = np.dstack([color, np.full(color.shape[:2], 7, np.uint8)])
        assert np.array_equal(features.to_gray(bgra), expected)
        sample = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        assert np.array_equal(
            features.to_gray(sample), features._png_gray_reference(sample)
        )

    def test_shared_crops_and_threads_parity(self):
        images = [_synthetic_page(seed) for seed in range(5)]
        pages = [
            {
                "width": 500,
                "height": 600,
                "parsing_res_list": [
                    {"block_label": label, "block_content": "x", "block_bbox": box}
                    for label, box in [
                        ("text", [10, 20 + seed, 490, 300]),
                        ("paragraph_title", [10, 40, 300, 70 + seed]),
                        ("footer", [10, 300, 490, 580]),
                    ]
                ],
            }
            for seed in range(5)
        ]
        gray = images.__getitem__
        expected = features.heading_features(pages, gray)
        text, labels = {}, {}
        features.line_features(pages, gray, text, labels)
        with concurrent.futures.ThreadPoolExecutor(3) as executor:
            measurements = features.Measurements()
            threaded = features.heading_features(pages, gray, measurements, executor)
            text2, labels2 = {}, {}
            features.line_features(pages, gray, text2, labels2, measurements, executor)
        assert list(threaded.items()) == list(expected.items())
        assert list(text2.items()) == list(text.items())
        assert list(labels2.items()) == list(labels.items())


class TestPortableTreesAndBundles:
    def test_safetensors_sklearn_trees_match_originals(self, tmp_path):
        rng = np.random.default_rng(123)
        x_train = rng.normal(size=(120, 8)).astype(np.float32)
        y_bin = (x_train[:, 0] * 0.7 - x_train[:, 1] * 0.4 > 0).astype(int)
        y_mc = np.where(x_train[:, 2] < -0.4, 0, np.where(x_train[:, 2] < 0.4, 1, 2))

        x_test = rng.normal(size=(50, 8)).astype(np.float32)
        x_test[0, 0] = np.nan

        et = ExtraTreesClassifier(n_estimators=8, random_state=0).fit(x_train, y_bin)
        hgb_bin = HistGradientBoostingClassifier(max_iter=8, random_state=1).fit(
            x_train, y_bin
        )
        hgb_mc = HistGradientBoostingClassifier(max_iter=8, random_state=2).fit(
            x_train, y_mc
        )

        pkg = {
            "et": et,
            "hgb_bin": hgb_bin,
            "hgb_mc": hgb_mc,
            "feature_version": 2,
            "training_documents": {"secret_doc"},
        }
        sf_path, js_path = portable_trees.save_portable_package(pkg, tmp_path / "m")
        assert sf_path.is_file() and js_path.is_file()
        loaded = portable_trees.load_portable_package(sf_path)
        assert "training_documents" not in loaded
        assert loaded["feature_version"] == 2

        for key in ("hgb_bin", "hgb_mc"):
            np.testing.assert_allclose(
                loaded[key].predict_proba(x_test),
                pkg[key].predict_proba(x_test),
                atol=1e-12,
                rtol=1e-12,
            )
        # ExtraTrees does not accept NaN in sklearn<1.3/some configs; compare finite rows
        np.testing.assert_allclose(
            loaded["et"].predict_proba(x_test[1:]),
            et.predict_proba(x_test[1:]),
            atol=1e-12,
            rtol=1e-12,
        )
        assert loaded["hgb_bin"].predict_proba(x_test[:0]).shape == (0, 2)
        assert loaded["et"].predict_proba(x_test[:0]).shape == (0, 2)

    def test_safetensors_lgbm_text_spec_binary_and_multiclass(self, tmp_path):
        fake_bin = _FakeLGBMClassifier(_SAMPLE_LGBM_BINARY_TEXT, [0, 1])
        fake_mc = _FakeLGBMClassifier(_SAMPLE_LGBM_MULTICLASS_TEXT, [-1, 0, 1])
        sf_path, _ = portable_trees.save_portable_package(
            {"bin": fake_bin, "mc": fake_mc}, tmp_path / "lgbm"
        )
        loaded = portable_trees.load_portable_package(sf_path)

        # Test decision_type=2 (nan->0), decision_type=10 (mt=2, nan->default_left=1), kZeroThreshold
        x = np.array(
            [
                [0.2, -0.5, 0.0, 0.0],  # 0.2<=0.5 -> left=1 -> leaf 0.8
                [
                    0.8,
                    -0.5,
                    0.0,
                    0.0,
                ],  # 0.8>0.5 -> right -> child1 (-0.5<=-0.2) -> -0.6
                [0.8, 1e-36, 0.0, 0.0],  # |1e-36|<=kZeroThreshold -> 0.0 > -0.2 -> -0.1
                [0.8, np.nan, 0.0, 0.0],  # mt=2, default_left=1 -> -0.6
                [np.nan, 0.0, 0.0, 0.0],  # mt=0, nan->0.0 <= 0.5 -> 0.8
            ],
            dtype=np.float32,
        )
        expected_raw = [0.8, -0.6, -0.1, -0.6, 0.8]
        expected_p1 = np.array([1.0 / (1.0 + math.exp(-v)) for v in expected_raw])
        np.testing.assert_allclose(
            loaded["bin"].booster_.predict(x, num_threads=1),
            expected_p1,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            loaded["bin"].predict_proba(x),
            np.column_stack([1.0 - expected_p1, expected_p1]),
            atol=1e-12,
        )

        mc_proba = loaded["mc"].predict_proba(x[:2])
        assert mc_proba.shape == (2, 3)
        np.testing.assert_allclose(mc_proba.sum(axis=1), [1.0, 1.0], atol=1e-12)

    @pytest.mark.skipif(LGBMClassifier is None, reason="lightgbm runtime unavailable")
    def test_live_lgbm_classifier_parity(self, tmp_path):
        rng = np.random.default_rng(9)
        x_train = rng.normal(size=(80, 6)).astype(np.float32)
        y_bin = (x_train[:, 0] > 0).astype(int)
        lgb_bin = LGBMClassifier(n_estimators=6, max_depth=3, verbose=-1).fit(
            x_train, y_bin
        )
        x_test = rng.normal(size=(30, 6)).astype(np.float32)
        x_test[0, 0] = np.nan
        x_test[1, 2] = 1e-36

        sf_path, _ = portable_trees.save_portable_package(
            {"lgb": lgb_bin}, tmp_path / "live_lgb"
        )
        loaded = portable_trees.load_portable_package(sf_path)
        np.testing.assert_allclose(
            loaded["lgb"].predict_proba(x_test),
            lgb_bin.predict_proba(x_test),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_bundle_export_portable_and_end_to_end_pipeline(self, tmp_path):
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib", include_oof=True)
        dst = tmp_path / "dst_portable"
        bundle.export_portable(src, dst)

        assert models.problems(dst) == []
        assert not list(dst.rglob("*.joblib"))
        assert len(list(dst.rglob("*.safetensors"))) == 7

        # Exercise the full 5-stage engine end-to-end with the portable bundle
        img0 = _synthetic_page(10)
        img1 = _synthetic_page(11)
        pages = [
            {
                "input_path": "sample.pdf",
                "page_index": 0,
                "page_count": 2,
                "width": 500,
                "height": 600,
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_layout_detection": False,
                    "use_chart_recognition": False,
                },
                "parsing_res_list": [
                    {
                        "block_label": "doc_title",
                        "block_content": "Chapter 1 Overview",
                        "block_bbox": [30, 40, 350, 70],
                    },
                    {
                        "block_label": "text",
                        "block_content": "This sentence continues across the page boundary",
                        "block_bbox": [30, 100, 460, 220],
                    },
                ],
            },
            {
                "input_path": "sample.pdf",
                "page_index": 1,
                "page_count": 2,
                "width": 500,
                "height": 600,
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_layout_detection": False,
                    "use_chart_recognition": False,
                },
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "and finishes cleanly here.",
                        "block_bbox": [30, 50, 460, 120],
                    },
                    {
                        "block_label": "paragraph_title",
                        "block_content": "1.1 Details",
                        "block_bbox": [30, 150, 240, 180],
                    },
                ],
            },
        ]
        processor = DocPostprocessor(models_dir=dst, device="cpu", num_threads=2)
        try:
            doc = processor.process(pages, images=[img0, img1])
            assert isinstance(doc, DocPostprocessResult)
            assert doc["page_count"] == 2
            assert "Chapter 1 Overview" in doc["markdown_text"]
            assert "1.1 Details" in doc["markdown_text"]
        finally:
            processor.close()


class TestTableMergeBridge:
    def test_cached_bridge_matches_default_merge_tables_across_pages(self):
        def make_pages():
            b1 = SimpleNamespace(
                label="table",
                bbox=[10, 100, 490, 500],
                content="<table><tr><th>ColA</th><th>ColB</th></tr><tr><td>1</td><td>2</td></tr></table>",
                global_block_id=0,
                global_group_id=0,
            )
            b2 = SimpleNamespace(
                label="table",
                bbox=[10, 50, 490, 300],
                content="<table><tr><th>ColA</th><th>ColB</th></tr><tr><td>3</td><td>4</td></tr></table>",
                global_block_id=1,
                global_group_id=1,
            )
            return [[b1], [b2]]

        p_default = merge_table.merge_tables_across_pages(make_pages())
        tok = table_merge_bridge.html_cache.set({})
        try:
            p_cached = merge_table.merge_tables_across_pages(
                make_pages(),
                soup_factory=table_merge_bridge.readonly_soup_factory,
                table_merger=table_merge_bridge.cached_table_merger,
            )
        finally:
            table_merge_bridge.html_cache.reset(tok)

        assert p_cached[0][0].content == p_default[0][0].content
        assert p_cached[1][0].content == p_default[1][0].content == ""
        assert p_cached[1][0].global_group_id == p_default[1][0].global_group_id == 0


class TestPaddleOCRVLIntegration:
    @staticmethod
    def _dummy_page_res(page_idx, text):
        return {
            "res": {
                "input_path": "doc.pdf",
                "page_index": page_idx,
                "page_count": 2,
                "width": 500,
                "height": 600,
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_layout_detection": False,
                    "use_chart_recognition": False,
                },
                "parsing_res_list": [
                    {
                        "block_label": "paragraph_title",
                        "block_content": text,
                        "block_bbox": [20, 30, 300, 60],
                    }
                ],
            }
        }

    def test_vl18_yaml_has_doc_postprocess_config(self):
        cfg_path = (
            Path(__file__).resolve().parents[2]
            / "paddlex"
            / "configs"
            / "pipelines"
            / "PaddleOCR-VL-1.8.yaml"
        )
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert "use_doc_postprocess" in cfg
        assert cfg["use_doc_postprocess"] is None
        assert cfg["doc_postprocess_device"] == "cpu"

    def test_restructure_pages_fallbacks_and_explicit_errors(self, tmp_path):
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = None
        pipeline.doc_postprocess_models_dir = str(tmp_path / "missing_models")
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        single = list(pipeline.restructure_pages([self._dummy_page_res(0, "Only")]))
        assert len(single) == 1
        assert pipeline._doc_postprocess_warned is False

        out_disabled = list(
            pipeline.restructure_pages(
                [self._dummy_page_res(0, "A"), self._dummy_page_res(1, "B")],
                use_doc_postprocess=False,
            )
        )
        assert len(out_disabled) == 2
        assert pipeline._doc_postprocess_warned is False

        with pytest.raises(ModelBundleError):
            list(
                pipeline.restructure_pages(
                    [self._dummy_page_res(0, "A"), self._dummy_page_res(1, "B")],
                    use_doc_postprocess=True,
                )
            )

        out_fallback = list(
            pipeline.restructure_pages(
                [self._dummy_page_res(0, "A"), self._dummy_page_res(1, "B")]
            )
        )
        assert len(out_fallback) == 2
        assert pipeline._doc_postprocess_warned is True

    def test_restructure_pages_uses_doc_postprocessor_when_ready(self, tmp_path):
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = None
        pipeline.doc_postprocess_models_dir = str(tmp_path)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocess_warned = False

        fake_doc = DocPostprocessResult(
            doc_name="doc",
            input_path="doc.pdf",
            page_count=2,
            markdown_text="# Title A\n\n## Title B\n",
            pages=[
                {
                    "width": 500,
                    "height": 600,
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                    },
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "Title A",
                            "block_bbox": [20, 30, 300, 60],
                            "global_block_id": 0,
                            "global_group_id": 0,
                            "title_level": 0,
                        }
                    ],
                },
                {
                    "width": 500,
                    "height": 600,
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                    },
                    "parsing_res_list": [
                        {
                            "block_label": "paragraph_title",
                            "block_content": "Title B",
                            "block_bbox": [20, 30, 300, 60],
                            "global_block_id": 1,
                            "global_group_id": 1,
                            "title_level": 1,
                        }
                    ],
                },
            ],
            images={},
            events={},
            timings={"total_ms": 1.2},
        )
        pipeline._doc_postprocessor = SimpleNamespace(
            models_dir=tmp_path,
            available=lambda: True,
            process=lambda res_list, doc_name=None, images=None: fake_doc,
            close=lambda: None,
        )

        concat = list(
            pipeline.restructure_pages(
                [
                    self._dummy_page_res(0, "Title A"),
                    self._dummy_page_res(1, "Title B"),
                ],
                concatenate_pages=True,
            )
        )
        assert len(concat) == 1
        assert isinstance(concat[0], DocPostprocessResult)
        assert concat[0]["markdown_text"] == "# Title A\n\n## Title B\n"
        assert concat[0].markdown["markdown_texts"] == "# Title A\n\n## Title B\n"
        concat[0].save_all(tmp_path / "out")
        assert (tmp_path / "out" / "doc.md").is_file()
        assert (tmp_path / "out" / "doc.json").is_file()

        per_page = list(
            pipeline.restructure_pages(
                [
                    self._dummy_page_res(0, "Title A"),
                    self._dummy_page_res(1, "Title B"),
                ],
                concatenate_pages=False,
            )
        )
        assert len(per_page) == 2
        assert per_page[0]["parsing_res_list"][0].label == "doc_title"
        assert per_page[1]["parsing_res_list"][0].title_level == 1
