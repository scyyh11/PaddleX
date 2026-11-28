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

import ast
import asyncio
import concurrent.futures
import difflib
import functools
import gc
import importlib
import json
import math
import os
import random
import sys
import threading
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
    markup_repair,
    native_similarity,
    postprocess,
    table_graph,
    table_merge_bridge,
    text_features,
    unified,
)
from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline
from paddlex.inference.pipelines.paddleocr_vl.result import (
    PaddleOCRVLBlock,
    PaddleOCRVLResult,
)
from paddlex.utils.deps import is_serving_plugin_available


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
        assert isinstance(concat[0], PaddleOCRVLResult)
        assert [
            b["block_content"] for b in concat[0].json["res"]["parsing_res_list"]
        ] == [
            "Title A",
            "Title B",
        ]
        md_data = concat[0]._to_markdown(pretty=True, show_formula_number=False)
        assert "Title A" in md_data["markdown_texts"]
        assert "Title B" in md_data["markdown_texts"]
        docx_dir = tmp_path / "docx_out"
        concat[0].save_to_word(docx_dir)
        docx_path = docx_dir / f"{Path(concat[0]._get_input_fn()).stem}.docx"
        assert docx_path.is_file()
        assert docx_path.read_bytes()[:2] == b"PK"

        if is_serving_plugin_available():
            from paddlex.inference.serving.app_shared.document_export import (
                docx_bytes_and_filename_from_word_mixin,
                refill_paddleocr_vl_images_from_markdown,
            )
            from paddlex.inference.serving.basic_serving._pipeline_apps._common import (
                common,
            )

            pruned = common.prune_result(concat[0].json["res"])
            assert len(pruned["parsing_res_list"]) == 2
            refill_paddleocr_vl_images_from_markdown(concat[0], {})
            docx_bytes, filename = docx_bytes_and_filename_from_word_mixin(concat[0])
            assert filename == "doc.docx"
            assert docx_bytes[:2] == b"PK"

        direct_doc = pipeline.postprocess_document(
            [
                self._dummy_page_res(0, "Title A"),
                self._dummy_page_res(1, "Title B"),
            ]
        )
        assert isinstance(direct_doc, DocPostprocessResult)
        assert direct_doc["markdown_text"] == "# Title A\n\n## Title B\n"
        direct_doc.save_all(tmp_path / "out")
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

    def test_restructure_pages_concatenate_serving_and_docx_export(self, tmp_path):
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(dst)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        pages = [
            self._dummy_page_res(0, "Section 1"),
            self._dummy_page_res(1, "1.1 Subsection"),
        ]
        try:
            out = list(pipeline.restructure_pages(pages, concatenate_pages=True))
            assert len(out) == 1
            res = out[0]
            assert isinstance(res, PaddleOCRVLResult)
            res_json = res.json["res"]
            assert res_json["page_count"] == 2
            assert len(res_json["parsing_res_list"]) == 2
            md_data = res._to_markdown(pretty=True, show_formula_number=False)
            assert "Section 1" in md_data["markdown_texts"]
            assert "1.1 Subsection" in md_data["markdown_texts"]
            docx_dir = tmp_path / "docx_e2e"
            res.save_to_word(docx_dir)
            docx_path = docx_dir / f"{Path(res._get_input_fn()).stem}.docx"
            assert docx_path.is_file()
            assert docx_path.read_bytes()[:2] == b"PK"

            if is_serving_plugin_available():
                from paddlex.inference.serving.app_shared.document_export import (
                    docx_bytes_and_filename_from_word_mixin,
                )
                from paddlex.inference.serving.basic_serving._pipeline_apps._common import (
                    common,
                )

                pruned = common.prune_result(res_json)
                assert len(pruned["parsing_res_list"]) == 2
                docx_bytes, filename = docx_bytes_and_filename_from_word_mixin(res)
                assert filename == "doc.docx"
                assert docx_bytes[:2] == b"PK"
        finally:
            if pipeline._doc_postprocessor is not None:
                pipeline._doc_postprocessor.close()

    def test_restructure_pages_respects_independent_merge_tables_and_relevel_titles(
        self, tmp_path
    ):
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(tmp_path)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocess_warned = False
        called = False

        def fail_if_called(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("postprocess_document should not run")

        pipeline._doc_postprocessor = SimpleNamespace(
            models_dir=tmp_path,
            available=lambda: True,
            process=fail_if_called,
            close=lambda: None,
        )

        def make_table_pages():
            return [
                {
                    "res": {
                        "input_path": "doc.pdf",
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
                                "block_label": "paragraph_title",
                                "block_content": "Chapter 1 Overview",
                                "block_bbox": [20, 30, 300, 60],
                            },
                            {
                                "block_label": "table",
                                "block_content": "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                                "block_bbox": [20, 100, 480, 500],
                            },
                        ],
                    }
                },
                {
                    "res": {
                        "input_path": "doc.pdf",
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
                                "block_label": "table",
                                "block_content": "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>",
                                "block_bbox": [20, 50, 480, 250],
                            },
                            {
                                "block_label": "paragraph_title",
                                "block_content": "1.1 Details",
                                "block_bbox": [20, 300, 260, 330],
                            },
                        ],
                    }
                },
            ]

        no_merge = list(
            pipeline.restructure_pages(
                make_table_pages(),
                merge_tables=False,
                relevel_titles=True,
                concatenate_pages=False,
            )
        )
        assert not called
        assert (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr></table>"
            in (no_merge[1]["parsing_res_list"][0].content)
        )

        no_relevel = list(
            pipeline.restructure_pages(
                make_table_pages(),
                merge_tables=True,
                relevel_titles=False,
                concatenate_pages=False,
            )
        )
        assert not called
        assert not hasattr(no_relevel[0]["parsing_res_list"][0], "title_level")
        assert no_relevel[1]["parsing_res_list"][0].content == ""

    def test_serving_restructure_pages_runs_generator_off_event_loop(self):
        app_path = (
            Path(__file__).resolve().parents[2]
            / "paddlex"
            / "inference"
            / "serving"
            / "basic_serving"
            / "_pipeline_apps"
            / "paddleocr_vl.py"
        )
        tree = ast.parse(app_path.read_text(encoding="utf-8"))

        # Verify both _infer and _restructure_pages consume restructure_pages inside call_async.
        call_async_targets = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "call_async"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                call_async_targets.append(node.args[0].id)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "restructure_pages"
            ):
                # Only _restructure_pages_sync should call pipeline.restructure_pages
                pass
        assert call_async_targets.count("_restructure_pages_sync") == 2

        fn_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_restructure_pages_sync"
        )
        mod = ast.Module(body=[fn_node], type_ignores=[])
        ns = {"Any": object, "List": list}
        exec(compile(mod, str(app_path), "exec"), ns)
        sync_fn = ns["_restructure_pages_sync"]

        loop_thread = None
        gen_thread = None

        class _FakePipeline:
            def restructure_pages(self, res_list, **kwargs):
                nonlocal gen_thread
                gen_thread = threading.get_ident()
                yield res_list[0]

        async def run():
            nonlocal loop_thread
            loop_thread = threading.get_ident()
            return await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    sync_fn,
                    _FakePipeline(),
                    ["ok"],
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                ),
            )

        res = asyncio.run(run())
        assert res == ["ok"]
        assert gen_thread is not None
        assert gen_thread != loop_thread

        if is_serving_plugin_available():
            from paddlex.inference.serving.basic_serving._pipeline_apps import (
                paddleocr_vl as vl_serving,
            )
            from paddlex.inference.serving.infra import utils as serving_utils

            gen_thread = None

            async def run_serving():
                nonlocal loop_thread
                loop_thread = threading.get_ident()
                return await serving_utils.call_async(
                    vl_serving._restructure_pages_sync,
                    _FakePipeline(),
                    ["ok"],
                    merge_tables=True,
                    relevel_titles=True,
                    concatenate_pages=True,
                )

            assert asyncio.run(run_serving()) == ["ok"]
            assert gen_thread is not None and gen_thread != loop_thread

    def test_per_page_json_preserves_geometry_and_group_id(self, tmp_path):
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(dst)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        poly0 = [[20, 30], [300, 30], [300, 60], [20, 60]]
        quad0 = [20, 30, 300, 30, 300, 60, 20, 60]
        poly1 = [[20, 80], [320, 80], [320, 120], [20, 120]]
        quad1 = [20, 80, 320, 80, 320, 120, 20, 120]
        pages = [
            {
                "res": {
                    "input_path": "doc.pdf",
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
                            "block_label": "paragraph_title",
                            "block_content": "Chapter 1",
                            "block_bbox": [20, 30, 300, 60],
                            "block_polygon_points": poly0,
                            "block_quad": quad0,
                            "group_id": 17,
                        }
                    ],
                }
            },
            {
                "res": {
                    "input_path": "doc.pdf",
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
                            "block_content": "Body paragraph content.",
                            "block_bbox": [20, 80, 320, 120],
                            "block_polygon_points": poly1,
                            "block_quad": quad1,
                            "group_id": 29,
                        }
                    ],
                }
            },
        ]
        try:
            per_page = list(pipeline.restructure_pages(pages, concatenate_pages=False))
            assert len(per_page) == 2
            blk0 = per_page[0].json["res"]["parsing_res_list"][0]
            blk1 = per_page[1].json["res"]["parsing_res_list"][0]
            assert blk0["block_polygon_points"] == poly0
            assert blk0["block_quad"] == quad0
            assert blk0["group_id"] == 17
            assert blk1["block_polygon_points"] == poly1
            assert blk1["block_quad"] == quad1
            assert blk1["group_id"] == 29
        finally:
            if pipeline._doc_postprocessor is not None:
                pipeline._doc_postprocessor.close()

    def test_models_problems_rejects_partial_heading_final_bundle(self, tmp_path):
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "partial_bundle")
        assert models.problems(dst) == []

        # Rename only headings to `final` while leaving parents/spans/graph at fold-0
        (dst / "headings" / "fold-0.safetensors").rename(
            dst / "headings" / "fold-final.safetensors"
        )
        (dst / "headings" / "fold-0.json").rename(dst / "headings" / "fold-final.json")
        issues = models.problems(dst)
        assert issues == ["headings/fold-0(.safetensors|.joblib)"]
        with pytest.raises(ModelBundleError):
            models.check(dst)

        # Providing `final` across all 4 heading families satisfies validation again
        for family, stem in (
            ("parents", "parent-fold"),
            ("spans", "span-fold"),
            ("graph", "graph-fold"),
        ):
            (dst / family / f"{stem}-0.safetensors").rename(
                dst / family / f"{stem}-final.safetensors"
            )
            (dst / family / f"{stem}-0.json").rename(
                dst / family / f"{stem}-final.json"
            )
        assert models.problems(dst) == []

    def test_empty_tables_across_pages_do_not_raise_index_error(self, tmp_path):
        def make_empty_table_pages():
            b1 = SimpleNamespace(
                label="table",
                bbox=[10, 100, 490, 500],
                content="<table></table>",
                global_block_id=0,
                global_group_id=0,
            )
            b2 = SimpleNamespace(
                label="table",
                bbox=[10, 50, 490, 300],
                content="<table></table>",
                global_block_id=1,
                global_group_id=1,
            )
            return [[b1], [b2]]

        tok = table_merge_bridge.html_cache.set({})
        try:
            merged = merge_table.merge_tables_across_pages(
                make_empty_table_pages(),
                soup_factory=table_merge_bridge.readonly_soup_factory,
                table_merger=table_merge_bridge.cached_table_merger,
            )
        finally:
            table_merge_bridge.html_cache.reset(tok)
        assert merged[0][0].content == "<table></table>"
        assert merged[1][0].content == ""

        # Also cover BS4 fallback and empty previous + non-empty current table
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pages = [
            {
                "input_path": "empty_tables.pdf",
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
                        "block_label": "table",
                        "block_content": "<table></table>",
                        "block_bbox": [20, 100, 480, 500],
                    }
                ],
            },
            {
                "input_path": "empty_tables.pdf",
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
                        "block_label": "table",
                        "block_content": "<table></table>",
                        "block_bbox": [20, 50, 480, 300],
                    }
                ],
            },
        ]
        processor = DocPostprocessor(models_dir=dst, device="cpu", num_threads=1)
        try:
            doc = processor.process(pages, images=[None, None])
            assert isinstance(doc, DocPostprocessResult)
            assert doc["page_count"] == 2
        finally:
            processor.close()

    def test_restructure_pages_applies_final_markdown_repairs(self, tmp_path):
        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(dst)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        canvas = np.full((600, 500, 3), 245, dtype=np.uint8)
        from PIL import Image

        tiny_img = Image.new("RGB", (5, 5), "white")
        real_img = Image.new("RGB", (300, 250), "white")
        noise_blk = PaddleOCRVLBlock("image", [10, 10, 15, 15], "")
        noise_blk.image = {
            "path": "imgs/img_in_image_box_10_10_15_15.jpg",
            "img": tiny_img,
        }
        keep_blk = PaddleOCRVLBlock("image", [50, 200, 350, 450], "")
        keep_blk.image = {
            "path": "imgs/img_in_image_box_50_200_350_450.jpg",
            "img": real_img,
        }
        pages = [
            PaddleOCRVLResult(
                {
                    "input_path": "repairs.pdf",
                    "page_index": 0,
                    "page_count": 2,
                    "width": 500,
                    "height": 600,
                    "doc_preprocessor_res": {"output_img": canvas},
                    "imgs_in_doc": [],
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                    },
                    "parsing_res_list": [
                        PaddleOCRVLBlock(
                            "text",
                            [20, 40, 460, 90],
                            "A. 正数 B. 负数 C. 非正数 D. 非负数",
                        ),
                        PaddleOCRVLBlock(
                            "display_formula",
                            [40, 120, 440, 170],
                            r"$$ \begin{align*} x = 1 \end{align*} $$",
                        ),
                        noise_blk,
                        keep_blk,
                    ],
                }
            ),
            PaddleOCRVLResult(
                {
                    "input_path": "repairs.pdf",
                    "page_index": 1,
                    "page_count": 2,
                    "width": 500,
                    "height": 600,
                    "doc_preprocessor_res": {"output_img": canvas},
                    "imgs_in_doc": [],
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                    },
                    "parsing_res_list": [
                        PaddleOCRVLBlock(
                            "display_formula",
                            [40, 60, 440, 150],
                            r"$$ \begin{align*} a&=b \\ &=c \tag{1} \end{align*} $$",
                        )
                    ],
                }
            ),
        ]

        try:
            for concat_flag in (True, False):
                results = list(
                    pipeline.restructure_pages(pages, concatenate_pages=concat_flag)
                )
                for pretty_flag in (True, False):
                    merged_text = "\n\n".join(
                        r._to_markdown(pretty=pretty_flag)["markdown_texts"]
                        for r in results
                    )
                    assert "A. 正数\n\nB. 负数\n\nC. 非正数\n\nD. 非负数" in merged_text
                    assert r"\begin{align*}" not in merged_text
                    assert (
                        r"\begin{aligned} a&=b \\ &=c \quad(1) \end{aligned}"
                        in merged_text
                    )
                    assert "10_10_15_15" not in merged_text
                    assert "50_200_350_450" in merged_text

                noise_images = [
                    k
                    for r in results
                    for k in r._to_markdown(pretty=False)["markdown_images"]
                    if "10_10_15_15" in k
                ]
                assert noise_images == []
                assert len(results[0].json["res"]["parsing_res_list"]) > 0

            concat_res = list(
                pipeline.restructure_pages(pages, concatenate_pages=True)
            )[0]
            docx_dir = tmp_path / "repairs_docx"
            concat_res.save_to_word(docx_dir)
            docx_path = docx_dir / f"{Path(concat_res._get_input_fn()).stem}.docx"
            assert docx_path.is_file()
            assert docx_path.read_bytes()[:2] == b"PK"
        finally:
            if pipeline._doc_postprocessor is not None:
                pipeline._doc_postprocessor.close()

    def test_readme_uses_public_postprocess_document_api(self):
        readme = (
            Path(__file__).resolve().parents[2]
            / "paddlex"
            / "inference"
            / "pipelines"
            / "paddleocr_vl"
            / "doc_postprocess"
            / "README.md"
        ).read_text(encoding="utf-8")
        assert "pipeline.postprocess_document(pages)" in readme
        assert "pipeline.pipeline.postprocess_document(" not in readme

    def test_noise_image_preserves_ocr_text_and_filters_image_maps(self, tmp_path):
        from PIL import Image

        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(dst)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        canvas = np.full((600, 500, 3), 245, dtype=np.uint8)
        sliver_path = "imgs/img_in_image_box_10_10_400_20.jpg"
        keep_path = "imgs/img_in_image_box_50_200_350_450.jpg"

        sliver_blk = PaddleOCRVLBlock("image", [10, 10, 400, 20], "REF C4801")
        sliver_blk.image = {
            "path": sliver_path,
            "img": Image.new("RGB", (390, 10), "white"),
        }
        keep_blk = PaddleOCRVLBlock("image", [50, 200, 350, 450], "Figure 1")
        keep_blk.image = {
            "path": keep_path,
            "img": Image.new("RGB", (300, 250), "white"),
        }

        pages = [
            PaddleOCRVLResult(
                {
                    "input_path": "ocr_noise.pdf",
                    "page_index": 0,
                    "page_count": 2,
                    "width": 500,
                    "height": 600,
                    "doc_preprocessor_res": {"output_img": canvas},
                    "imgs_in_doc": [
                        {"path": sliver_path, "img": sliver_blk.image["img"]}
                    ],
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                        "use_ocr_for_image_block": True,
                    },
                    "parsing_res_list": [sliver_blk, keep_blk],
                }
            ),
            PaddleOCRVLResult(
                {
                    "input_path": "ocr_noise.pdf",
                    "page_index": 1,
                    "page_count": 2,
                    "width": 500,
                    "height": 600,
                    "doc_preprocessor_res": {"output_img": canvas},
                    "imgs_in_doc": [],
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_layout_detection": False,
                        "use_chart_recognition": False,
                        "use_ocr_for_image_block": True,
                    },
                    "parsing_res_list": [
                        PaddleOCRVLBlock(
                            "text", [40, 80, 320, 110], "Follow-up page text."
                        )
                    ],
                }
            ),
        ]

        app_path = (
            Path(__file__).resolve().parents[2]
            / "paddlex"
            / "inference"
            / "serving"
            / "basic_serving"
            / "_pipeline_apps"
            / "paddleocr_vl.py"
        )
        tree = ast.parse(app_path.read_text(encoding="utf-8"))
        filter_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_filter_markdown_images"
        )
        ns = {"Any": object}
        exec(
            compile(
                ast.Module(body=[filter_node], type_ignores=[]), str(app_path), "exec"
            ),
            ns,
        )
        filter_fn = ns["_filter_markdown_images"]

        try:
            for concat_flag in (True, False):
                results = list(
                    pipeline.restructure_pages(pages, concatenate_pages=concat_flag)
                )
                assert sliver_path in getattr(
                    results[0], "_dropped_markdown_image_paths", ()
                )
                filtered_http_imgs = filter_fn(
                    {sliver_path: "b64_noise", keep_path: "b64_keep"}, results[0]
                )
                assert sliver_path not in filtered_http_imgs
                assert keep_path in filtered_http_imgs

                for pretty_flag in (True, False):
                    md = results[0]._to_markdown(pretty=pretty_flag)
                    assert "REF C4801" in md["markdown_texts"]
                    assert "10_10_400_20" not in md["markdown_texts"]
                    assert "50_200_350_450" in md["markdown_texts"]
                    assert sliver_path not in md["markdown_images"]
                    assert keep_path in md["markdown_images"]
        finally:
            if pipeline._doc_postprocessor is not None:
                pipeline._doc_postprocessor.close()

    def test_markup_repair_preserves_currency_and_converts_real_math(self, tmp_path):
        assert (
            markup_repair.apply("Base price $10.00 and tax $2.25.")
            == "Base price $10.00 and tax $2.25."
        )
        assert (
            markup_repair.apply("Single item costs $19.99.")
            == "Single item costs $19.99."
        )
        assert (
            markup_repair.apply("The price is $10.00 where $x + 1 > 0$.")
            == r"The price is $10.00 where \(x + 1 > 0\)."
        )
        assert (
            markup_repair.apply(
                "Range $10-$20, sum $10.00 + $20.00 = $30.00, math $1 + 2 = 3$ and $10^2$."
            )
            == r"Range $10-$20, sum $10.00 + $20.00 = $30.00, math \(1 + 2 = 3\) and \(10^2\)."
        )

        src = _create_dummy_joblib_bundle(tmp_path / "src_joblib")
        dst = bundle.export_portable(src, tmp_path / "dst_portable")
        pipeline = object.__new__(_PaddleOCRVLPipeline)
        pipeline.use_doc_postprocess = True
        pipeline.doc_postprocess_models_dir = str(dst)
        pipeline.doc_postprocess_device = "cpu"
        pipeline._doc_postprocessor = None
        pipeline._doc_postprocess_warned = False

        pages = [
            self._dummy_page_res(0, "Pricing Overview"),
            {
                "res": {
                    "input_path": "doc.pdf",
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
                            "block_content": (
                                "Base price $10.00 and tax $2.25. "
                                "Single item costs $19.99. Formula $x + y = z$."
                            ),
                            "block_bbox": [20, 80, 440, 130],
                        }
                    ],
                }
            },
        ]
        try:
            for concat_flag in (True, False):
                results = list(
                    pipeline.restructure_pages(pages, concatenate_pages=concat_flag)
                )
                text = "\n\n".join(r._to_markdown()["markdown_texts"] for r in results)
                assert "Base price $10.00 and tax $2.25." in text
                assert "Single item costs $19.99." in text
                assert r"\textdollar" not in text
                assert r"\(x + y = z\)" in text
        finally:
            if pipeline._doc_postprocessor is not None:
                pipeline._doc_postprocessor.close()
