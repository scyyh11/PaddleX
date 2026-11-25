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

"""Exercise the real serving adapters with a fake OCR-VL predictor."""

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from paddlex.utils.deps import is_serving_plugin_available

if not is_serving_plugin_available():
    pytest.skip("PaddleX serving dependencies are required", allow_module_level=True)

from paddlex.inference.serving.app_shared.document_export import postprocess_docx_bytes
from paddlex.inference.serving.app_shared.image_output import postprocess_image
from paddlex.inference.serving.app_shared.ocr_input import create_ocr_context
from paddlex.inference.serving.infra.config import AppConfig
from paddlex.inference.serving.infra.storage import (
    FileSystemStorage,
    FileSystemStorageConfig,
    InvalidStoragePathError,
)
from paddlex.inference.serving.infra.utils import (
    image_array_to_bytes,
    image_bytes_to_image,
)
from paddlex.inference.serving.schemas.paddleocr_vl import (
    InferRequest,
    RestructurePagesRequest,
)

HPS = Path(__file__).resolve().parents[2] / "deploy/hps"
SDK_PIPELINES = HPS / "sdk/pipelines"


class Prediction(dict):
    def __init__(self, text, image, *, restore_image=False):
        blocks = []
        if restore_image:
            blocks.append(SimpleNamespace(image={"path": "page.png", "img": None}))
        super().__init__(parsing_res_list=blocks)
        self.text = text
        self.img = {"layout": image}
        self.json = {"res": {"text": text, "input_path": "page.png", "page_index": 0}}

    def _to_markdown(self, *, pretty, show_formula_number):
        return {
            "markdown_texts": f"{self.text}:{pretty}:{show_formula_number}",
            "markdown_images": {"page.png": self.img["layout"]},
        }

    def _get_input_fn(self):
        return "page.png"

    def save_to_word(self, directory):
        text = self.text
        for block in self["parsing_res_list"]:
            if block.image["img"] is not None:
                text += f" [{block.image['img'].size}]"
        (Path(directory) / "page.docx").write_bytes(text.encode())


class Pipeline:
    def __init__(self):
        self.inference_calls = []
        self.restructure_calls = []

    def __call__(self, images, **kwargs):
        return self.predict(images, **kwargs)

    def predict(self, images, **kwargs):
        self.inference_calls.append((len(images), kwargs))
        if (
            kwargs.get("layout_deform_min_roll") == 99
            or kwargs.get("text_det_limit_side_len") == 99
        ):
            yield {"error": "Invalid model settings"}
            raise AssertionError("The serving adapter must stop after a pipeline error")
        for image in images:
            yield Prediction(f"prediction {len(self.inference_calls)}", image)

    def restructure_pages(self, pages, **kwargs):
        self.restructure_calls.append(kwargs)
        if isinstance(pages[0], Prediction):
            return pages
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        if kwargs["concatenate_pages"]:
            text = " + ".join(page["res"]["text"] for page in pages)
            return [Prediction(text, image, restore_image=True)]
        return [
            Prediction(page["res"]["text"], image, restore_image=True) for page in pages
        ]


@pytest.fixture
def file_payload():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return base64.b64encode(image_array_to_bytes(image, ext=".png")).decode()


@pytest.fixture
def hps_model(monkeypatch):
    monkeypatch.syspath_prepend(str(HPS / "server_env/paddlex-hps-server/src"))

    def load_model(name, pipeline="PaddleOCR-VL"):
        model_file = (
            SDK_PIPELINES / pipeline / "server/model_repo" / name / "1/model.py"
        )
        spec = importlib.util.spec_from_file_location(name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.TritonPythonModel()
        model.pipeline = Pipeline()
        model.app_config = AppConfig()
        model.context = create_ocr_context(model.app_config)
        return model

    return load_model


@pytest.fixture
def basic_client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from paddlex.inference.serving.basic_serving._pipeline_apps.paddleocr_vl import (
        create_pipeline_app,
    )

    pipeline = Pipeline()
    with TestClient(create_pipeline_app(pipeline, AppConfig())) as client:
        yield client, pipeline


@pytest.mark.parametrize("visualize", [False, True])
def test_layout_parsing_matches_between_adapters(
    basic_client, hps_model, file_payload, visualize
):
    payload = {
        "file": file_payload,
        "fileType": 1,
        "useLayoutDeform": True,
        "layoutDeformConfThresh": 0.8,
        "layoutDeformMinRoll": 2,
        "layoutDeformMinShear": 3,
        "layoutDeformMinKeystone": 4,
        "restructurePages": True,
        "returnMarkdownImages": visualize,
        "visualize": visualize,
        "outputFormats": ["docx"],
        "logId": "test",
    }
    client, basic_pipeline = basic_client
    response = client.post("/layout-parsing", json=payload)
    assert response.status_code == 200

    hps = hps_model("layout-parsing")
    (result,) = hps.run_batch([InferRequest(**payload)], ["test"], "batch")
    assert response.json()["result"] == result.model_dump(
        mode="json", exclude_none=True
    )
    assert basic_pipeline.inference_calls == hps.pipeline.inference_calls
    _, kwargs = hps.pipeline.inference_calls[0]
    assert kwargs["use_layout_deform"] is True
    assert kwargs["layout_deform_conf_thresh"] == 0.8
    assert kwargs["layout_deform_min_roll"] == 2
    assert kwargs["layout_deform_min_shear"] == 3
    assert kwargs["layout_deform_min_keystone"] == 4
    assert basic_pipeline.restructure_calls == hps.pipeline.restructure_calls
    page = result.layoutParsingResults[0]
    assert page.prunedResult == {"text": "prediction 1"}
    assert bool(page.inputImage) is visualize
    assert bool(page.markdown.images) is visualize
    assert base64.b64decode(page.exports.docx.content) == b"prediction 1"


def test_hps_batch_uses_inference_parameters_to_group_requests(hps_model, file_payload):
    payload = {
        "file": file_payload,
        "fileType": 1,
        "useLayoutDeform": True,
        "vlmExtraArgs": {"custom": {"option": 1, "flag": True}},
    }
    requests = [
        InferRequest(**payload, layoutDeformMinRoll=1),
        InferRequest(**payload, layoutDeformMinRoll=2),
        InferRequest(
            **{
                **payload,
                "vlmExtraArgs": {"custom": {"flag": True, "option": 1}},
            },
            layoutDeformMinRoll=1,
            returnMarkdownImages=False,
        ),
        InferRequest(
            **{**payload, "vlmExtraArgs": {"custom": [["flag", True], ["option", 1]]}},
            layoutDeformMinRoll=1,
        ),
    ]
    model = hps_model("layout-parsing")
    results = model.run_batch(requests, ["a", "b", "c", "d"], "batch")
    assert [
        (count, args["layout_deform_min_roll"])
        for count, args in model.pipeline.inference_calls
    ] == [
        (2, 1),
        (1, 2),
        (1, 1),
    ]
    assert [
        result.layoutParsingResults[0].prunedResult["text"] for result in results
    ] == [
        "prediction 1",
        "prediction 2",
        "prediction 1",
        "prediction 3",
    ]
    assert results[2].layoutParsingResults[0].markdown.images is None


@pytest.mark.parametrize("concatenate", [False, True])
def test_restructure_pages_matches_between_adapters(
    basic_client, hps_model, concatenate
):
    payload = {
        "pages": [
            {"prunedResult": {"text": "one"}, "markdownImages": {"a.png": "a"}},
            {"prunedResult": {"text": "two"}, "markdownImages": {"b.png": "b"}},
        ],
        "concatenatePages": concatenate,
        "mergeTables": False,
        "relevelTitles": False,
        "prettifyMarkdown": False,
        "showFormulaNumber": True,
        "outputFormats": ["docx"],
        "logId": "test",
    }
    client, basic_pipeline = basic_client
    response = client.post("/restructure-pages", json=payload)
    assert response.status_code == 200

    hps = hps_model("restructure-pages")
    result = hps.run(RestructurePagesRequest(**payload), "test")
    assert response.json()["result"] == result.model_dump(
        mode="json", exclude_none=True
    )
    assert basic_pipeline.restructure_calls == hps.pipeline.restructure_calls
    first = result.layoutParsingResults[0]
    text = "one + two" if concatenate else "one"
    assert first.markdown.text == f"{text}:False:True"
    assert base64.b64decode(first.exports.docx.content).decode() == text
    assert len(result.layoutParsingResults) == (1 if concatenate else 2)
    assert first.markdown.images == (
        {"a.png": "a", "b.png": "b"} if concatenate else {"a.png": "a"}
    )


def test_hps_pipeline_error_only_affects_its_request_group(
    basic_client, hps_model, file_payload
):
    payload = {"file": file_payload, "fileType": 1, "useLayoutDeform": True}
    invalid = {**payload, "layoutDeformMinRoll": 99}
    valid = {**payload, "layoutDeformMinRoll": 2}

    client, _ = basic_client
    response = client.post("/layout-parsing", json={**invalid, "logId": "first"})
    assert response.status_code == 500

    model = hps_model("layout-parsing")
    failed, success, same_group = model.run_batch(
        [InferRequest(**p) for p in (invalid, valid, invalid)],
        ["first", "second", "third"],
        "batch",
    )
    assert failed.model_dump() == response.json()
    assert success.layoutParsingResults[0].prunedResult == {"text": "prediction 2"}
    assert same_group.errorCode == 500
    assert same_group.logId == "third"


@pytest.mark.parametrize("image_source", ["base64", "data", "url"])
def test_docx_restructure_accepts_only_inline_images(
    basic_client, hps_model, file_payload, image_source, monkeypatch
):
    def reject_network(*args, **kwargs):
        raise AssertionError("DOCX image restoration must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    images = {
        "base64": file_payload,
        "data": f"data:image/png;base64,{file_payload}",
        "url": "https://example.invalid/page.png",
    }
    payload = {
        "pages": [
            {
                "prunedResult": {"text": "one"},
                "markdownImages": {"page.png": images[image_source]},
            }
        ],
        "outputFormats": ["docx"],
        "logId": "test",
    }
    client, _ = basic_client
    response = client.post("/restructure-pages", json=payload)
    hps = hps_model("restructure-pages")
    result = hps.run(RestructurePagesRequest(**payload), "test")
    if image_source == "url":
        assert response.status_code == 422
        assert response.json() == result.model_dump()
        assert "inline Base64" in result.errorMsg
    else:
        assert response.status_code == 200
        assert response.json()["result"] == result.model_dump(
            mode="json", exclude_none=True
        )
        docx = result.layoutParsingResults[0].exports.docx.content
        assert base64.b64decode(docx) == b"one [(5, 4)]"


@pytest.mark.parametrize(
    ("image_shape", "limits", "expected"),
    [((60, 100), (25, 70), (25, 41)), ((60, 100), (50, 35), (21, 35))],
)
def test_image_output_fits_rectangular_limits(image_shape, limits, expected):
    image = np.zeros((*image_shape, 3), dtype=np.uint8)
    encoded = postprocess_image(image, "test", "page.png", max_img_size=limits)
    resized = image_bytes_to_image(base64.b64decode(encoded))
    assert (resized.height, resized.width) == expected


def test_image_and_docx_output_use_the_same_storage_policy():
    class Storage:
        def __init__(self):
            self.items = {}

        def set(self, key, data):
            self.items[key] = data

        def get_url(self, key, expires_in):
            return f"https://storage.example.invalid/{key}?expires={expires_in}"

    storage = Storage()
    options = {"file_storage": storage, "return_url": True, "url_expires_in": 17}
    image_url = postprocess_image(
        np.zeros((4, 5, 3), dtype=np.uint8), "test", "page.png", **options
    )
    docx_url = postprocess_docx_bytes(b"docx", "test", "page.docx", **options)
    assert image_url.endswith("/test/page.png?expires=17")
    assert docx_url.endswith("/test/page.docx?expires=17")
    assert image_bytes_to_image(storage.items["test/page.png"]).size == (5, 4)
    assert storage.items["test/page.docx"] == b"docx"


@pytest.mark.parametrize(
    ("file", "detail"),
    [
        ("aGVsbG8=", "File type cannot be determined"),
        ("https://example.invalid/notes.txt", "Unsupported file type"),
    ],
)
def test_layout_parsing_reports_the_same_input_type_errors(
    basic_client, hps_model, file, detail
):
    payload = {"file": file, "logId": "input"}
    client, _ = basic_client
    response = client.post("/layout-parsing", json=payload)
    model = hps_model("layout-parsing")
    (result,) = model.run_batch([InferRequest(**payload)], ["input"], "batch")
    assert response.status_code == 422
    assert response.json() == result.model_dump()
    assert result.errorMsg == detail


@pytest.mark.parametrize(
    ("pipeline", "route"), [("OCR", "ocr"), ("PP-StructureV3", "layout-parsing")]
)
def test_other_hps_models_group_by_effective_inference_arguments(
    hps_model, file_payload, pipeline, route
):
    model = hps_model(route, pipeline)
    model._postprocess = lambda images, info, visualize, preds, log_id, req: preds
    request_type = model.get_input_model_type()
    requests = [
        request_type(file=file_payload, fileType=1, textDetLimitSideLen=size)
        for size in (99, 64, 128, 64)
    ]
    error, *results = model.run_batch(requests, ["bad", "a", "b", "c"], "batch")
    assert (error.errorCode, error.errorMsg, error.logId) == (
        500,
        "Invalid model settings",
        "bad",
    )
    assert [
        (count, kwargs["text_det_limit_side_len"])
        for count, kwargs in model.pipeline.inference_calls
    ] == [(1, 99), (2, 64), (1, 128)]
    assert [preds[0].text for preds in results] == [
        "prediction 2",
        "prediction 3",
        "prediction 2",
    ]


def test_file_storage_writes_markdown_images_in_nested_directories(
    basic_client, file_payload, tmp_path
):
    client, _ = basic_client
    storage = FileSystemStorage(FileSystemStorageConfig(directory=tmp_path))
    client.app.state.context.extra["file_storage"] = storage
    payload = {
        "file": file_payload,
        "fileType": 1,
        "returnMarkdownImages": True,
        "visualize": False,
        "outputFormats": ["docx"],
    }

    rejected = client.post("/layout-parsing", json={**payload, "logId": "two/parts"})
    assert rejected.status_code == 422
    assert list(tmp_path.iterdir()) == []

    response = client.post("/layout-parsing", json={**payload, "logId": "single-id"})
    assert response.status_code == 200
    (page,) = response.json()["result"]["layoutParsingResults"]
    assert storage.get("single-id/markdown_0/page.png") == base64.b64decode(
        page["markdown"]["images"]["page.png"]
    )
    assert storage.get("single-id/page.docx") == b"prediction 2"


def test_filesystem_storage_rejects_a_path_outside_its_root(tmp_path):
    root = tmp_path / "storage"
    storage = FileSystemStorage(FileSystemStorageConfig(directory=root))
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (root / "link").symlink_to(sibling, target_is_directory=True)

    with pytest.raises(InvalidStoragePathError):
        storage.get("link/image.png")


def test_hps_rejects_invalid_storage_log_id_without_failing_other_requests(
    hps_model, file_payload, tmp_path, monkeypatch
):
    from paddlex_hps_server import base_model

    model = hps_model("layout-parsing")
    model._id = "test-model"
    model.context["file_storage"] = FileSystemStorage(
        FileSystemStorageConfig(directory=tmp_path)
    )
    monkeypatch.setattr(
        base_model,
        "pb_utils",
        SimpleNamespace(
            get_input_tensor_by_name=lambda request, name: SimpleNamespace(
                as_numpy=lambda: request
            ),
            Tensor=lambda name, value: value,
            InferenceResponse=lambda output_tensors: output_tensors[0],
        ),
    )
    requests = [
        np.array(
            [
                [
                    json.dumps(
                        {
                            "file": file_payload,
                            "fileType": 1,
                            "logId": log_id,
                            "visualize": False,
                        }
                    ).encode()
                ]
            ],
            dtype=object,
        )
        for log_id in ("two/parts", "single-id")
    ]

    rejected, succeeded = [
        json.loads(output[0, 0]) for output in model.execute(requests)
    ]
    assert rejected["errorCode"] == 422
    assert succeeded["errorCode"] == 0
    assert (tmp_path / "single-id/markdown_0/page.png").is_file()
    assert {entry.name for entry in tmp_path.iterdir()} == {"single-id"}
