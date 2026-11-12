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
from pathlib import Path

import numpy as np
import pytest

from paddlex.inference.serving.infra.config import AppConfig
from paddlex.inference.serving.infra.utils import image_array_to_bytes
from paddlex.inference.serving.schemas.paddleocr_vl import (
    InferRequest,
    RestructurePagesRequest,
)

HPS = Path(__file__).resolve().parents[2] / "deploy/hps"
MODEL_REPO = HPS / "sdk/pipelines/PaddleOCR-VL/server/model_repo"


class Prediction(dict):
    def __init__(self, text, image):
        super().__init__(parsing_res_list=[])
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
        (Path(directory) / "page.docx").write_bytes(self.text.encode())


class Pipeline:
    def __init__(self):
        self.inference_calls = []
        self.restructure_calls = []

    def __call__(self, images, **kwargs):
        return self.predict(images, **kwargs)

    def predict(self, images, **kwargs):
        self.inference_calls.append((len(images), kwargs))
        for image in images:
            yield Prediction(f"prediction {len(self.inference_calls)}", image)

    def restructure_pages(self, pages, **kwargs):
        self.restructure_calls.append(kwargs)
        if isinstance(pages[0], Prediction):
            return pages
        return [
            Prediction(page["res"]["text"], np.zeros((4, 5, 3), dtype=np.uint8))
            for page in pages
        ]


@pytest.fixture
def file_payload():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return base64.b64encode(image_array_to_bytes(image, ext=".png")).decode()


@pytest.fixture
def hps_model(monkeypatch):
    monkeypatch.syspath_prepend(str(HPS / "server_env/paddlex-hps-server/src"))

    def load_model(name):
        spec = importlib.util.spec_from_file_location(
            name, MODEL_REPO / name / "1/model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.TritonPythonModel()
        model.pipeline = Pipeline()
        model.app_config = AppConfig()
        model.context = {
            "file_storage": None,
            "return_urls": False,
            "url_expires_in": -1,
            "max_num_input_imgs": 10,
            "max_output_img_size": (2000, 2000),
        }
        return model

    return load_model


@pytest.fixture
def basic_client():
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
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
    assert first.markdown.text == "one:False:True"
    assert len(result.layoutParsingResults) == (1 if concatenate else 2)
    assert first.markdown.images == (
        {"a.png": "a", "b.png": "b"} if concatenate else {"a.png": "a"}
    )
