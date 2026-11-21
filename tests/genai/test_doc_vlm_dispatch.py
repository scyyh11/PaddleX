# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2] / "paddlex/inference"


def load_method(path, class_name, name, **scope):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    fn.decorator_list = []
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec")
    exec(code, scope)
    return scope[name]


def test_submit_each_crop_before_encoding_the_next_and_cancel_on_error():
    run = load_method(
        "models/doc_vlm/predictor.py",
        "DocVLMGenAIClientPredictor",
        "_genai_client_process",
    )
    pending = Mock()
    pending.done.return_value = False
    sent = []

    def encode(client, data, *args):
        assert len(data) == 1
        if data[0] == 2:
            assert sent == [1]
            raise ValueError("encode failed")
        return [(data[0], {})]

    def send(messages, **kwargs):
        sent.append(messages)
        return pending

    obj = SimpleNamespace(
        genai_client=SimpleNamespace(
            backend="vllm-server", create_chat_completion=send
        ),
        _doc_vlm_genai_build_request_specs=encode,
    )
    with pytest.raises(ValueError, match="encode failed"):
        run(obj, [1, 2], *([None] * 7))
    pending.cancel.assert_called_once_with()


@pytest.mark.parametrize("remote", [True, False])
def test_pixel_groups_only_overlap_for_remote_models(remote):
    run = load_method(
        "pipelines/paddleocr_vl/pipeline.py",
        "_PaddleOCRVLPipeline",
        "_paddleocr_vl_run_vl_recognition_batches",
        ThreadPoolExecutor=ThreadPoolExecutor,
    )
    gate = Barrier(2)

    def predict(requests, **kwargs):
        if remote:
            gate.wait(timeout=2)
        return iter([kwargs["skip_special_tokens"]])

    model = SimpleNamespace(model_name="PaddleOCR-VL-1.8-0.9B", predict=predict)
    obj = SimpleNamespace(_vl_rec_uses_server=remote, vl_rec_model=model)
    groups = {
        key: {"images": ["image"], "queries": ["query"]}
        for key in [(1, 2), (1, 2, "table")]
    }
    run(obj, groups, False, {})
    assert groups[(1, 2)]["vlm_results"] == [True]
    assert groups[(1, 2, "table")]["vlm_results"] == [False]
