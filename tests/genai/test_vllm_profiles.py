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


import importlib.metadata
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(monkeypatch):
    # Import just the launcher and config translator; no Paddle, Torch, GPU or
    # installed vLLM is needed to exercise dispatch and registration in CI.
    for name in (
        "paddlex",
        "paddlex.inference",
        "paddlex.inference.genai",
        "paddlex.inference.genai.backends",
        "paddlex.inference.genai.configs",
        "paddlex.utils",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(sys.modules["paddlex.utils"], "logging", Mock(), raising=False)
    version = Mock(return_value="0.33.0")
    genai = "paddlex.inference.genai"
    deps = ModuleType("paddlex.utils.deps")
    deps.is_vllm_legacy_version = lambda: version() == "0.19.0"
    deps.is_vllm_performance_version = lambda: version() == "0.33.0"
    deps.is_genai_engine_plugin_available = Mock(return_value=True)
    deps.require_genai_engine_plugin = Mock()
    monkeypatch.setitem(sys.modules, deps.__name__, deps)
    models = ModuleType(f"{genai}.models")
    models.ALL_MODEL_INFO = ["PaddleOCR-VL-1.8-0.9B"]
    models.get_model_components = Mock()
    models.is_integrated_model_available = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, models.__name__, models)
    for name in (f"{genai}.configs.utils", f"{genai}.backends.vllm"):
        monkeypatch.setitem(sys.modules, name, None)
    _load(f"{genai}.configs.utils", "paddlex/inference/genai/configs/utils.py")
    mod = _load(f"{genai}.backends.vllm", "paddlex/inference/genai/backends/vllm.py")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0")),
    )
    return mod, version


def _launch(mod, config):
    mod.run_vllm_server(
        "127.0.0.1", 8080, "PaddleOCR-VL-1.8-0.9B", "/models/paddleocr-vl", config, None
    )


def test_launches_native_multi_frontend_serve(backend, monkeypatch):
    mod, _ = backend
    execute = Mock()
    monkeypatch.setattr(mod.os, "execv", execute)
    cfg = {
        "paddlex-vision-rope": True,
        "api-server-count": 2,
        "enable-prefix-caching": False,
    }
    _launch(mod, cfg)
    executable, argv = execute.call_args.args
    assert executable == sys.executable
    assert argv[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    assert argv[argv.index("--api-server-count") + 1] == "2"
    assert "--no-enable-prefix-caching" in argv
    assert "--enable-prefix-caching" not in argv
    assert "--paddlex-vision-rope" not in argv
    args = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert args == {"architectures": [mod._PERFORMANCE_ARCH]}
    assert cfg["paddlex-vision-rope"] is True


def test_stock_033_does_not_select_paddlex_arch(backend, monkeypatch):
    mod, _ = backend
    execute = Mock()
    monkeypatch.setattr(mod.os, "execv", execute)
    _launch(mod, {})
    assert "--hf-overrides" not in execute.call_args.args[1]


@pytest.mark.parametrize(
    "cfg",
    [
        {"paddlex-vision-rope": "false"},
        {"paddlex-vision-rope": True, "hf-overrides": "{}"},
    ],
)
def test_rejects_ambiguous_rope_options(backend, cfg):
    mod, _ = backend
    with pytest.raises(ValueError, match="paddlex-vision-rope|hf-overrides"):
        _launch(mod, cfg)


def test_019_native_cli_and_optional_vision_batching(backend, monkeypatch):
    mod, version = backend
    version.return_value = "0.19.0"
    execute = Mock()
    monkeypatch.setattr(mod.os, "execv", execute)
    _launch(mod, {"api-server-count": 1})
    assert "--hf-overrides" not in execute.call_args.args[1]
    _launch(mod, {"paddlex-vision-batching": True, "api-server-count": 2})
    argv = execute.call_args.args[1]
    assert argv[3] == "serve"
    assert argv[argv.index("--api-server-count") + 1] == "2"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "architectures": [mod._LEGACY_BATCHED_ARCH]
    }
    with pytest.raises(ValueError, match="vLLM 0.33.0"):
        _launch(mod, {"paddlex-vision-rope": True})
    with pytest.raises(ValueError, match="prefill-schedule-interval"):
        _launch(mod, {"prefill_schedule_interval": 8})


def test_033_already_has_native_batching(backend, monkeypatch):
    mod, _ = backend
    execute = Mock()
    monkeypatch.setattr(mod.os, "execv", execute)
    _launch(mod, {"paddlex-vision-batching": True})
    assert "--hf-overrides" not in execute.call_args.args[1]
    _launch(mod, {"paddlex-vision-batching": True, "paddlex-vision-rope": True})
    argv = execute.call_args.args[1]
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "architectures": [mod._PERFORMANCE_ARCH]
    }


def test_registration_is_lazy_and_only_for_033(backend, monkeypatch):
    mod, version = backend
    registry = SimpleNamespace(register_model=Mock())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    mod.register_models()
    arch, loader = registry.register_model.call_args.args
    assert arch == mod._PERFORMANCE_ARCH
    assert loader.endswith("_vllm033:" + arch)
    registry.register_model.reset_mock()
    version.return_value = "0.19.0"
    mod.register_models()
    assert registry.register_model.call_args.args[0] == mod._LEGACY_BATCHED_ARCH
    assert "_vllm019:" in registry.register_model.call_args.args[1]
    registry.register_model.reset_mock()
    version.return_value = "0.28.0"
    mod.register_models()
    registry.register_model.assert_not_called()


def test_release_extras_are_separately_pinned(monkeypatch):
    extras = runpy.run_path(str(ROOT / "setup.py"))["extras"]()
    assert "vllm == 0.19.0" in extras["genai-vllm-server"]
    assert "torch == 2.10.0" in extras["genai-vllm-server"]
    assert "vllm == 0.33.0" in extras["genai-vllm-server-033"]
    assert "torch == 2.14.0" in extras["genai-vllm-server-033"]
    assert "transformers == 5.20.2" in extras["genai-vllm-server-033"]
    # Recreate installed wheel metadata just for the dependency selector.
    names = ["genai-vllm-server", "genai-vllm-server-033"]
    monkeypatch.setattr(
        importlib.metadata,
        "metadata",
        lambda _: SimpleNamespace(get_all=lambda k, d: names),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda _: [f"{r}; extra == '{n}'" for n in names for r in extras[n]],
    )
    pkg = ModuleType("paddlex.utils")
    pkg.logging = Mock()
    monkeypatch.setitem(sys.modules, "paddlex.utils", pkg)
    monkeypatch.setitem(sys.modules, "paddlex.utils.deps", None)
    deps = _load("paddlex.utils.deps", "paddlex/utils/deps.py")
    monkeypatch.setattr(deps, "get_dep_version", lambda _: "0.33.0+cu130")
    assert deps.is_vllm_performance_version()
    requires = {
        Requirement(s).name: str(Requirement(s).specifier)
        for s in deps.get_genai_dep_specs("vllm-server-033")
    }
    assert requires["vllm"] == "==0.33.0"
    monkeypatch.setattr(deps, "get_dep_version", lambda _: "0.19.0")
    assert not deps.is_vllm_performance_version()
