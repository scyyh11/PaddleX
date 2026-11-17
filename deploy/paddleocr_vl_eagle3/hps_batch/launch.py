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

"""Run the existing PaddleX or vLLM server with an optional EAGLE3 plugin.

No replacement Python environment, request lock, dependency installation or
CUDA-specific kernel is introduced. Pass the original server arguments after
``--``. Both modes preserve the same base configuration and existing plugins.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PLUGIN_NAME = "paddlex_eagle3_hps"
ENABLE_ENV = "PADDLEX_HPS_EAGLE3_ENABLED"
VISION_ENV = "PADDLEX_HPS_BATCH_VISION"
API_DISPATCH_ENV = "PADDLEX_HPS_NATIVE_API_DISPATCH"
CPU_GRID_ENV = "PADDLEX_HPS_CPU_GRID"
AUDIT_DIR_ENV = "PADDLEX_HPS_EAGLE3_AUDIT_DIR"
SECRET_KEYS = {"api-key", "hf-token", "token", "password", "secret"}


def _is_secret(key):
    key = str(key).lstrip("-").replace("_", "-").lower()
    return key in SECRET_KEYS or key.endswith(("-token", "-password", "-secret"))


def redact(value):
    if isinstance(value, dict):
        return {
            k: "<redacted>" if _is_secret(k) else redact(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def redact_argv(argv):
    result = []
    hide_next = False
    for value in argv:
        if hide_next:
            result.append("<redacted>")
            hide_next = False
        elif value.startswith("--") and "=" in value:
            name, item = value.split("=", 1)
            result.append(name + "=" + ("<redacted>" if _is_secret(name) else item))
        else:
            result.append(value)
            hide_next = value.startswith("--") and _is_secret(value)
    return result


def configure_plugin_environment(
    environment, *, enabled, batch_vision=False, cpu_grid=False, root=HERE
):
    """Return a copy; preserve the caller's plugin policy except adding ours."""
    result = dict(environment)
    policy = result.get("VLLM_PLUGINS")
    if policy is not None:
        names = [name.strip() for name in policy.split(",") if name.strip()]
        if not names:
            raise ValueError(
                "VLLM_PLUGINS disables all plugins. Restore the existing PaddleX "
                "plugin policy before using this integration; it will not be overridden."
            )
        if PLUGIN_NAME not in names:
            names.append(PLUGIN_NAME)
        result["VLLM_PLUGINS"] = ",".join(names)
    # An unset VLLM_PLUGINS retains native discovery of all installed plugins.
    result[ENABLE_ENV] = "1" if enabled else "0"
    result[VISION_ENV] = "1" if batch_vision else "0"
    result[CPU_GRID_ENV] = "1" if cpu_grid else "0"
    paths = [str(root.parent.resolve()), str((root / "plugin_site").resolve())]
    inherited = [
        part for part in result.get("PYTHONPATH", "").split(os.pathsep) if part
    ]
    result["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + inherited))
    return result


def speculative_config(args):
    if args.mode == "baseline":
        return None
    return {
        "method": "eagle3",
        "model": str(args.draft_model.expanduser().resolve()),
        "num_speculative_tokens": args.num_speculative_tokens,
    }


def merge_speculative_config(config, spec):
    """Preserve every batching/cache/precision option; reject hidden speculation."""
    if "speculative-config" in config or "speculative_config" in config:
        raise ValueError(
            "The shared base backend config must not contain speculative-config; "
            "the mode switch owns this one difference."
        )
    merged = dict(config)
    if spec is not None:
        merged["speculative-config"] = json.dumps(spec, separators=(",", ":"))
    return merged


def native_vllm_argv(argv, spec):
    if not argv or argv[0] != "serve":
        raise ValueError(
            "The vLLM entrypoint requires original arguments beginning with serve"
        )
    if any(
        arg.split("=", 1)[0] in {"--speculative-config", "--speculative_config"}
        for arg in argv
    ):
        raise ValueError(
            "Remove speculative-config from the shared base server arguments"
        )
    return list(argv) + (
        ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
        if spec is not None
        else []
    )


def server_file_arguments(argv):
    """Hash caller-supplied configuration/template files without interpreting them."""
    options = {"--config", "--backend_config", "--backend-config", "--chat-template"}
    files = []
    for index, arg in enumerate(argv):
        name, separator, value = arg.partition("=")
        if name not in options:
            continue
        if not separator:
            if index + 1 >= len(argv):
                raise ValueError("Missing value for " + name)
            value = argv[index + 1]
        if name == "--chat-template" and ("{{" in value or "{%" in value):
            continue
        path = Path(value)
        if not path.is_file():
            if name == "--chat-template":
                # Native vLLM also accepts an inline Jinja template, already
                # preserved in original_server_argv.
                continue
            raise ValueError(
                "Server configuration/template file does not exist: " + value
            )
        files.append(path)
    return files


def validate_native_configuration(argv):
    # Import the image's existing YAML parser only at real native startup, never
    # for dry-run. vLLM also accepts --config YAML; do not hide speculation there.
    for index, arg in enumerate(argv):
        name, separator, value = arg.partition("=")
        if name != "--config":
            continue
        if not separator:
            value = argv[index + 1]
        import yaml

        config = yaml.safe_load(Path(value).read_text())
        if not isinstance(config, dict):
            raise ValueError("The native vLLM YAML config must be a mapping")
        merge_speculative_config(config, None)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_audit(args, environment, *, root=HERE):
    versions = {}
    for name in (
        "paddlex",
        "paddleocr",
        "paddlex-genai-vllm-server",
        "vllm",
        "torch",
        "transformers",
        "pydantic",
        "nvidia-cublas-cu12",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    plugins = []
    for entry in importlib.metadata.entry_points(group="vllm.general_plugins"):
        plugins.append(
            {
                "name": entry.name,
                "value": entry.value,
                "distribution": entry.dist.metadata.get("Name") if entry.dist else None,
                "version": entry.dist.version if entry.dist else None,
            }
        )
    sources = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and (path.suffix in {".py", ".txt", ".yaml"} or path.name == "METADATA")
        and "__pycache__" not in path.parts
    }
    return {
        "format_version": 1,
        "status": "planned",
        "created_unix": time.time(),
        "mode": args.mode,
        "entrypoint": args.entrypoint,
        "image_reference": args.image_reference,
        "image_digest_verified": False,
        "original_server_argv": redact_argv(args.server_args),
        "speculative_config": speculative_config(args),
        "package_versions": versions,
        "discovered_general_plugins": sorted(plugins, key=lambda row: row["name"]),
        "plugin_allowlist": environment.get("VLLM_PLUGINS"),
        "adapter_enabled": environment[ENABLE_ENV] == "1",
        "batch_vision_enabled": environment[VISION_ENV] == "1",
        "cpu_grid_enabled": environment[CPU_GRID_ENV] == "1",
        "native_api_dispatch_enabled": args.native_api_dispatch,
        "source_sha256": sources,
        "explicit_source_files": {
            str(path.resolve()): _sha(path)
            for path in [*args.record_file, *server_file_arguments(args.server_args)]
        },
        "notes": [
            "Plugin discovery is not proof that a particular model or vision implementation was selected.",
            "Image reference is caller-supplied; verify the running container digest externally.",
            "This launcher does not change client concurrency, pipeline queues, max_num_seqs, batching, dtype or CUDA libraries.",
        ],
    }


def write_audit(path, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(redact(audit), file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if os.environ.get(CPU_GRID_ENV, "0") not in ("0", "1"):
        parser.error(CPU_GRID_ENV + " must be 0 or 1")
    parser.add_argument("--mode", choices=("baseline", "eagle3"), required=True)
    parser.add_argument(
        "--entrypoint", choices=("paddlex", "paddleocr", "vllm"), default="paddlex"
    )
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--image-reference", default=os.environ.get("PADDLEX_HPS_IMAGE_REFERENCE")
    )
    parser.add_argument("--record-file", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--batch-vision",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get(VISION_ENV, "0") == "1",
        help="Opt in to the native batch vision bridge; set identically in both comparison groups",
    )
    parser.add_argument(
        "--cpu-grid",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get(CPU_GRID_ENV, "0") == "1",
        help="Keep audited PaddleOCR image-grid metadata on CPU; enable identically in EAGLE3 comparison groups",
    )
    parser.add_argument(
        "--native-api-dispatch",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get(API_DISPATCH_ENV, "0") == "1",
        help="Honor the existing PaddleX api-server-count using vLLM's native CLI dispatcher; enable identically in both comparison groups",
    )
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.cpu_grid and args.batch_vision:
        parser.error(
            "CPU-grid has not been validated with the native batch vision bridge"
        )
    if args.server_args[:1] == ["--"]:
        args.server_args = args.server_args[1:]
    if not args.server_args:
        parser.error("Pass the unchanged existing server arguments after --")
    if args.mode == "eagle3" and (
        args.draft_model is None or not args.draft_model.is_dir()
    ):
        parser.error("EAGLE3 requires an existing --draft-model directory")
    if args.mode == "baseline" and args.draft_model is not None:
        parser.error("The baseline must not receive a draft model")
    if not 1 <= args.num_speculative_tokens <= 16:
        parser.error("num-speculative-tokens must be in 1..16")
    if args.audit_output.exists():
        parser.error("Use a new audit-output path for each service instance")
    if any(not path.is_file() for path in args.record_file):
        parser.error("Every record-file must exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    environment = configure_plugin_environment(
        os.environ,
        enabled=args.mode == "eagle3",
        batch_vision=args.batch_vision,
        cpu_grid=args.cpu_grid,
    )
    environment[AUDIT_DIR_ENV] = str(
        args.audit_output.resolve().with_suffix(".workers")
    )
    environment[API_DISPATCH_ENV] = "1" if args.native_api_dispatch else "0"
    os.environ.update(environment)
    for path in reversed(environment["PYTHONPATH"].split(os.pathsep)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = speculative_config(args)
    audit = runtime_audit(args, environment)
    write_audit(args.audit_output, audit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "audit": str(args.audit_output),
                    "adapter_enabled": spec is not None,
                }
            )
        )
        return
    if not any(
        item["name"] == PLUGIN_NAME for item in audit["discovered_general_plugins"]
    ):
        raise RuntimeError("The mounted adapter entry point was not discovered")
    if args.entrypoint == "vllm":
        native_args = native_vllm_argv(args.server_args, spec)
        validate_native_configuration(args.server_args)
        audit["status"] = "entering_native_server"
        audit["effective_server_argv"] = redact_argv(native_args)
        write_audit(args.audit_output, audit)
        from vllm.entrypoints.cli.main import main as native_main

        sys.argv = ["vllm", *native_args]
        native_main()
    else:
        from paddlex.inference.genai import server
        from paddlex.inference.genai.backends import vllm as backend

        original = backend.run_vllm_server
        if args.native_api_dispatch:
            from hps_batch.native_api_dispatch import adapt_backend

            original = adapt_backend(original)
            audit["native_api_dispatch"] = original._paddlex_hps_dispatch_audit

        def run_server(host, port, model_name, model_dir, config, chat_template_path):
            effective = merge_speculative_config(config, spec)
            audit.update(
                status="entering_native_server",
                target_model=model_name,
                target_directory=str(model_dir),
                shared_backend_config=redact(config),
                effective_backend_config=redact(effective),
                host=host,
                port=port,
            )
            if chat_template_path:
                audit["chat_template_sha256"] = _sha(Path(chat_template_path))
            write_audit(args.audit_output, audit)
            return original(
                host, port, model_name, model_dir, effective, chat_template_path
            )

        server_args = args.server_args
        if args.entrypoint == "paddleocr":
            if server_args[:1] != ["genai_server"]:
                raise ValueError(
                    "The PaddleOCR entrypoint requires genai_server arguments"
                )
            server_args = server_args[1:]
        parsed = server.get_arg_parser().parse_args(server_args)
        if parsed.backend != "vllm":
            raise ValueError("This integration requires the existing vLLM backend")
        backend.run_vllm_server = run_server
        if args.entrypoint == "paddleocr":
            from paddleocr._cli import main as paddleocr_main

            sys.argv = ["paddleocr", *args.server_args]
            paddleocr_main()
        else:
            server.run_genai_server(parsed)


if __name__ == "__main__":
    main()
