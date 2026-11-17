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

"""Execute the unchanged HPS entrypoint while wrapping its VLM console command."""

import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROGRAMS = ("paddleocr", "paddlex_genai_server")


def prepare_environment(environment, *, root=HERE):
    result = dict(environment)
    for program in PROGRAMS:
        original = shutil.which(program, path=result.get("PATH"))
        if original:
            if Path(original).resolve() == (root / "console_shim.py").resolve():
                raise ValueError("The HPS compose bridge must not wrap itself twice")
            result["PADDLEX_HPS_ORIGINAL_" + program.upper()] = original
    result["PATH"] = str(root / "bin") + os.pathsep + result.get("PATH", "")
    return result


def launcher_arguments(program, original_args, environment):
    if program not in PROGRAMS:
        raise ValueError("Unsupported console command")
    mode = environment.get("EAGLE3_MODE", "baseline")
    if mode not in ("baseline", "eagle3"):
        raise ValueError("EAGLE3_MODE must be baseline or eagle3")
    output = environment.get("EAGLE3_AUDIT_OUTPUT")
    if not output:
        raise ValueError("Set EAGLE3_AUDIT_OUTPUT to a new per-instance JSON path")
    result = [
        "--mode",
        mode,
        "--entrypoint",
        "paddleocr" if program == "paddleocr" else "paddlex",
        "--audit-output",
        output,
    ]
    if mode == "eagle3":
        draft = environment.get("EAGLE3_DRAFT_MODEL")
        if not draft:
            raise ValueError("Set EAGLE3_DRAFT_MODEL for EAGLE3 mode")
        result.extend(
            [
                "--draft-model",
                draft,
                "--num-speculative-tokens",
                environment.get("EAGLE3_K", "4"),
            ]
        )
    for name in (
        environment.get("PIPELINE_CONFIG", "/config/pipeline_config.yaml"),
        "/entrypoint.sh",
    ):
        if Path(name).is_file():
            result.extend(["--record-file", name])
    return [*result, "--", *original_args]


def console_main():
    program = Path(sys.argv[0]).name
    original_args = sys.argv[1:]
    if program == "paddleocr" and original_args[:1] != ["genai_server"]:
        original = os.environ.get("PADDLEX_HPS_ORIGINAL_PADDLEOCR")
        if not original:
            raise RuntimeError("The original paddleocr command is unavailable")
        os.execvpe(original, [original, *original_args], os.environ)
    sys.path.insert(0, str(HERE.parent))
    from hps_batch.launch import main as launch

    launch(launcher_arguments(program, original_args, os.environ))


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("Pass the existing container command after --")
    environment = prepare_environment(os.environ)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
