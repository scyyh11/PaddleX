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

"""Opt-in correction of PaddleX's single-process vLLM server call site.

The inspected PaddleX backend parses api-server-count but directly calls the
single-worker API coroutine. Clone that function and replace only the final
known call with vLLM's native CLI dispatcher. Keep the installed source, native
parser, merged defaults, plugins and backend configuration unchanged.
"""

import ast
import functools
import hashlib
import inspect
import textwrap

DISPATCH_NAME = "_paddlex_hps_native_serve_dispatch"


def dispatch_native_serve(args):
    from vllm.entrypoints.cli.serve import ServeSubcommand

    return ServeSubcommand.cmd(args)


def adapt_backend(original, *, dispatch=dispatch_native_serve):
    """Return a patched copy; reject unknown call sites instead of guessing."""
    if getattr(original, "_paddlex_hps_native_api_dispatch", False):
        return original
    if original.__closure__ or original.__name__ != "run_vllm_server":
        raise ValueError("Expected the module-level PaddleX run_vllm_server function")
    lines, start = inspect.getsourcelines(original)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Unexpected PaddleX backend function structure")
    definition = tree.body[0]
    if definition.decorator_list:
        raise ValueError("Cannot replace a decorated PaddleX backend function")
    expected = ast.parse("uvloop.run(run_server(args))", mode="eval").body
    matches = [
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected)
    ]
    if (
        len(matches) != 1
        or not isinstance(definition.body[-1], ast.Expr)
        or definition.body[-1].value is not matches[0]
    ):
        raise ValueError(
            "PaddleX backend no longer ends with the known single-worker call"
        )
    replacement = ast.Call(
        func=ast.Name(id=DISPATCH_NAME, ctx=ast.Load()),
        args=[ast.Name(id="args", ctx=ast.Load())],
        keywords=[],
    )
    definition.body[-1].value = ast.copy_location(replacement, matches[0])
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start - 1)
    namespace = dict(original.__globals__)
    if DISPATCH_NAME in namespace:
        raise ValueError("The native dispatcher helper name is already in use")
    namespace[DISPATCH_NAME] = dispatch
    exec(
        compile(tree, inspect.getsourcefile(original) or "<paddlex-backend>", "exec"),
        namespace,
    )
    patched = namespace[original.__name__]
    functools.update_wrapper(patched, original)
    patched._paddlex_hps_native_api_dispatch = True
    patched._paddlex_hps_dispatch_audit = {
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "changed_call": "uvloop.run(run_server(args)) -> ServeSubcommand.cmd(args)",
        "api_server_count_overridden": False,
    }
    return patched
