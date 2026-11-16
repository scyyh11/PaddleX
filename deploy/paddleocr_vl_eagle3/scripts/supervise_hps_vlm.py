#!/usr/bin/env python3
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

"""Supervise one explicitly configured local HPS worker and its /health/ready route."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Timing:
    poll: float = 2
    http: float = 3
    startup: float = 240
    slow_health: float = 45
    terminate: float = 10
    initial_backoff: float = 1
    max_backoff: float = 30
    stable: float = 120


@contextmanager
def exclusive(path):
    # Keep lock inodes across exits, so waiting supervisors cannot lock
    # different files with the same name.
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another supervisor already owns {path}") from None
        yield


def atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n")
    tmp.replace(path)


def check_free_port(port):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            s.listen(1)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot claim 127.0.0.1:{port}: {exc.strerror}"
            ) from exc


def owns_listener(pid, port):
    """Verify the socket belongs to our one-worker uvicorn child before probing it."""
    endpoints = {f"{addr}:{port:04X}" for addr in ("0100007F", "00000000")}
    inodes = {
        fields[9]
        for row in Path("/proc/net/tcp").read_text().splitlines()[1:]
        if (fields := row.split()) and fields[1] in endpoints and fields[3] == "0A"
    }
    if not inodes:
        return False
    sockets = {f"socket:[{inode}]" for inode in inodes}
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if os.readlink(fd) in sockets:
                return True
        except FileNotFoundError:
            continue
    raise RuntimeError(
        f"Port {port} is occupied by a different process; refusing to adopt it"
    )


def probe(port, timeout):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/health/ready")
        response = conn.getresponse()
        body = response.read(65537)
        if len(body) > 65536:
            raise RuntimeError("Unexpectedly large health response")
        return response.status, json.loads(body)
    finally:
        conn.close()


def living_group_member(pgid):
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _, group, *_ = stat.read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(group) == pgid and state not in (b"Z", b"X"):
            return True
    return False


def stop_child(child, grace):
    # The group leader stays unreaped until both signals have been sent.
    # Its PID therefore cannot be recycled into an unrelated process group.
    try:
        os.killpg(child.pid, signal.SIGTERM)
        end = time.monotonic() + grace
        while living_group_member(child.pid):
            if time.monotonic() >= end:
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(min(0.1, grace))
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def watch(child, args, stop, timing, record, previous):
    start = time.monotonic()
    ready_since = None
    unhealthy_since = None
    fatal_streak = 0
    instance = None
    while not stop.is_set():
        if os.waitid(os.P_PID, child.pid, os.WNOHANG | os.WEXITED | os.WNOWAIT):
            return instance, ready_since, "worker exited"
        try:
            if not owns_listener(child.pid, args.port):
                raise ConnectionRefusedError("no listener")
            status, body = probe(args.port, timing.http)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            status, body = 0, {}
            detail = type(exc).__name__
        else:
            detail = f"HTTP {status}"
        now = time.monotonic()
        if status == 200:
            if not isinstance(body, dict):
                raise RuntimeError("Invalid HPS health response")
            new_instance = body.get("instance_id")
            if (
                body.get("status") != "ready"
                or body.get("mode") != args.mode
                or not isinstance(new_instance, str)
                or not new_instance
            ):
                raise RuntimeError(
                    "Health response does not identify the configured HPS worker"
                )
            if args.mode == "eagle3-recovery":
                worker = body.get("speculative_decoding", {}).get("worker", {})
                if (
                    worker.get("method") != "eagle3"
                    or worker.get("target") != "PaddleOCRVLForConditionalGeneration"
                ):
                    raise RuntimeError(
                        "Recovery readiness did not attest to the EAGLE3 target"
                    )
            requested_cap = getattr(args, "hps_max_completion_tokens", None)
            if (
                requested_cap is not None
                and body.get("max_completion_tokens") != requested_cap
            ):
                raise RuntimeError(
                    "HPS readiness reported a different completion limit"
                )
            if new_instance == previous or (
                instance is not None and new_instance != instance
            ):
                raise RuntimeError(
                    "Unexpected HPS instance identity on the owned socket"
                )
            if instance is None:
                instance, ready_since = new_instance, now
                record(
                    "ready",
                    child.pid,
                    instance,
                    max_completion_tokens=body.get("max_completion_tokens"),
                )
            elif unhealthy_since is not None:
                record("ready", child.pid, instance, reason="health recovered")
            unhealthy_since, fatal_streak = None, 0
        else:
            if unhealthy_since is None:
                unhealthy_since = now
                if instance:
                    record("health_degraded", child.pid, instance, reason=detail)
            # This explicit 503 is the HPS response to engine.check_health()
            # failing. A slow or timed-out probe is handled with a longer grace.
            fatal = (
                status == 503
                and isinstance(body, dict)
                and body.get("detail") == "Inference engine unavailable"
            )
            fatal_streak = fatal_streak + 1 if fatal else 0
            if fatal_streak >= 2:
                return (
                    instance,
                    ready_since,
                    "engine failed two consecutive readiness checks",
                )
            if instance and now - unhealthy_since >= timing.slow_health:
                return (
                    instance,
                    ready_since,
                    f"continuous unhealthy interval ({detail})",
                )
            if not instance and now - start >= timing.startup:
                return instance, ready_since, f"startup readiness deadline ({detail})"
        stop.wait(timing.poll)
    return instance, ready_since, "stop requested"


def supervise(args, command, stop, timing=Timing()):
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    locks = ROOT / "locks"
    locks.mkdir(exist_ok=True)
    with exclusive(locks / f"hps-supervisor-port-{args.port}.lock"), exclusive(
        directory / ".supervisor.lock"
    ):
        check_free_port(args.port)
        restarts = 0
        previous = None
        backoff = timing.initial_backoff

        def record(state, child_pid=None, instance=None, **extra):
            data = dict(
                time=time.time(),
                state=state,
                supervisor_pid=os.getpid(),
                child_pid=child_pid,
                port=args.port,
                gpu=args.gpu,
                mode=args.mode,
                instance_id=instance,
                restarts=restarts,
                **extra,
            )
            line = json.dumps(data)
            with (directory / "supervisor.jsonl").open("a") as log:
                log.write(line + "\n")
            atomic_text(directory / "status.json", line)
            print(line, flush=True)

        atomic_text(directory / "supervisor.pid", str(os.getpid()))
        try:
            with (directory / "vlm.log").open("ab", buffering=0) as service_log:
                while not stop.is_set():
                    check_free_port(args.port)
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=service_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        atomic_text(directory / "vlm.pid", str(child.pid))
                        record("starting", child.pid, command=command)
                        instance, since, reason = watch(
                            child, args, stop, timing, record, previous
                        )
                    finally:
                        stop_child(child, timing.terminate)
                        (directory / "vlm.pid").unlink(missing_ok=True)
                    if stop.is_set():
                        break
                    previous = instance or previous
                    if since is not None and time.monotonic() - since >= timing.stable:
                        backoff = timing.initial_backoff
                    restarts += 1
                    record(
                        "restarting",
                        instance=previous,
                        reason=reason,
                        delay_seconds=backoff,
                    )
                    stop.wait(backoff)
                    backoff = min(timing.max_backoff, backoff * 2)
        except Exception as exc:
            record("failed", reason=str(exc))
            raise
        else:
            record("stopped")
        finally:
            (directory / "supervisor.pid").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "eagle3-recovery"), required=True
    )
    parser.add_argument("--model-version", choices=("1.6",), default="1.6")
    parser.add_argument("--recovery-decode-graphs", choices=("bf16", "int8"))
    parser.add_argument("--vllm019-packed-attention-library", type=Path)
    parser.add_argument(
        "--hps-max-completion-tokens", type=int, choices=(1024, 4096, 8192)
    )
    parser.add_argument("--queue-size", type=int, choices=range(33), default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--kv-cache-memory-mib", type=int, default=4096)
    args = parser.parse_args()
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("Choose a valid GPU index and port")
    if not 0 < args.gpu_memory_utilization <= 1 or args.kv_cache_memory_mib <= 0:
        parser.error(
            "Choose a GPU memory fraction in (0, 1] and a positive KV cache size"
        )
    if args.recovery_decode_graphs and args.mode != "eagle3-recovery":
        parser.error("--recovery-decode-graphs requires --mode eagle3-recovery")
    if args.vllm019_packed_attention_library is not None and (
        args.mode != "eagle3-recovery"
        or args.recovery_decode_graphs not in (None, "int8")
        or args.hps_max_completion_tokens not in (None, 1024, 8192)
    ):
        parser.error(
            "--vllm019-packed-attention-library requires recovery/int8 and context 4096 or 12288"
        )
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/serve_hps_vlm.py"),
        "--gpu",
        str(args.gpu),
        "--port",
        str(args.port),
        "--mode",
        args.mode,
        "--queue-size",
        str(args.queue_size),
        "--model-version",
        args.model_version,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-mib",
        str(args.kv_cache_memory_mib),
    ]
    if args.recovery_decode_graphs:
        cmd += ["--recovery-decode-graphs", args.recovery_decode_graphs]
    if args.vllm019_packed_attention_library is not None:
        cmd += [
            "--vllm019-packed-attention-library",
            str(args.vllm019_packed_attention_library.expanduser().resolve()),
        ]
    if args.hps_max_completion_tokens is not None:
        cmd += ["--hps-max-completion-tokens", str(args.hps_max_completion_tokens)]
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        supervise(args, cmd, stop)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
