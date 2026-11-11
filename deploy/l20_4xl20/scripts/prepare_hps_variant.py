"""Make a Triton runtime directory for one GPU from an existing PaddleX HPS runtime.

Usage: prepare_hps_variant.py NAME --base BASE_DIR --count N --vlm-url http://HOST:PORT/v1
                              [--max-batch B --preferred 2 --delay-us US --gpu G --out-root DIR]

BASE_DIR contains runtime/server (as produced by the PaddleX HPS SDK for PaddleOCR-VL-1.6).
Only layout-parsing scheduling (instance count, max batch, dynamic batching) and the VLM
server_url change; the pipeline YAML is otherwise copied unchanged. --gpu only records the
physical GPU; inside the container the GPU is always index 0.
Measured setup on 4x L20: --count 10 --max-batch 2 --preferred 2 --delay-us 20000.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--base', required=True)
parser.add_argument('--out-root', default='.')
parser.add_argument('--count', type=int, required=True)
parser.add_argument('--max-batch', type=int, default=None)
parser.add_argument('--preferred', type=int, nargs='+', default=None)
parser.add_argument('--delay-us', type=int, default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--vlm-url', required=True)
args = parser.parse_args()

base = Path(args.base).resolve()
root = Path(args.out_root).resolve() / args.name
destination = root / 'runtime/server'
assert not root.exists(), 'Use a fresh directory'
shutil.copytree(base / 'runtime/server', destination)

layout = destination / 'model_repo/layout-parsing'
source = (layout / 'config_gpu.pbtxt').read_text()
changed, n = re.subn(r'(?m)^(\s*count:)\s*\d+\s*$', rf'\g<1> {args.count}', source)
assert n == 1, 'expected one instance_group count'
if args.max_batch is not None:
    changed, n = re.subn(r'(?m)^max_batch_size:\s*\d+\s*$', f'max_batch_size: {args.max_batch}', changed)
    assert n == 1, 'expected one max_batch_size'
if args.delay_us is not None or args.preferred:
    lines = []
    if args.preferred:
        lines.append('  preferred_batch_size: [ ' + ', '.join(map(str, args.preferred)) + ' ]')
    if args.delay_us is not None:
        lines.append(f'  max_queue_delay_microseconds: {args.delay_us}')
    changed, n = re.subn(r'(?ms)^dynamic_batching\s*\{.*?\}\s*$', 'dynamic_batching {\n' + '\n'.join(lines) + '\n}\n', changed)
    assert n == 1, 'expected one dynamic_batching block'
(layout / 'config_gpu.pbtxt').write_text(changed)
(layout / 'config.pbtxt').write_text(changed)

# Drop benchmark-only telemetry if the base runtime carried it.
model_py = layout / '1/model.py'
code = model_py.read_text()
code = re.sub(r'\n+# Read-only experiment telemetry[^\n]*\nimport hps_observer\nhps_observer\.install\(TritonPythonModel\)\n?', '\n', code)
model_py.write_text(code)

pipeline_yaml = destination / 'pipeline_config.yaml'
text, n = re.subn(r'(?m)^(\s*server_url:)\s*\S+\s*$', rf'\g<1> {args.vlm_url}', pipeline_yaml.read_text())
assert n == 1, 'expected one VLRecognition server_url'
pipeline_yaml.write_text(text)

provenance = dict(physical_gpu=args.gpu, layout_instances=args.count, base_directory=str(base),
                  vlm_url=args.vlm_url, max_batch_size=args.max_batch,
                  preferred_batch_size=args.preferred, max_queue_delay_microseconds=args.delay_us,
                  model_config_before=source, model_config_after=changed,
                  pipeline_sha256=hashlib.sha256(pipeline_yaml.read_bytes()).hexdigest())
(root / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
print(json.dumps(provenance, ensure_ascii=False, indent=2))
