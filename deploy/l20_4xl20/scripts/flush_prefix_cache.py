"""Evict a VLM server's prefix cache by sending unique text-only prompts.

Usage: flush_prefix_cache.py BASE_URL TOTAL_TOKENS [SEED_OFFSET]
Use a new SEED_OFFSET for every flush; repeating prompts would only hit the cache.
Each prompt is distinct random words, so it is cached once and pushes older
blocks (earlier image crops) out of the LRU prefix cache. max_tokens=1.
"""
import concurrent.futures
import json
import random
import sys
import urllib.request

base, total = sys.argv[1].rstrip("/"), int(sys.argv[2])
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
per_request = 1000  # words; several tokens each, about 4-5k tokens per prompt
words = [f"w{i}" for i in range(50000)]


def send(seed):
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(per_request))
    body = json.dumps({"model": "PaddleOCR-VL-1.6-0.9B", "max_tokens": 1, "temperature": 0,
                       "prompt": text}).encode()
    req = urllib.request.Request(base + "/v1/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["usage"]["prompt_tokens"]


tokens, count = 0, 0
with concurrent.futures.ThreadPoolExecutor(32) as pool:
    while tokens < total:
        batch = list(pool.map(send, range(offset + count, offset + count + 32)))
        tokens += sum(batch)
        count += len(batch)
print(json.dumps({"url": base, "requests": count, "prompt_tokens": tokens}))
