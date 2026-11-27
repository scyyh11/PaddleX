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

"""Reference copy of the v44 module (verification only). Nonlocal ordinal factors used inside a parent/subtree beam search."""
import math

import numpy as np

from .heading_graph import graph_edges, relation_features
from .heading_joint_relations import MODEL  # shared with the production module


def factors(meta, x):
    model = MODEL.get()
    n = len(meta)
    refs = [[] for _ in meta]
    if model is None or n < 2:
        return refs
    edges = graph_edges(meta, x)
    for start in range(0, len(edges), 4096):
        batch = edges[start : start + 4096]
        xx = np.asarray(
            [relation_features(meta[i], meta[j], x[i], x[j]) for i, j in batch]
        )
        probs = model.predict_proba(xx)
        for (i, j), pr in zip(batch, probs):
            lp = {
                int(c): math.log(max(float(v), 1e-5))
                for c, v in zip(model.classes_, pr)
            }
            refs[int(j)].append((int(i), lp))
    return refs


def score(refs, path, level):
    if not refs:
        return 0.0
    # Each heading contributes one normalized factor. Long documents therefore
    # do not overwhelm parent/closure evidence merely by having more edges.
    return sum(
        lp.get(int(np.sign(level - path[j])), math.log(1e-5)) for j, lp in refs
    ) / len(refs)
