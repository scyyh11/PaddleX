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

"""Safe, pickle-free serialization and evaluation for frozen tree classifiers.

Supports ``ExtraTreesClassifier``, ``HistGradientBoostingClassifier``, and
``LGBMClassifier`` exported into ``.safetensors`` + ``.json`` pairs so runtime
inference does not depend on ``pickle``/``joblib`` or pinned ``scikit-learn``/
``lightgbm`` versions.
"""

import json
import math
import pathlib

import numpy as np

_KZERO = float(np.float32(1e-35))
FORMAT_VERSION = 1


def _parse_lgbm_text(model_string):
    head, _, body = model_string.partition("\nTree=")
    header = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    trees = []
    for block in ("Tree=" + body).split("\nTree="):
        block = block if block.startswith("Tree=") else "Tree=" + block
        text = block.split("end of trees")[0]
        kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        if "num_leaves" in kv:
            trees.append(kv)
    return header, trees


def _sigmoid_list(values, scale=1.0):
    out = []
    exp = math.exp
    for r in values:
        try:
            out.append(1.0 / (1.0 + exp(-scale * r)))
        except OverflowError:
            out.append(0.0)
    return np.asarray(out, dtype=np.float64)


def _softmax_rows(raw):
    out = []
    exp = math.exp
    for row in raw:
        wmax = row[0]
        for v in row[1:]:
            if v > wmax:
                wmax = v
        exps = [exp(v - wmax) for v in row]
        wsum = 0.0
        for v in exps:
            wsum += v
        out.append([v / wsum for v in exps])
    return np.asarray(out, dtype=np.float64).reshape(len(raw), len(raw[0]))


def _extract_model_spec(prefix, model):
    """Extract tensors and metadata for one tree classifier under ``prefix``."""
    tensors = {}
    meta = {"classes": np.asarray(model.classes_).tolist()}
    cls_name = type(model).__name__

    if cls_name == "ExtraTreesClassifier":
        meta["kind"] = "extra"
        outputs = len(model.classes_)
        meta["outputs"] = outputs
        roots, left, right, feature, threshold, missing, values = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for est in model.estimators_:
            tr = est.tree_
            offset = len(left)
            roots.append(offset)
            v = tr.value[:, 0, :].copy()
            den = v.sum(axis=1, keepdims=True)
            v /= np.where(den == 0, 1, den)
            f = tr.feature.copy()
            f[tr.children_left < 0] = -1
            left.extend((np.asarray(tr.children_left) + offset).tolist())
            right.extend((np.asarray(tr.children_right) + offset).tolist())
            feature.extend(np.asarray(f).tolist())
            threshold.extend(np.asarray(tr.threshold).tolist())
            miss = getattr(tr, "missing_go_to_left", np.zeros(tr.node_count))
            missing.extend(np.asarray(miss).tolist())
            values.extend(np.asarray(v).tolist())
        tensors[f"{prefix}.roots"] = np.asarray(roots, dtype=np.int32)
        tensors[f"{prefix}.left"] = np.asarray(left, dtype=np.int32)
        tensors[f"{prefix}.right"] = np.asarray(right, dtype=np.int32)
        tensors[f"{prefix}.feature"] = np.asarray(feature, dtype=np.int32)
        tensors[f"{prefix}.threshold"] = np.asarray(threshold, dtype=np.float64)
        tensors[f"{prefix}.missing_left"] = np.asarray(missing, dtype=np.uint8)
        tensors[f"{prefix}.leaf_values"] = np.asarray(values, dtype=np.float64)
        tensors[f"{prefix}.base"] = np.zeros(outputs, dtype=np.float64)
        return meta, tensors

    if cls_name == "HistGradientBoostingClassifier":
        meta["kind"] = "hgb"
        outputs = int(model.n_trees_per_iteration_)
        meta["outputs"] = outputs
        roots, left, right, feature, threshold, missing, values = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for iteration in model._predictors:
            for c, tree in enumerate(iteration):
                n = tree.nodes
                if np.any(n["is_categorical"]):
                    raise ValueError(
                        "Categorical HistGradientBoostingClassifier trees are not supported"
                    )
                offset = len(left)
                roots.append(offset)
                f = n["feature_idx"].astype(np.int32)
                f[n["is_leaf"] != 0] = -1
                v = np.zeros((len(n), outputs), dtype=np.float64)
                v[:, c] = n["value"]
                left.extend((np.asarray(n["left"]) + offset).tolist())
                right.extend((np.asarray(n["right"]) + offset).tolist())
                feature.extend(f.tolist())
                threshold.extend(np.asarray(n["num_threshold"]).tolist())
                missing.extend(np.asarray(n["missing_go_to_left"]).tolist())
                values.extend(v.tolist())
        tensors[f"{prefix}.roots"] = np.asarray(roots, dtype=np.int32)
        tensors[f"{prefix}.left"] = np.asarray(left, dtype=np.int32)
        tensors[f"{prefix}.right"] = np.asarray(right, dtype=np.int32)
        tensors[f"{prefix}.feature"] = np.asarray(feature, dtype=np.int32)
        tensors[f"{prefix}.threshold"] = np.asarray(threshold, dtype=np.float64)
        tensors[f"{prefix}.missing_left"] = np.asarray(missing, dtype=np.uint8)
        tensors[f"{prefix}.leaf_values"] = np.asarray(values, dtype=np.float64)
        tensors[f"{prefix}.base"] = np.asarray(
            model._baseline_prediction.reshape(-1), dtype=np.float64
        )
        return meta, tensors

    if cls_name == "LGBMClassifier":
        booster = model.booster_
        best = booster.best_iteration
        text = booster.model_to_string(num_iteration=best if best and best > 0 else -1)
        header, trees = _parse_lgbm_text(text)
        if any(
            line.strip() == "average_output"
            for line in text.partition("\nTree=")[0].splitlines()
        ):
            raise ValueError("LightGBM average_output models are not supported")
        objective = header["objective"].split()
        obj_kind = objective[0]
        if obj_kind == "binary":
            params = dict(p.split(":", 1) for p in objective[1:] if ":" in p)
            sigmoid = float(params.get("sigmoid", "1"))
        elif obj_kind == "multiclass":
            sigmoid = 1.0
        else:
            raise ValueError(f"Unsupported LightGBM objective: {header['objective']}")
        meta["kind"] = f"lgbm_{obj_kind}"
        meta["k"] = int(header["num_tree_per_iteration"])
        meta["n_features"] = int(header["max_feature_idx"]) + 1
        meta["sigmoid"] = sigmoid
        base, nleaf, leafoff, feat, thr, dt, left, right, leafval = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for kv in trees:
            nl = int(kv["num_leaves"])
            if int(kv.get("num_cat", "0")) or kv.get("is_linear", "0") != "0":
                raise ValueError(
                    "Categorical or linear LightGBM trees are not supported"
                )
            base.append(len(feat))
            nleaf.append(nl)
            leafoff.append(len(leafval))
            leafval.extend(float(v) for v in kv["leaf_value"].split())
            if nl > 1:
                f = [int(v) for v in kv["split_feature"].split()]
                feat.extend(f)
                thr.extend(float(v) for v in kv["threshold"].split())
                dt.extend(int(v) for v in kv["decision_type"].split())
                left.extend(int(v) for v in kv["left_child"].split())
                right.extend(int(v) for v in kv["right_child"].split())
                if any(d & 1 for d in dt[-len(f) :]):
                    raise ValueError("Categorical LightGBM splits are not supported")
        tensors[f"{prefix}.nleaf"] = np.asarray(nleaf, dtype=np.int32)
        tensors[f"{prefix}.base"] = np.asarray(base, dtype=np.int32)
        tensors[f"{prefix}.leafoff"] = np.asarray(leafoff, dtype=np.int32)
        tensors[f"{prefix}.feature"] = np.asarray(feat, dtype=np.int32)
        tensors[f"{prefix}.threshold"] = np.asarray(thr, dtype=np.float64)
        tensors[f"{prefix}.decision_type"] = np.asarray(dt, dtype=np.uint8)
        tensors[f"{prefix}.left"] = np.asarray(left, dtype=np.int32)
        tensors[f"{prefix}.right"] = np.asarray(right, dtype=np.int32)
        tensors[f"{prefix}.leaf_values"] = np.asarray(leafval, dtype=np.float64)
        return meta, tensors

    raise TypeError(f"Unsupported model type for portable export: {cls_name}")


class _PortableBoosterAdapter:
    """Minimal ``.booster_.predict(x, num_threads=1)`` adapter for LightGBM callers."""

    def __init__(self, classifier):
        self._classifier = classifier

    def predict(self, x, **kwargs):
        del kwargs
        return self._classifier._lgbm_predict(x)


class PortableTreeClassifier:
    """Pickle-free tree ensemble classifier with pure-NumPy CPU evaluation."""

    def __init__(self, meta, tensors):
        self.kind = meta["kind"]
        self.classes_ = np.asarray(meta["classes"])
        if self.kind in ("extra", "hgb"):
            self.outputs = int(meta["outputs"])
            self.roots = tensors["roots"]
            self.left = tensors["left"]
            self.right = tensors["right"]
            self.feature = tensors["feature"]
            self.threshold = tensors["threshold"]
            self.missing_left = tensors["missing_left"]
            self.leaf_values = tensors["leaf_values"]
            self.base = tensors["base"]
            self.trees = len(self.roots)
        else:
            self.k = int(meta["k"])
            self.n_features = int(meta["n_features"])
            self.sigmoid = float(meta["sigmoid"])
            self.nleaf = tensors["nleaf"]
            self.base = tensors["base"]
            self.leafoff = tensors["leafoff"]
            self.feature = tensors["feature"]
            self.threshold = tensors["threshold"]
            self.decision_type = tensors["decision_type"]
            self.left = tensors["left"]
            self.right = tensors["right"]
            self.leaf_values = tensors["leaf_values"]
            self.trees = len(self.base)
            self.booster_ = _PortableBoosterAdapter(self)

    def _predict_sklearn(self, x):
        if self.kind == "extra":
            x_arr = np.asarray(x, dtype=np.float32).astype(np.float64, copy=False)
        else:
            x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.shape[0]
        if n == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)

        acc = np.broadcast_to(self.base, (n, self.outputs)).astype(
            np.float64, copy=True
        )
        for t in range(self.trees):
            nodes = np.full(n, int(self.roots[t]), dtype=np.int32)
            while True:
                f = self.feature[nodes]
                active = f >= 0
                if not np.any(active):
                    break
                act_idx = np.flatnonzero(active)
                act_nodes = nodes[act_idx]
                vals = x_arr[act_idx, f[act_idx]]
                go_left = np.where(
                    np.isnan(vals),
                    self.missing_left[act_nodes] != 0,
                    vals <= self.threshold[act_nodes],
                )
                nodes[act_idx] = np.where(
                    go_left, self.left[act_nodes], self.right[act_nodes]
                )
            acc += self.leaf_values[nodes]

        if self.kind == "extra":
            return acc / self.trees
        if self.outputs == 1:
            p = _sigmoid_list(acc[:, 0].tolist(), scale=1.0)
            return np.column_stack([1.0 - p, p])
        return _softmax_rows(acc.tolist())

    def _lgbm_raw(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.shape[0]
        if n == 0:
            return np.empty((0, self.k), dtype=np.float64)
        x_arr = np.where(np.abs(x_arr) <= _KZERO, 0.0, x_arr)
        raw = np.zeros((n, self.k), dtype=np.float64)

        for t in range(self.trees):
            nl = int(self.nleaf[t])
            loff = int(self.leafoff[t])
            c = t % self.k
            if nl <= 1:
                raw[:, c] += float(self.leaf_values[loff])
                continue
            b_off = int(self.base[t])
            nodes = np.zeros(n, dtype=np.int32)
            active = np.ones(n, dtype=bool)
            while np.any(active):
                act_idx = np.flatnonzero(active)
                g_nodes = b_off + nodes[act_idx]
                vals = x_arr[act_idx, self.feature[g_nodes]]
                dt = self.decision_type[g_nodes]
                mt = (dt >> 2) & 3
                isnan = np.isnan(vals)
                vals_clean = np.where(isnan & (mt != 2), 0.0, vals)
                is_zero = (vals_clean >= -_KZERO) & (vals_clean <= _KZERO)
                use_default = ((mt == 1) & is_zero) | ((mt == 2) & isnan)
                go_left = np.where(
                    use_default,
                    (dt & 2) != 0,
                    vals_clean <= self.threshold[g_nodes],
                )
                nxt = np.where(go_left, self.left[g_nodes], self.right[g_nodes])
                nodes[act_idx] = nxt
                active[act_idx] = nxt >= 0
            raw[:, c] += self.leaf_values[loff + (~nodes)]
        return raw

    def _lgbm_predict(self, x):
        raw = self._lgbm_raw(x)
        if len(raw) == 0:
            if self.kind == "lgbm_binary":
                return np.empty((0,), dtype=np.float64)
            return np.empty((0, self.k), dtype=np.float64)
        if self.kind == "lgbm_binary":
            return _sigmoid_list(raw[:, 0].tolist(), scale=self.sigmoid)
        return _softmax_rows(raw.tolist())

    def predict_proba(self, x):
        if self.kind in ("extra", "hgb"):
            return self._predict_sklearn(x)
        pred = self._lgbm_predict(x)
        if self.kind == "lgbm_binary":
            if len(pred) == 0:
                return np.empty((0, 2), dtype=np.float64)
            return np.column_stack([1.0 - pred, pred])
        return pred


def save_portable_package(package, base_path):
    """Export a model package dict to ``<base_path>.safetensors`` and ``<base_path>.json``."""
    from safetensors.numpy import save_file

    base = pathlib.Path(base_path)
    if base.suffix in (".safetensors", ".json", ".joblib"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    tensors = {}
    models_meta = {}
    extras = {}
    for key, val in package.items():
        if key == "training_documents":
            continue
        cls_name = type(val).__name__
        if cls_name in (
            "ExtraTreesClassifier",
            "HistGradientBoostingClassifier",
            "LGBMClassifier",
        ):
            m_meta, m_tensors = _extract_model_spec(key, val)
            models_meta[key] = m_meta
            tensors.update(m_tensors)
        elif isinstance(val, PortableTreeClassifier):
            raise TypeError("Re-exporting PortableTreeClassifier is not required")
        elif isinstance(val, (str, int, float, bool, list, dict)) or val is None:
            extras[key] = val

    meta_doc = {
        "format_version": FORMAT_VERSION,
        "models": models_meta,
        "extras": extras,
    }
    sf_path = base.with_suffix(".safetensors")
    js_path = base.with_suffix(".json")
    save_file(tensors, str(sf_path))
    js_path.write_text(json.dumps(meta_doc, indent=1), encoding="utf-8")
    return sf_path, js_path


def load_portable_package(safetensors_path):
    """Load a portable ``.safetensors`` + ``.json`` model package without pickle."""
    from safetensors.numpy import load_file

    sf_path = pathlib.Path(safetensors_path)
    js_path = sf_path.with_suffix(".json")
    if not js_path.is_file():
        raise FileNotFoundError(f"Missing portable model metadata: {js_path}")
    meta_doc = json.loads(js_path.read_text(encoding="utf-8"))
    all_tensors = load_file(str(sf_path))

    package = dict(meta_doc.get("extras", {}))
    for key, m_meta in meta_doc.get("models", {}).items():
        prefix = f"{key}."
        m_tensors = {
            tkey[len(prefix) :]: tval
            for tkey, tval in all_tensors.items()
            if tkey.startswith(prefix)
        }
        package[key] = PortableTreeClassifier(m_meta, m_tensors)
    return package
