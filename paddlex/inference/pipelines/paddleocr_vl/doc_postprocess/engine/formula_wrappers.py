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

"""Unwrap single-line equation environments inside display math.

PaddleOCR-VL sometimes writes a one-line display formula as
\\begin{align*}x=1\\end{align*} inside $$...$$. Standard LaTeX rejects an
align/equation environment nested in display math, so such a formula does not
render. A single-line body needs no environment, so the wrapper and its
alignment markers are removed. Multi-line environments are left unchanged."""
import re

ENV = re.compile(r"\\begin\{(align\*?|equation\*?|gather\*?)\}(.*?)\\end\{\1\}", re.S)
AMP = re.compile(r"(?<!\\)&")


def _fix(m):
    body = m.group(2)
    if "\\\\" in body:
        return m.group(0)
    return AMP.sub("", body).strip()


def apply(text):
    count = [0]

    def sub(m):
        out = _fix(m)
        if out != m.group(0):
            count[0] += 1
        return out

    new = ENV.sub(sub, text)
    return new, count[0]


if __name__ == "__main__":
    tests = [
        (
            r"$$ \begin{align*}P(z)=\sum_{k=0}^{2019}b_kz^k.\end{align*} $$",
            r"$$ P(z)=\sum_{k=0}^{2019}b_kz^k. $$",
        ),
        (
            r"$$ \begin{align*}a&=b\\&=c\end{align*} $$",
            r"$$ \begin{align*}a&=b\\&=c\end{align*} $$",
        ),
        (r"\begin{equation}x\&y&=1\end{equation}", r"x\&y=1"),
        (r"\begin{aligned}a&=b\end{aligned}", r"\begin{aligned}a&=b\end{aligned}"),
    ]
    for a, b in tests:
        out, _ = apply(a)
        assert out == b, (a, out, b)
    print("ok")
