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

"""Make predicted display formulas compile; formula content is kept.

PaddleOCR-VL sometimes writes display formulas that standard LaTeX rejects:
document-level environments (align*, gather*, eqnarray) or \\tag inside
display math, unbalanced braces, table rules (\\hline) inside aligned rows,
row breaks inside a \\frac argument, a backslash left in front of a digit by a
removed currency sign, alignment tabs inside a \\left..\\right group, starred
operator names such as \\operatorname*{m a x}, and font or delimiter commands
used without an argument the renderer understands. A formula that does not
compile cannot be read at all, so each construct is replaced by its valid
in-math counterpart. Formulas inside HTML tables are left byte-identical.
"""
import re

DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)
ALIGN = re.compile(
    r"\\(begin|end)\{(?:align|eqnarray|flalign|align\*|eqnarray\*|flalign\*)\}"
)
ALIGNAT = re.compile(r"\\begin\{alignat\*?\}\{\d+\}|\\end\{alignat\*?\}")
GATHER = re.compile(
    r"\\(begin|end)\{(?:gather|gather\*|gathered|multline|multline\*)\}"
)
EQUATION = re.compile(r"\\(?:begin|end)\{(?:equation|equation\*|displaymath|split)\}")
TAG = re.compile(r"\\tag\*?\{([^{}]*)\}")
DROP = re.compile(r"\\(?:label\{[^{}]*\}|nonumber\b|notag\b)")
FONT = re.compile(r"\\(?:mathsf|mathit|mathtt|mathnormal|mathscr|textsl)(?![a-zA-Z])")
FONT_MAP = {"\\mathscr": "\\mathcal"}
RULE = re.compile(
    r"\\(?:hline|toprule|midrule|bottomrule)(?![a-zA-Z])|\\cline\{[^{}]*\}"
)
SIZED = re.compile(r"\\(?:big|Big|bigg|Bigg)[lrm]?(?![a-zA-Z])\s*(?=[/|.])")
STRAY = re.compile(r"(?<!\\)\\(?=\d)")
MOD = re.compile(r"\\mod(?![a-zA-Z])")
BEGIN = re.compile(r"\\begin\{[^{}]*\}(?:\{[^{}]*\})?|\\end\{[^{}]*\}")
OPNAME = re.compile(r"\\operatorname\*?\{([^{}]*)\}")
NAMED = {
    "max",
    "min",
    "sup",
    "inf",
    "lim",
    "det",
    "exp",
    "log",
    "ln",
    "limsup",
    "liminf",
}
# A currency sign dropped from "\$\mathrm{..}" leaves "\\mathrm{..}": a row break
# followed by the letters of a command name, which is never meant literally.
ESCAPED = re.compile(
    r"(?<!\\)\\\\(?=(?:mathrm|mathbf|mathit|mathsf|text|textrm|textbf|quad|qquad|times|frac|cdot|div|operatorname|left|right|sum|int|sqrt|pm)(?![a-zA-Z]))"
)
LRTOK = re.compile(
    r"\\left(?![a-zA-Z])\s*(\\[{}|]|\\[a-zA-Z]+|[()\[\]|./<>])|\\right(?![a-zA-Z])\s*(\\[{}|]|\\[a-zA-Z]+|[()\[\]|./<>])|\\begin\{[^{}]*\}|\\end\{[^{}]*\}|\\\\|&|\\.",
    re.S,
)


def _braces(s):
    depth = 0
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(s[i : i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                i += 1
                continue
            depth -= 1
        out.append(c)
        i += 1
    return "".join(out) + "}" * depth


def _group_breaks(s):
    # A row break belongs to the innermost environment; one nested inside an
    # argument group (e.g. in \frac{..}{..}) has no row to end.
    out = []
    depth = 0
    env = [0]
    i = 0
    while i < len(s):
        m = BEGIN.match(s, i)
        if m:
            if m.group(0).startswith("\\begin"):
                env.append(depth)
            elif len(env) > 1:
                env.pop()
            out.append(m.group(0))
            i = m.end()
            continue
        if s.startswith("\\\\", i):
            out.append("\\\\" if depth == env[-1] else " ")
            i += 2
            continue
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(s[i : i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        out.append(c)
        i += 1
    return "".join(out)


def _operator(m):
    name = re.sub(r"[\s~\\,]+", "", m.group(1))
    return "\\" + name + " " if name in NAMED else "\\operatorname{" + name + "}"


def _aligned_delimiters(s):
    # An alignment tab or row break may not sit inside a \left..\right group of
    # the same environment; such a pair keeps its delimiters without sizing.
    stack = []
    env = 0
    spans = []
    for m in LRTOK.finditer(s):
        t = m.group(0)
        if t.startswith("\\left") and m.group(1) is not None:
            stack.append([m.start(), m.end(), m.group(1), env, False])
        elif t.startswith("\\right") and m.group(2) is not None:
            if stack:
                o = stack.pop()
                if o[4]:
                    spans.append((o, (m.start(), m.end(), m.group(2))))
        elif t.startswith("\\begin"):
            env += 1
        elif t.startswith("\\end"):
            env = max(0, env - 1)
        elif t in ("&", "\\\\") and stack and stack[-1][3] == env:
            stack[-1][4] = True
    rep = []
    for o, c in spans:
        rep += [
            (o[0], o[1], "" if o[2] == "." else o[2]),
            (c[0], c[1], "" if c[2] == "." else c[2]),
        ]
    for a, b, t in sorted(rep, reverse=True):
        s = s[:a] + t + s[b:]
    return s


def repair(body):
    s = ESCAPED.sub(r"\\", body)
    s = ALIGN.sub(lambda m: "\\" + m.group(1) + "{aligned}", s)
    s = ALIGNAT.sub(
        lambda m: (
            "\\begin{aligned}" if m.group(0).startswith("\\begin") else "\\end{aligned}"
        ),
        s,
    )
    s = GATHER.sub(
        lambda m: "\\begin{array}{c}" if m.group(1) == "begin" else "\\end{array}", s
    )
    s = EQUATION.sub("", s)
    s = TAG.sub(lambda m: "\\quad(" + m.group(1).strip() + ")", s)
    s = DROP.sub("", s)
    s = _braces(s)
    s = FONT.sub(lambda m: FONT_MAP.get(m.group(0), "\\mathrm"), s)
    s = RULE.sub(" ", s)
    s = SIZED.sub("", s)
    s = STRAY.sub("", s)
    s = _group_breaks(s)
    s = MOD.sub(r"\\bmod", s)
    s = OPNAME.sub(_operator, s)
    s = _aligned_delimiters(s)
    return s


def apply(text):
    n = [0]

    def sub(m):
        new = repair(m.group(1))
        if new != m.group(1):
            n[0] += 1
        return "\\[" + new + "\\]"

    chunks = re.split(r"(<table\b.*?</table>)", text, flags=re.S)
    return (
        "".join(c if c.startswith("<table") else DISPLAY.sub(sub, c) for c in chunks),
        n[0],
    )


if __name__ == "__main__":
    t = [
        (
            r"\[\begin{align*}a&=b\\&=c\tag{6.3}\end{align*}\]",
            r"\[\begin{aligned}a&=b\\&=c\quad(6.3)\end{aligned}\]",
        ),
        (
            r"\[\mathbf{0.025\cdot\bar{q}_{e}+1.275\geq1.00\]",
            r"\[\mathbf{0.025\cdot\bar{q}_{e}+1.275\geq1.00}\]",
        ),
        (
            r"\[\begin{gathered}a\\b\end{gathered}\]",
            r"\[\begin{array}{c}a\\b\end{array}\]",
        ),
        (r"\[a=\{b\}\]", r"\[a=\{b\}\]"),
        (r"\[a}=b\]", r"\[a=b\]"),
        (
            r"\[\text{HK}\70\times\frac{a\\b}{c}\\=d\]",
            r"\[\text{HK}70\times\frac{a b}{c}\\=d\]",
        ),
        (r"\[\mathsf{A}\big/B\hline k\mod p\]", r"\[\mathrm{A}/B  k\bmod p\]"),
        (
            r"<table><tr><td>\[\begin{align*}a\end{align*}\]</td></tr></table>",
            r"<table><tr><td>\[\begin{align*}a\end{align*}\]</td></tr></table>",
        ),
        (
            r"\[{\begin{aligned}a\\b\end{aligned}}\]",
            r"\[{\begin{aligned}a\\b\end{aligned}}\]",
        ),
        (
            r"\[M=\operatorname*{m a x}_{k}x,\operatorname*{arg\,min} y\]",
            r"\[M=\max _{k}x,\operatorname{argmin} y\]",
        ),
        (
            r"\[y=\left\{\begin{aligned}&0\\&1/\left[1+&a\right]\end{aligned}\right.\]",
            r"\[y=\left\{\begin{aligned}&0\\&1/[1+&a]\end{aligned}\right.\]",
        ),
        (
            r"\[\mathrm{HK}\\mathrm{207}\times2\\quad3\]",
            r"\[\mathrm{HK}\mathrm{207}\times2\quad3\]",
        ),
    ]
    for a, b in t:
        o, _ = apply(a)
        assert o == b, (a, o, b)
    print("ok")
