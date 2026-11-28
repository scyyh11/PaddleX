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

"""Exact C-level replacements for per-character counting loops.

cjk(s)          == sum('\u4e00'<=c<='\u9fff' for c in s)
latin(s)        == sum(c.isascii() and c.isalpha() for c in s)   (the ASCII letters are exactly A-Z, a-z)
counter(chars)  -> f with f(s) == sum(c in chars for c in s) for a string of single characters
"""
import re

_CJK = re.compile("[\u4e00-\u9fff]+")
_LATIN = re.compile("[A-Za-z]+")


def cjk(s):
    return len(s) - len(_CJK.sub("", s))


def latin(s):
    return len(s) - len(_LATIN.sub("", s))


def counter(chars):
    pattern = re.compile("[" + "".join(re.escape(c) for c in sorted(set(chars))) + "]+")

    def count(s):
        return len(s) - len(pattern.sub("", s))

    return count
