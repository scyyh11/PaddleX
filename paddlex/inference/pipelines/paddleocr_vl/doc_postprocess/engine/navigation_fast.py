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

"""Linear-time equivalent of the frozen trailing page-number regex."""
import re

MARKERS = re.compile(r"[.．…·]{2,}|\s\s+")


def page_number_suffix(s):
    s = s.rstrip()
    if not s:
        return False
    # Alternative 2: whitespace + integer, optionally a hyphenated range.
    end = len(s)
    i = end
    while i and s[i - 1].isdecimal():
        i -= 1
    if i < end:
        if i and s[i - 1].isspace():
            return True
        if i and s[i - 1] in "-–":
            j = i - 1
            while j and s[j - 1].isspace():
                j -= 1
            k = j
            while k and s[k - 1].isdecimal():
                k -= 1
            if k < j and k and s[k - 1].isspace():
                return True
    # Alternative 1: dot leader / two spaces, then a final digit-dash run.
    i = len(s)
    while i and (s[i - 1].isdecimal() or s[i - 1] in "–—-"):
        i -= 1
    while i < len(s) and not s[i].isdecimal():
        i += 1
    if i == len(s):
        return False
    before = s[:i]
    last_newline = before.rfind("\n")
    return any(m.end() > last_newline for m in MARKERS.finditer(before))
