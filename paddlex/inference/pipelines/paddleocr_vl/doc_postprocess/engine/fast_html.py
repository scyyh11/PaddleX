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

"""Small read-only HTML view; irregular HTML falls back to the frozen parser.

Only the text/find_all interface used by text-region rules is provided. Table
editing continues to use BeautifulSoup. Instances are request-local and immutable.
"""
from html.parser import HTMLParser

_formatter = None


def _get_formatter():
    global _formatter
    if _formatter is None:
        from bs4.formatter import HTMLFormatter

        _formatter = HTMLFormatter.REGISTRY["minimal"]
    return _formatter


VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class Node:
    __slots__ = ("name", "attrs", "children", "_text", "_rows", "_cells", "_html")

    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []
        self._text = None
        self._rows = None
        self._cells = None
        self._html = None

    def get(self, key, default=None):
        return self.attrs.get(key, default)

    def __str__(self):
        # nodes are never modified once built (append_rows makes new nodes), so the serialization of a
        # node is kept; merged tables then reuse the markup of the rows they share with earlier merges
        html = self._html
        if html is not None:
            return html
        fmt = _get_formatter()
        content = "".join(
            fmt.substitute(c) if isinstance(c, str) else str(c) for c in self.children
        )
        if self.name == "[document]":
            html = content
        else:
            attrs = "".join(
                " "
                + key
                + "="
                + fmt.quoted_attribute_value(fmt.attribute_value(str(value)))
                for key, value in sorted(self.attrs.items())
            )
            html = (
                "<"
                + self.name
                + attrs
                + (
                    "/>"
                    if self.name in VOID
                    else ">" + content + "</" + self.name + ">"
                )
            )
        self._html = html
        return html

    def __deepcopy__(self, memo):
        from bs4 import BeautifulSoup

        result = BeautifulSoup(str(self), "html.parser").find(self.name)
        memo[id(self)] = result
        return result

    def get_text(self, separator="", strip=False):
        if separator == " " and strip and self._text is not None:
            return self._text

        def strings(node):
            for child in node.children:
                if isinstance(child, str):
                    t = child.strip() if strip else child
                    if t:
                        yield t
                else:
                    yield from strings(child)

        result = separator.join(strings(self))
        if separator == " " and strip:
            self._text = result
        return result

    def find_all(self, names, recursive=True):
        if names == "tr" and recursive and self._rows is not None:
            return self._rows
        if names == ["td", "th"] and not recursive and self._cells is not None:
            return self._cells
        allowed = {names} if isinstance(names, str) else set(names)
        result = []
        for child in self.children:
            if isinstance(child, str):
                continue
            if child.name in allowed:
                result.append(child)
            if recursive:
                result.extend(child.find_all(names, True))
        if names == "tr" and recursive:
            self._rows = result
        if names == ["td", "th"] and not recursive:
            self._cells = result
        return result


class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("[document]")
        self.stack = [self.root]
        self.valid = True

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "template", "textarea", "pre"}:
            self.valid = False
        node = Node(tag)
        node.attrs = {k: v or "" for k, v in attrs}
        self.stack[-1].children.append(node)
        if any(
            k
            in {
                "class",
                "rel",
                "rev",
                "headers",
                "accept-charset",
                "accesskey",
                "dropzone",
            }
            for k, v in attrs
        ):
            self.valid = False
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if len(self.stack) <= 1 or self.stack[-1].name != tag:
            self.valid = False
            return
        self.stack.pop()

    def handle_data(self, text):
        # BS4 coalesces adjacent data callbacks before a markup boundary.
        children = self.stack[-1].children
        if not text.strip():
            text = "\n" if "\n" in text else " "
        if children and isinstance(children[-1], str):
            children[-1] += text
        else:
            children.append(text)

    def handle_comment(self, data):
        self.valid = False

    def handle_decl(self, decl):
        self.valid = False

    def unknown_decl(self, data):
        self.valid = False

    def handle_pi(self, data):
        self.valid = False


def parse_html(text):
    from bs4 import BeautifulSoup

    p = Parser()
    try:
        p.feed(text)
        p.close()
    except Exception:
        return BeautifulSoup(text, "html.parser")
    if not p.valid or len(p.stack) != 1:
        return BeautifulSoup(text, "html.parser")
    return p.root


def append_rows(previous, current, skip=0):
    """Persistent DOM edit: untouched nodes remain immutable and can be shared."""
    if not isinstance(previous, Node) or not isinstance(current, Node):
        from bs4 import BeautifulSoup

        a = BeautifulSoup(str(previous), "html.parser")
        b = BeautifulSoup(str(current), "html.parser")
        addition = b.find_all("tr")[skip:]
        if not addition:
            return a
        rows = a.find_all("tr")
        parent = rows[-1].parent if rows else (a.find("tbody") or a.find("table") or a)
        for row in addition:
            parent.append(row.extract())
        return a
    addition = current.find_all("tr")[skip:]
    if not addition:
        return previous
    prev_rows = previous.find_all("tr")
    if not prev_rows:
        containers = previous.find_all("tbody") or previous.find_all("table")
        if not containers:
            return previous
        target = containers[-1]

        def append_into(node):
            if node is target:
                result = Node(node.name)
                result.attrs = node.attrs
                result.children = node.children + addition
                return result, True
            for i, child in enumerate(node.children):
                if isinstance(child, Node):
                    replacement, changed = append_into(child)
                    if changed:
                        result = Node(node.name)
                        result.attrs = node.attrs
                        result.children = (
                            node.children[:i] + [replacement] + node.children[i + 1 :]
                        )
                        return result, True
            return node, False

        return append_into(previous)[0]
    target = prev_rows[-1]

    def replace(node):
        if target in node.children:
            result = Node(node.name)
            result.attrs = node.attrs
            result.children = node.children + addition
            return result, True
        for i, child in enumerate(node.children):
            if isinstance(child, Node):
                replacement, changed = replace(child)
                if changed:
                    result = Node(node.name)
                    result.attrs = node.attrs
                    result.children = (
                        node.children[:i] + [replacement] + node.children[i + 1 :]
                    )
                    return result, True
        return node, False

    return replace(previous)[0]


def parse(text):
    # Strict XML is a safe fast path for well-formed OCR table markup. HTML-only
    # constructs, entities and irregular nesting retain the original HTML path.
    if any(token in text for token in ("<!", "<?", "\r")):
        return parse_html(text)
    from lxml import etree

    try:
        root = etree.fromstring(
            ("<root>" + text + "</root>").encode(),
            parser=etree.XMLParser(
                resolve_entities=False, no_network=True, load_dtd=False
            ),
        )
    except (etree.XMLSyntaxError, ValueError):
        return parse_html(text)
    allowed = {
        "root",
        "table",
        "tbody",
        "thead",
        "tfoot",
        "tr",
        "td",
        "th",
        "b",
        "i",
        "em",
        "strong",
        "span",
        "sup",
        "sub",
        "p",
        "div",
        "br",
        "u",
        "s",
        "a",
    }
    for elem in root.iter():
        if elem.tag not in allowed or any(
            k
            in {
                "class",
                "rel",
                "rev",
                "headers",
                "accept-charset",
                "accesskey",
                "dropzone",
            }
            or ":" in k
            or any(c in v for c in "\n\r\t")
            for k, v in elem.attrib.items()
        ):
            return parse_html(text)

    def data(value):
        return (
            ("\n" if "\n" in value else " ") if value and not value.strip() else value
        )

    def convert(elem):
        node = Node("[document]" if elem is root else elem.tag)
        node.attrs = dict(elem.attrib)
        if elem.text:
            node.children.append(data(elem.text))
        for child in elem:
            node.children.append(convert(child))
            if child.tail:
                node.children.append(data(child.tail))
        return node

    return convert(root)
