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

"""CPU-only contracts for external archives and artifact admission; no CUDA."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = dict(
            commit="abc",
            sha256="",
            url="https://example.invalid",
            prefixes=("include/",),
            files=("LICENSE",),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, entries):
        path = self.root / "test.tar.gz"
        with tarfile.open(path, "w:gz") as handle:
            for name, body, kind in entries:
                member = tarfile.TarInfo(name)
                if kind == "link":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../outside"
                    handle.addfile(member)
                else:
                    data = body.encode()
                    member.size = len(data)
                    handle.addfile(member, io.BytesIO(data))
        return path

    def test_verified_cache_never_downloads(self):
        path = self.root / "test-abc.tar.gz"
        path.write_bytes(b"archive bytes")
        self.spec["sha256"] = build.sha256(path)
        with patch.object(
            build.urllib.request, "urlopen", side_effect=AssertionError("No network")
        ):
            self.assertEqual(
                build.checked_archive("test", self.spec, self.root, False), path
            )

    def test_changed_cache_is_rejected_and_retained(self):
        path = self.root / "test-abc.tar.gz"
        path.write_bytes(b"corrupt")
        self.spec["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            build.checked_archive("test", self.spec, self.root, False)
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_offline_missing_archive_fails(self):
        with self.assertRaises(FileNotFoundError):
            build.checked_archive("test", self.spec, self.root, True)

    def test_extracts_only_selected_regular_files(self):
        path = self.archive(
            [
                ("test-abc/include/a.h", "header", "file"),
                ("test-abc/LICENSE", "license", "file"),
                ("test-abc/unrelated.py", "ignored", "file"),
            ]
        )
        root = build.extract_headers(path, "test", self.spec, self.root / "out")
        self.assertEqual((root / "include/a.h").read_text(), "header")
        self.assertFalse((root / "unrelated.py").exists())

    def test_traversal_is_rejected(self):
        path = self.archive([("test-abc/include/../../outside", "bad", "file")])
        with self.assertRaises(ValueError):
            build.extract_headers(path, "test", self.spec, self.root / "out")
        self.assertFalse((self.root / "outside").exists())

    def test_archive_symlink_is_rejected(self):
        path = self.archive([("test-abc/include/link", "", "link")])
        with self.assertRaises(ValueError):
            build.extract_headers(path, "test", self.spec, self.root / "out")

    def test_existing_output_symlink_cannot_escape(self):
        path = self.archive([("test-abc/include/a.h", "bad", "file")])
        outside = self.root / "outside"
        outside.mkdir()
        target = self.root / "out/test-abc"
        target.mkdir(parents=True)
        (target / "include").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            build.extract_headers(path, "test", self.spec, self.root / "out")
        self.assertFalse((outside / "a.h").exists())

    def test_patch_context_must_be_unique(self):
        for text in ("absent", "needle needle"):
            with self.assertRaises(ValueError):
                build.replace_once(text, "needle", "replacement")

    def test_cuda_sources_are_the_frozen_validated_variant(self):
        for name, expected in build.EXPECTED_SOURCE_HASHES.items():
            self.assertEqual(build.sha256(ROOT / "src" / name), expected)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import packed_fa2

        cls.adapter = packed_fa2

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.library = self.root / build.LIBRARY_NAME
        self.library.write_bytes(
            b"fixture: CPU admission test only; never load this as a library"
        )
        self.manifest = {
            "schema_version": 1,
            "profile": self.adapter._PROFILE,
            "gpu_arch": "sm80",
            "nvcc_version": "12.8.93",
            "torch_headers_version": "2.10.0",
            "cuda_source_sha256": self.adapter._SOURCE_HASHES,
            "patched_header_sha256": self.adapter._HEADER_HASH,
            "library_name": self.library.name,
            "library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
            "source_pins": {
                name: dict(commit=commit, sha256=archive_hash)
                for name, (commit, archive_hash) in self.adapter._SOURCE_PINS.items()
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, manifest=None):
        (self.root / "build_manifest.json").write_text(
            json.dumps(self.manifest if manifest is None else manifest)
        )

    def test_exact_profile_and_checksum_pass_cpu_admission(self):
        self.write_manifest()
        self.assertEqual(self.adapter._verify_artifact(self.library), self.manifest)

    def test_profile_and_source_changes_are_rejected(self):
        for key, value in (
            ("schema_version", 2),
            ("profile", "unknown"),
            ("gpu_arch", "sm90"),
            ("nvcc_version", "13.0.0"),
            ("torch_headers_version", "2.13.0"),
            ("cuda_source_sha256", {}),
            ("patched_header_sha256", "bad"),
            ("source_pins", {}),
        ):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = value
            self.write_manifest(manifest)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter._verify_artifact(self.library)

    def test_binary_change_is_rejected(self):
        self.write_manifest()
        self.library.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.adapter._verify_artifact(self.library)


if __name__ == "__main__":
    unittest.main()
