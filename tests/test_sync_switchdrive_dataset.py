from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "sync_switchdrive_dataset.py"
SPEC = importlib.util.spec_from_file_location("sync_switchdrive_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class SwitchdriveSyncTests(unittest.TestCase):
    def test_multistatus_parses_sha1_and_directory(self) -> None:
        digest = "0123456789abcdef0123456789abcdef01234567"
        document = f"""<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
          <d:response><d:href>/public.php/webdav/</d:href>
            <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
            <d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>
          <d:response><d:href>/public.php/webdav/images/</d:href>
            <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
            <d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>
          <d:response><d:href>/public.php/webdav/metadata.json</d:href>
            <d:propstat><d:prop><d:resourcetype/><d:getcontentlength>12</d:getcontentlength>
            <d:getetag>&quot;etag&quot;</d:getetag><oc:checksums>
            <oc:checksum>MD5:bad SHA1:{digest}</oc:checksum></oc:checksums>
            </d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>
        </d:multistatus>""".encode()
        entries = sync.parse_multistatus(
            document,
            "https://drive.switch.ch/public.php/webdav/",
            "https://drive.switch.ch/public.php/webdav/",
        )
        self.assertEqual([item.relative_path for item in entries], ["images", "metadata.json"])
        self.assertTrue(entries[0].is_directory)
        self.assertEqual(entries[1].size, 12)
        self.assertEqual(entries[1].sha1, digest)

    def test_staging_guard_and_explicit_hardlink_seed_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            canonical.mkdir()
            (canonical / "images").mkdir()
            with self.assertRaises(ValueError):
                sync.prepare_staging(canonical, sync.DEFAULT_BASE_URL, "token", False)

            staging = root / "dataset_staging"
            staging.mkdir()
            (staging / "seed.txt").write_text("seed", encoding="utf-8")
            sync.prepare_staging(
                staging, sync.DEFAULT_BASE_URL, "token", False, initialize_staging=True
            )
            self.assertTrue((staging / sync.STAGING_MARKER).is_file())
            sync.prepare_staging(staging, sync.DEFAULT_BASE_URL, "token", False)
            with self.assertRaises(ValueError):
                sync.prepare_staging(staging, sync.DEFAULT_BASE_URL, "other", False)

    def test_resume_and_atomic_replace_preserve_seed_hardlink(self) -> None:
        remote = b"abcdefghij"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.bin"
            canonical.write_bytes(b"old-data!!")
            staging = root / "dataset_staging"
            staging.mkdir()
            target = staging / "images" / "case.bin"
            target.parent.mkdir()
            target.hardlink_to(canonical)
            part = target.with_name(target.name + ".part")
            part.write_bytes(remote[:3])
            entry = sync.RemoteEntry(
                relative_path="images/case.bin",
                url="https://drive.switch.ch/public.php/webdav/images/case.bin",
                is_directory=False,
                size=len(remote),
                etag='"new"',
                modified=None,
                sha1=hashlib.sha1(remote).hexdigest(),
            )
            requests = []

            def fake_urlopen(request, timeout):
                requests.append((request, timeout))
                return FakeResponse(
                    remote[3:], 206, {"Content-Range": "bytes 3-9/10"}
                )

            with mock.patch.object(sync.urllib.request, "urlopen", fake_urlopen):
                result = sync.download_entry(entry, staging, "Basic secret", 1, 0, 0)
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(target.read_bytes(), remote)
            self.assertEqual(canonical.read_bytes(), b"old-data!!")
            self.assertNotEqual(target.stat().st_ino, canonical.stat().st_ino)
            self.assertEqual(requests[0][0].get_header("Range"), "bytes=3-")

    def test_dry_run_discovers_every_top_level_directory_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "new_staging"
            directories = [
                sync.RemoteEntry(
                    relative_path=f"dir_{index}",
                    url=f"{sync.DEFAULT_BASE_URL}dir_{index}/",
                    is_directory=True,
                    size=None,
                    etag=None,
                    modified=None,
                    sha1=None,
                )
                for index in range(5)
            ]
            metadata = sync.RemoteEntry(
                relative_path="metadata.json",
                url=f"{sync.DEFAULT_BASE_URL}metadata.json",
                is_directory=False,
                size=2,
                etag='"meta"',
                modified=None,
                sha1=hashlib.sha1(b"{}").hexdigest(),
            )
            arguments = sync.argparse.Namespace(
                destination=destination,
                share_token="token",
                base_url=sync.DEFAULT_BASE_URL,
                workers=2,
                retries=1,
                backoff=0.0,
                timeout=1.0,
                expected_top_level_dirs=5,
                dry_run=True,
                initialize_staging=False,
                delete_extraneous=False,
            )
            with (
                mock.patch.object(sync, "parse_args", return_value=arguments),
                mock.patch.object(
                    sync,
                    "discover_tree",
                    return_value=(directories + [metadata], [f"dir_{i}" for i in range(5)]),
                ),
                mock.patch("builtins.print"),
            ):
                sync.main()
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
