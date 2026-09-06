"""Safely mirror a public SWITCHdrive WebDAV share into a staging directory."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import posixpath
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_BASE_URL = "https://drive.switch.ch/public.php/webdav/"
DEFAULT_SHARE_TOKEN = "O36U43RkChkNcHd"
STAGING_MARKER = ".switchdrive_staging.json"
DEFAULT_MANIFEST = "switchdrive_sync_manifest.json"
MAX_WORKERS = 16
CHUNK_SIZE = 8 * 1024 * 1024
PROPFIND_BODY = b"""<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:prop>
<d:resourcetype/><d:getcontentlength/><d:getetag/><d:getlastmodified/>
<oc:checksums/>
</d:prop></d:propfind>"""


@dataclass(frozen=True)
class RemoteEntry:
    relative_path: str
    url: str
    is_directory: bool
    size: int | None
    etag: str | None
    modified: str | None
    sha1: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha1(value: str | None) -> str | None:
    for checksum in (value or "").replace(",", " ").split():
        algorithm, separator, digest = checksum.partition(":")
        if separator and algorithm.upper() == "SHA1" and len(digest) == 40:
            try:
                int(digest, 16)
            except ValueError:
                continue
            return digest.lower()
    return None


def normalized_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("--base-url must be an absolute HTTPS URL")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/", "", "")
    )


def authorization_header(token: str) -> str:
    encoded = base64.b64encode(f"{token}:".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def retryable(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


def backoff_sleep(attempt: int, base_seconds: float) -> None:
    time.sleep(min(base_seconds * (2**attempt), 30.0))


def request_bytes(
    request: urllib.request.Request,
    timeout: float,
    retries: int,
    backoff: float,
) -> tuple[bytes, Any]:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers
        except BaseException as error:
            if attempt >= retries or not retryable(error):
                raise
            backoff_sleep(attempt, backoff)
    raise AssertionError("unreachable")


def safe_relative_path(href: str, requested_url: str, base_url: str) -> tuple[str, str]:
    absolute = urllib.parse.urljoin(requested_url, href)
    child = urllib.parse.urlsplit(absolute)
    base = urllib.parse.urlsplit(base_url)
    if (child.scheme, child.netloc) != (base.scheme, base.netloc):
        raise ValueError(f"WebDAV response escaped the share host: {href}")
    child_path = urllib.parse.unquote(child.path)
    base_path = urllib.parse.unquote(base.path).rstrip("/") + "/"
    if child_path.rstrip("/") == base_path.rstrip("/"):
        return "", absolute
    if not child_path.startswith(base_path):
        raise ValueError(f"WebDAV response escaped the share root: {href}")
    relative = child_path[len(base_path) :].strip("/")
    normalized = posixpath.normpath(relative)
    parts = PurePosixPath(normalized).parts
    if (
        not relative
        or normalized in {"", ".", ".."}
        or normalized.startswith("../")
        or any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts)
    ):
        raise ValueError(f"Unsafe WebDAV path: {href}")
    return PurePosixPath(*parts).as_posix(), absolute


def _find_text(element: ET.Element, local_name: str) -> str | None:
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] == local_name:
            value = (child.text or "").strip()
            if value:
                return value
    return None


def parse_multistatus(
    document: bytes, requested_url: str, base_url: str
) -> list[RemoteEntry]:
    root = ET.fromstring(document)
    entries: list[RemoteEntry] = []
    for response in root.iter():
        if response.tag.rsplit("}", 1)[-1] != "response":
            continue
        href = _find_text(response, "href")
        if not href:
            continue
        relative, absolute = safe_relative_path(href, requested_url, base_url)
        if not relative:
            continue
        successful_prop = None
        for propstat in response:
            if propstat.tag.rsplit("}", 1)[-1] != "propstat":
                continue
            status = _find_text(propstat, "status") or ""
            if " 200 " in status:
                successful_prop = propstat
                break
        if successful_prop is None:
            continue
        is_directory = any(
            child.tag.rsplit("}", 1)[-1] == "collection"
            for child in successful_prop.iter()
        )
        length = _find_text(successful_prop, "getcontentlength")
        size = None if is_directory or length is None else int(length)
        if not is_directory and size is None:
            raise ValueError(f"Remote file has no expected size: {relative}")
        entries.append(
            RemoteEntry(
                relative_path=relative,
                url=absolute,
                is_directory=is_directory,
                size=size,
                etag=_find_text(successful_prop, "getetag"),
                modified=_find_text(successful_prop, "getlastmodified"),
                sha1=parse_sha1(
                    _find_text(successful_prop, "checksum")
                    or _find_text(successful_prop, "checksums")
                ),
            )
        )
    return entries


def propfind(
    url: str,
    base_url: str,
    auth: str,
    timeout: float,
    retries: int,
    backoff: float,
) -> list[RemoteEntry]:
    request = urllib.request.Request(
        url,
        data=PROPFIND_BODY,
        method="PROPFIND",
        headers={
            "Authorization": auth,
            "Depth": "1",
            "Content-Type": "application/xml; charset=utf-8",
            "User-Agent": "TopAneu-SWITCHdrive-sync/1",
        },
    )
    document, _ = request_bytes(request, timeout, retries, backoff)
    return parse_multistatus(document, url, base_url)


def discover_tree(
    base_url: str,
    auth: str,
    timeout: float,
    retries: int,
    backoff: float,
) -> tuple[list[RemoteEntry], list[str]]:
    queue = [base_url]
    seen_directories: set[str] = set()
    entries: dict[str, RemoteEntry] = {}
    top_level_directories: list[str] = []
    while queue:
        current = queue.pop(0)
        current_relative = (
            safe_relative_path(current, base_url, base_url)[0]
            if current != base_url
            else ""
        )
        if current_relative in seen_directories:
            continue
        seen_directories.add(current_relative)
        children = propfind(current, base_url, auth, timeout, retries, backoff)
        for entry in children:
            previous = entries.get(entry.relative_path)
            if previous is not None and previous != entry:
                raise ValueError(f"Conflicting WebDAV entry: {entry.relative_path}")
            entries[entry.relative_path] = entry
            if entry.is_directory:
                if "/" not in entry.relative_path:
                    top_level_directories.append(entry.relative_path)
                queue.append(entry.url.rstrip("/") + "/")
    ordered = sorted(entries.values(), key=lambda item: item.relative_path)
    return ordered, sorted(set(top_level_directories))


def staging_identity(base_url: str, token: str) -> dict[str, str]:
    return {
        "format": "topaneu-switchdrive-staging-v1",
        "base_url": base_url,
        "share_token_sha256": token_fingerprint(token),
    }


def prepare_staging(
    destination: Path,
    base_url: str,
    token: str,
    dry_run: bool,
    initialize_staging: bool = False,
) -> Path:
    raw_destination = destination.expanduser()
    if raw_destination.exists() and raw_destination.is_symlink():
        raise ValueError(f"Destination itself must not be a symlink: {raw_destination}")
    destination = raw_destination.resolve()
    if destination == Path(destination.anchor) or destination == Path.home().resolve():
        raise ValueError(f"Refusing broad destination: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Destination is not a directory: {destination}")
    expected = staging_identity(base_url, token)
    children = list(destination.iterdir()) if destination.exists() else []
    marker = destination / STAGING_MARKER
    if children:
        if not marker.is_file():
            if not initialize_staging or "staging" not in destination.name.lower():
                raise ValueError(
                    "Destination is non-empty and has no SWITCHdrive staging marker; "
                    "refusing to overwrite a canonical dataset. A hardlink-seeded "
                    "directory must have 'staging' in its name and be explicitly "
                    "registered with --initialize-staging."
                )
            if not dry_run:
                atomic_json_dump({**expected, "created_at": utc_now()}, marker)
            return destination
        actual = json.loads(marker.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"Staging marker mismatch for {key}: {destination}")
    elif not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        marker_payload = {**expected, "created_at": utc_now()}
        atomic_json_dump(marker_payload, marker)
    return destination


def local_path(destination: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    target = destination.joinpath(*parts)
    resolved_parent = target.parent.resolve()
    if destination != resolved_parent and destination not in resolved_parent.parents:
        raise ValueError(f"Local path escaped staging: {relative_path}")
    return target


def set_remote_mtime(path: Path, modified: str | None) -> None:
    if not modified:
        return
    try:
        timestamp = parsedate_to_datetime(modified).timestamp()
        os.utime(path, (timestamp, timestamp))
    except (TypeError, ValueError, OverflowError):
        pass


def _validate_complete(path: Path, entry: RemoteEntry) -> tuple[bool, str | None]:
    if entry.size is None or path.stat().st_size != entry.size:
        return False, None
    digest = sha1_file(path) if entry.sha1 is not None else None
    return entry.sha1 is None or digest == entry.sha1, digest


def download_entry(
    entry: RemoteEntry,
    destination: Path,
    auth: str,
    timeout: float,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    if entry.is_directory or entry.size is None:
        raise ValueError(f"Expected a sized remote file: {entry.relative_path}")
    target = local_path(destination, entry.relative_path)
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        valid, digest = _validate_complete(target, entry)
        if valid:
            return {
                **asdict(entry),
                "status": "skipped_verified" if entry.sha1 else "skipped_same_size",
                "local_size": target.stat().st_size,
                "local_sha1": digest,
            }

    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            if start > entry.size:
                part.unlink()
                start = 0
            headers = {
                "Authorization": auth,
                "User-Agent": "TopAneu-SWITCHdrive-sync/1",
            }
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(entry.url, headers=headers, method="GET")
            try:
                response = urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                if error.code == 416 and start == entry.size:
                    valid, _ = _validate_complete(part, entry)
                    if valid:
                        os.replace(part, target)
                        set_remote_mtime(target, entry.modified)
                        digest = entry.sha1 or sha1_file(target)
                        return {
                            **asdict(entry),
                            "status": "downloaded",
                            "local_size": target.stat().st_size,
                            "local_sha1": digest,
                        }
                    part.unlink(missing_ok=True)
                    raise OSError("Complete partial failed SHA1 validation") from error
                raise
            with response:
                status = getattr(response, "status", response.getcode())
                if start and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise OSError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif start and status == 200:
                    start = 0
                    mode = "wb"
                elif not start and status in {200, 206}:
                    mode = "wb"
                else:
                    raise OSError(f"Unexpected HTTP status {status}")
                with part.open(mode) as handle:
                    while chunk := response.read(CHUNK_SIZE):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            valid, digest = _validate_complete(part, entry)
            if not valid:
                if part.stat().st_size == entry.size:
                    part.unlink()
                raise OSError(
                    f"Downloaded validation failed for {entry.relative_path}: "
                    f"size={part.stat().st_size if part.exists() else 'reset'}/"
                    f"{entry.size}, sha1={digest}/{entry.sha1}"
                )
            # Even when target is a hardlink into a canonical dataset, replacing
            # it with the completed .part creates a new inode and preserves source.
            os.replace(part, target)
            set_remote_mtime(target, entry.modified)
            return {
                **asdict(entry),
                "status": "downloaded",
                "local_size": target.stat().st_size,
                "local_sha1": digest,
            }
        except BaseException as error:
            last_error = error
            if attempt >= retries or not retryable(error):
                break
            backoff_sleep(attempt, backoff)
    assert last_error is not None
    raise last_error


def extraneous_files(destination: Path, remote_paths: set[str]) -> list[Path]:
    ignored = {STAGING_MARKER, DEFAULT_MANIFEST}
    extras = []
    for path in destination.rglob("*"):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        relative = path.relative_to(destination).as_posix()
        if relative not in ignored and relative not in remote_paths:
            extras.append(path)
    return sorted(extras)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--share-token", default=DEFAULT_SHARE_TOKEN)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--expected-top-level-dirs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--initialize-staging",
        action="store_true",
        help="Register a non-empty hardlink seed whose directory name contains staging",
    )
    parser.add_argument(
        "--delete-extraneous",
        action="store_true",
        help="Delete only-local files, but only inside a validated staging directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.share_token:
        raise ValueError("--share-token must not be empty")
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be in [1, {MAX_WORKERS}]")
    if args.retries < 0 or args.backoff < 0 or args.timeout <= 0:
        raise ValueError("Invalid retry, backoff, or timeout value")
    if args.expected_top_level_dirs < 0:
        raise ValueError("--expected-top-level-dirs must be non-negative")
    base_url = normalized_base_url(args.base_url)
    destination = prepare_staging(
        args.destination,
        base_url,
        args.share_token,
        args.dry_run,
        args.initialize_staging,
    )
    auth = authorization_header(args.share_token)
    started_at = utc_now()
    entries, top_level_directories = discover_tree(
        base_url, auth, args.timeout, args.retries, args.backoff
    )
    if (
        args.expected_top_level_dirs
        and len(top_level_directories) != args.expected_top_level_dirs
    ):
        raise RuntimeError(
            f"Expected {args.expected_top_level_dirs} top-level directories, found "
            f"{len(top_level_directories)}: {top_level_directories}"
        )
    reserved = {STAGING_MARKER.casefold(), DEFAULT_MANIFEST.casefold()}
    files = [entry for entry in entries if not entry.is_directory]
    relative_paths = {entry.relative_path for entry in files}
    folded_paths: dict[str, str] = {}
    for entry in entries:
        folded = entry.relative_path.casefold()
        if folded in folded_paths and folded_paths[folded] != entry.relative_path:
            raise ValueError(
                f"Case-insensitive local path collision: {folded_paths[folded]} / "
                f"{entry.relative_path}"
            )
        folded_paths[folded] = entry.relative_path
        if entry.relative_path.casefold() in reserved:
            raise ValueError(f"Remote path collides with sync metadata: {entry.relative_path}")
    extras = extraneous_files(destination, relative_paths) if destination.exists() else []
    print(
        f"Remote: {len(top_level_directories)} top-level directories, "
        f"{len(files)} files, {sum(entry.size or 0 for entry in files)} bytes"
    )
    print(f"Only-local files in staging: {len(extras)}")
    if args.dry_run:
        for entry in files:
            target = local_path(destination, entry.relative_path)
            state = "missing"
            if target.exists():
                state = "same-size" if target.stat().st_size == entry.size else "changed"
            print(f"{state:9s} {entry.relative_path}")
        for path in extras:
            print(f"only-local {path.relative_to(destination).as_posix()}")
        return

    for entry in entries:
        if entry.is_directory:
            local_path(destination, entry.relative_path).mkdir(parents=True, exist_ok=True)
    deleted: list[str] = []
    if args.delete_extraneous:
        for path in extras:
            relative = path.relative_to(destination).as_posix()
            path.unlink()
            deleted.append(relative)

    manifest_path = destination / DEFAULT_MANIFEST
    manifest: dict[str, Any] = {
        "format": "topaneu-switchdrive-sync-v1",
        "status": "running",
        "started_at": started_at,
        "base_url": base_url,
        "share_token_sha256": token_fingerprint(args.share_token),
        "destination": str(destination),
        "top_level_directories": top_level_directories,
        "expected_top_level_directories": args.expected_top_level_dirs,
        "workers": args.workers,
        "retries": args.retries,
        "remote_entries": [asdict(entry) for entry in entries],
        "files": {},
        "only_local": [path.relative_to(destination).as_posix() for path in extras],
        "deleted_only_local": deleted,
    }
    atomic_json_dump(manifest, manifest_path)
    failures: list[tuple[str, str]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_entry,
                entry,
                destination,
                auth,
                args.timeout,
                args.retries,
                args.backoff,
            ): entry
            for entry in files
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                result = {
                    **asdict(entry),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append((entry.relative_path, str(error)))
            manifest["files"][entry.relative_path] = result
            completed += 1
            print(f"[{completed}/{len(files)}] {result['status']}: {entry.relative_path}")
            if completed % 25 == 0 or failures:
                atomic_json_dump(manifest, manifest_path)
    manifest["finished_at"] = utc_now()
    manifest["status"] = "failed" if failures else "completed"
    manifest["summary"] = {
        "remote_files": len(files),
        "downloaded": sum(
            item.get("status") == "downloaded" for item in manifest["files"].values()
        ),
        "skipped": sum(
            str(item.get("status", "")).startswith("skipped")
            for item in manifest["files"].values()
        ),
        "failed": len(failures),
        "only_local": len(extras),
        "deleted_only_local": len(deleted),
    }
    atomic_json_dump(manifest, manifest_path)
    if failures:
        preview = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        raise RuntimeError(f"{len(failures)} downloads failed; see {manifest_path}: {preview}")
    print(f"Completed SWITCHdrive staging sync: {manifest_path}")


if __name__ == "__main__":
    main()
