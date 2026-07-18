"""
Repository bootstrap script for the ATLAS development-data bundle.

Run this on a real developer machine or CI runner (never inside this
sandboxed agent environment, which has no network path to GitHub Releases
or Google Drive). It downloads a private GitHub Release asset containing a
versioned ATLAS development-data bundle, verifies every checksum, extracts
the data OUTSIDE the repository, and documents/sets ``ATLAS_DATA_ROOT`` so
``atlas/config/paths.py`` resolves against the extracted data instead of
the Google Drive default.

Authentication
---------------

Requires a GitHub token with at least read access to the private release,
supplied via ``--token`` or the ``GITHUB_TOKEN`` / ``GH_TOKEN`` environment
variable (in that priority order).

Failure modes
-------------

This script fails clearly and distinctly for:

- missing/invalid authentication (exit code 2)
- release or asset not found (exit code 3)
- checksum mismatch, for a part, the reassembled archive, or an extracted
  artifact (exit code 4)
- files missing after extraction (exit code 5)
- any other network/HTTP error (exit code 6)

Usage
-----

    export GITHUB_TOKEN=ghp_xxx
    python scripts/dev_data_bundle/bootstrap_dev_data_bundle.py \\
        --repo brianrags07-afk/Project-ATLAS \\
        --tag v1.0.0 \\
        --dest ~/atlas_dev_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev_data_bundle.manifest import (  # noqa: E402
    sha256_of_bytes,
    sha256_of_file,
    validate_manifest_or_raise,
)

GITHUB_API_ROOT = "https://api.github.com"
DEFAULT_MANIFEST_ASSET_NAME = "release_manifest.json"
ENV_FILE_NAME = "atlas_dev_data.env"
# GitHub's REST API expects the HTTP Authorization header in the form
# "<scheme> <token>". Kept as a separate constant (rather than inline in
# the header dict) purely for readability of github_request().
AUTH_SCHEME = "Bearer"


class BootstrapError(RuntimeError):
    """Base class for all clearly-classified bootstrap failures."""

    exit_code = 6


class AuthenticationError(BootstrapError):
    exit_code = 2


class AssetNotFoundError(BootstrapError):
    exit_code = 3


class ChecksumMismatchError(BootstrapError):
    exit_code = 4


class MissingDataError(BootstrapError):
    exit_code = 5


def resolve_token(explicit_token: str | None) -> str:
    token = (
        explicit_token
        or os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
    )

    if not token:
        raise AuthenticationError(
            "No GitHub token supplied. Pass --token or set GITHUB_TOKEN / "
            "GH_TOKEN. A private-release download requires authenticated "
            "GitHub access; this tool never falls back to unauthenticated "
            "requests."
        )

    return token


def github_request(
    url: str,
    token: str,
    accept: str = "application/vnd.github+json",
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": AUTH_SCHEME + " " + token,
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "atlas-dev-data-bundle-bootstrap",
        },
    )

    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise AuthenticationError(
                f"GitHub rejected the supplied token ({exc.code} {exc.reason}) "
                f"for {url}. Confirm the token has access to this private "
                "repository's releases."
            ) from exc
        if exc.code == 404:
            raise AssetNotFoundError(
                f"GitHub returned 404 Not Found for {url}. Confirm the "
                "repository, release tag, and asset name are correct."
            ) from exc
        raise BootstrapError(
            f"GitHub request failed ({exc.code} {exc.reason}) for {url}."
        ) from exc
    except urllib.error.URLError as exc:
        raise BootstrapError(f"Network error requesting {url}: {exc}") from exc


def get_release(
    repo: str,
    tag: str,
    token: str,
) -> dict[str, Any]:
    if tag == "latest":
        url = f"{GITHUB_API_ROOT}/repos/{repo}/releases/latest"
    else:
        url = f"{GITHUB_API_ROOT}/repos/{repo}/releases/tags/{tag}"

    payload = github_request(url, token)
    return json.loads(payload)


def find_asset(
    release: dict[str, Any],
    asset_name: str,
) -> dict[str, Any]:
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            return asset

    available = [asset.get("name") for asset in release.get("assets", [])]

    raise AssetNotFoundError(
        f"Release '{release.get('tag_name')}' has no asset named "
        f"'{asset_name}'. Available assets: {available}"
    )


def download_asset(
    asset: dict[str, Any],
    token: str,
) -> bytes:
    return github_request(
        asset["url"],
        token,
        accept="application/octet-stream",
    )


def reassemble_archive(
    manifest: dict[str, Any],
    release: dict[str, Any],
    token: str,
    archive_path: Path,
) -> None:
    part_files = manifest.get("part_files") or []

    if not part_files:
        asset_name = f"{manifest['bundle_name']}-{manifest['bundle_version']}.tar.gz"
        asset = find_asset(release, asset_name)
        data = download_asset(asset, token)

        actual_sha256 = sha256_of_bytes(data)
        expected_sha256 = manifest.get("archive_sha256")

        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ChecksumMismatchError(
                f"Archive '{asset_name}' checksum mismatch: expected "
                f"{expected_sha256}, got {actual_sha256}."
            )

        archive_path.write_bytes(data)
        return

    ordered_parts = sorted(part_files, key=lambda part: part["part_index"])

    with archive_path.open("wb") as handle:
        for part in ordered_parts:
            asset = find_asset(release, part["filename"])
            data = download_asset(asset, token)

            actual_sha256 = sha256_of_bytes(data)

            if actual_sha256 != part["sha256"]:
                raise ChecksumMismatchError(
                    f"Part '{part['filename']}' checksum mismatch: expected "
                    f"{part['sha256']}, got {actual_sha256}."
                )

            handle.write(data)

    reassembled_sha256 = sha256_of_file(archive_path)
    expected_sha256 = manifest.get("archive_sha256")

    if expected_sha256 and reassembled_sha256 != expected_sha256:
        raise ChecksumMismatchError(
            "Reassembled archive checksum mismatch: expected "
            f"{expected_sha256}, got {reassembled_sha256}."
        )


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract_all(archive, destination)


def _safe_extract_all(
    archive: tarfile.TarFile,
    destination: Path,
) -> None:
    resolved_destination = destination.resolve()

    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()

        if not str(member_path).startswith(str(resolved_destination)):
            raise BootstrapError(
                f"Refusing to extract archive member outside destination: "
                f"{member.name}"
            )

    archive.extractall(destination, filter="data")  # noqa: S202 - paths validated above


def verify_extracted_artifacts(
    manifest: dict[str, Any],
    extraction_root: Path,
) -> None:
    """
    Verify every artifact exists (and checksum-matches) under
    ``extraction_root``. ``bundled_relative_path`` already carries its own
    ``data/`` prefix, so ``extraction_root`` here is the directory the
    archive was extracted into, not the ATLAS data root itself.
    """

    missing: list[str] = []
    mismatched: list[str] = []

    for artifact in manifest.get("artifacts", []):
        artifact_path = extraction_root / artifact["bundled_relative_path"]

        if not artifact_path.exists():
            missing.append(str(artifact_path))
            continue

        actual_sha256 = sha256_of_file(artifact_path)

        if actual_sha256 != artifact["sha256"]:
            mismatched.append(
                f"{artifact_path} (expected {artifact['sha256']}, got "
                f"{actual_sha256})"
            )

    if missing:
        raise MissingDataError(
            "The following required artifacts are missing after "
            f"extraction:\n" + "\n".join(f"- {path}" for path in missing)
        )

    if mismatched:
        raise ChecksumMismatchError(
            "The following extracted artifacts failed checksum "
            "verification:\n" + "\n".join(f"- {entry}" for entry in mismatched)
        )


def write_env_file(
    dest: Path,
    data_root: Path,
) -> Path:
    env_path = dest / ENV_FILE_NAME

    with env_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Generated by scripts/dev_data_bundle/bootstrap_dev_data_bundle.py\n"
            f"export ATLAS_DATA_ROOT={data_root}\n"
        )

    return env_path


def bootstrap(
    *,
    repo: str,
    tag: str,
    dest: Path,
    token: str,
    manifest_asset_name: str = DEFAULT_MANIFEST_ASSET_NAME,
) -> dict[str, Any]:
    release = get_release(repo, tag, token)
    manifest_asset = find_asset(release, manifest_asset_name)
    manifest = json.loads(download_asset(manifest_asset, token))

    validate_manifest_or_raise(manifest)

    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / f"{manifest['bundle_name']}-{manifest['bundle_version']}.tar.gz"

    reassemble_archive(manifest, release, token, archive_path)

    extraction_root = dest / manifest["bundle_version"]
    data_root = extraction_root / "data"
    extract_archive(archive_path, extraction_root)

    verify_extracted_artifacts(manifest, extraction_root)

    archive_path.unlink(missing_ok=True)

    env_path = write_env_file(dest, data_root)

    os.environ["ATLAS_DATA_ROOT"] = str(data_root)

    print(
        f"Bootstrapped {manifest['artifact_count']} verified artifact(s) "
        f"(bundle_version={manifest['bundle_version']}) into {data_root}."
    )
    print(f"ATLAS_DATA_ROOT set for this process: {data_root}")
    print(f"To persist in your shell, run: source {env_path}")

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--repo",
        required=True,
        help="owner/repo of the private GitHub repository, e.g. "
        "brianrags07-afk/Project-ATLAS.",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="GitHub Release tag to download (default: latest).",
    )
    parser.add_argument(
        "--dest",
        default=str(Path.home() / "atlas_dev_data"),
        help="Directory OUTSIDE the repository to extract data into "
        "(default: ~/atlas_dev_data).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token. Falls back to GITHUB_TOKEN / GH_TOKEN env vars.",
    )
    parser.add_argument(
        "--manifest-asset-name",
        default=DEFAULT_MANIFEST_ASSET_NAME,
        help="Name of the manifest asset attached to the release.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        token = resolve_token(args.token)

        bootstrap(
            repo=args.repo,
            tag=args.tag,
            dest=Path(args.dest).expanduser(),
            token=token,
            manifest_asset_name=args.manifest_asset_name,
        )
    except BootstrapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
