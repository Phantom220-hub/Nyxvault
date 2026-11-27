#!/usr/bin/env python3
"""Upload a ZIP of NYXVAULT model artifacts to a GitHub Release.

The GitHub token is read only from the GITHUB_TOKEN environment variable.
Never hard-code a token or commit it to the repository.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests


API = "https://api.github.com"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish NYXVAULT model artifacts to a GitHub Release.")
    parser.add_argument("--repo", default="Phantom220-hub/Nyxvault")
    parser.add_argument("--tag", required=True, help="Release tag, e.g. nyxvault-models-v1")
    parser.add_argument("--asset", required=True, help="ZIP archive to upload")
    parser.add_argument("--name", default="", help="Optional asset filename; defaults to archive filename")
    parser.add_argument("--release-name", default="NYXVAULT model artifacts")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN is missing. Store it in Colab Secrets; do not paste it into notebook code.")

    asset_path = Path(args.asset).expanduser().resolve()
    if not asset_path.is_file():
        raise SystemExit(f"Artifact archive not found: {asset_path}")
    asset_name = args.name.strip() or asset_path.name
    if not asset_name or "/" in asset_name or "\\" in asset_name:
        raise SystemExit("Asset name must be a simple filename.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    session = requests.Session()
    session.headers.update(headers)
    release_url = f"{API}/repos/{args.repo}/releases/tags/{quote(args.tag, safe='')}"
    response = session.get(release_url, timeout=30)

    if response.status_code == 404:
        response = session.post(
            f"{API}/repos/{args.repo}/releases",
            json={
                "tag_name": args.tag,
                "name": args.release_name,
                "body": "NYXVAULT training outputs. Checkpoint and dataset versions are identified by the archive manifest.",
                "draft": False,
                "prerelease": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        release = response.json()
    else:
        response.raise_for_status()
        release = response.json()

    # Replace an existing asset with the same name to make the upload repeatable.
    for existing in release.get("assets", []):
        if existing.get("name") == asset_name:
            deleted = session.delete(existing["url"], timeout=30)
            deleted.raise_for_status()

    upload_url = release["upload_url"].split("{", 1)[0]
    params = {"name": asset_name}
    upload_headers = dict(headers)
    upload_headers["Content-Type"] = "application/zip"

    size = asset_path.stat().st_size
    print(f"Uploading {asset_name} ({size / (1024 * 1024):.2f} MiB) to {args.repo} release {args.tag}...", flush=True)
    with asset_path.open("rb") as handle:
        uploaded = requests.post(
            upload_url,
            params=params,
            headers=upload_headers,
            data=handle,
            timeout=(30, 1800),
        )
    uploaded.raise_for_status()
    result = uploaded.json()
    print("Upload complete.")
    print("Release:", release.get("html_url", ""))
    print("Asset:", result.get("browser_download_url", ""))


if __name__ == "__main__":
    main()
