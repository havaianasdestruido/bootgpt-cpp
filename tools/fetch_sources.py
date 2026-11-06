#!/usr/bin/env python3
"""Recreate ignored training sources from the pinned upstream files."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from strip_cpp_comments import prepare_source

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "upstream_sources.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_original(entry):
    repo = entry["repository"].removeprefix("https://github.com/")
    path = quote(entry["path"], safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={entry['commit']}"
    request = Request(url, headers={"User-Agent": "bootgpt-cpp"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("encoding") != "base64":
        raise ValueError(f"Unexpected GitHub response for {entry['source_url']}")
    return base64.b64decode(payload["content"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, help="read originals from a local "
                        "directory arranged as PROJECT/UPSTREAM_PATH")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in entries:
        dest = ROOT / entry["snippet"]
        if dest.is_file() and digest(dest.read_bytes()) == entry["snippet_sha256"]:
            continue
        if args.from_dir:
            raw = (args.from_dir / entry["project"] / entry["path"]).read_bytes()
        else:
            raw = fetch_original(entry)
        if digest(raw) != entry["original_sha256"]:
            raise ValueError(f"Upstream hash mismatch: {entry['source_url']}")
        prepared = prepare_source(raw.decode("utf-8")).encode("utf-8")
        if digest(prepared) != entry["snippet_sha256"]:
            raise ValueError(f"Prepared hash mismatch: {entry['snippet']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(prepared)
        print(f"Prepared {entry['snippet']}")
    print(f"Ready: {len(entries)} upstream files in source_code/.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, URLError, ValueError, KeyError) as exc:
        print(f"Failed to prepare upstream sources: {exc}", file=sys.stderr)
        sys.exit(1)
