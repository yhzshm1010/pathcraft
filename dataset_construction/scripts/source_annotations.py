"""Read processed CWQ TRAIN annotations without assuming an archive folder name."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile


def load_source_annotations(
    path: Path, member: str = ""
) -> tuple[list[dict], str, bytes]:
    """Return source rows, the selected member name, and its original bytes.

    A JSON array or a ZIP containing CWQ_train_extracted.json is accepted.
    Ambiguous archives require an explicit member rather than a guessed source.
    Dataset provenance remains documented in data/train/NOTICE.md.
    """
    if path.suffix.casefold() == ".json":
        raw = path.read_bytes()
        selected = path.name
    else:
        with zipfile.ZipFile(path) as archive:
            if member:
                selected = member
            else:
                candidates = [
                    name for name in archive.namelist()
                    if not name.startswith("__MACOSX/")
                    and name.rsplit("/", 1)[-1] == "CWQ_train_extracted.json"
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "source archive must contain one CWQ_train_extracted.json; "
                        "use --source-member to resolve an ambiguous archive"
                    )
                selected = candidates[0]
            raw = archive.read(selected)
    rows = json.loads(raw)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("source annotations must be a JSON array of objects")
    return rows, selected, raw
