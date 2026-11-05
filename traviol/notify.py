"""Send violation evidence to Dropbox (Python 3 port of original_2018/dropboxupload.py).

The access token is read from the DROPBOX_ACCESS_TOKEN environment variable and
is never stored in code. Create one at https://www.dropbox.com/developers/apps
with only the `files.content.write` permission, scoped to an app folder.
"""

from __future__ import annotations

import os
from pathlib import Path


def upload(local_path: str | Path, remote_folder: str = "/violations") -> str:
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set DROPBOX_ACCESS_TOKEN to upload evidence to Dropbox")
    import dropbox  # optional dependency: pip install dropbox

    local_path = Path(local_path)
    remote = f"{remote_folder.rstrip('/')}/{local_path.name}"
    with dropbox.Dropbox(token) as dbx, local_path.open("rb") as f:
        dbx.files_upload(f.read(), remote, mode=dropbox.files.WriteMode.overwrite)
    return remote
