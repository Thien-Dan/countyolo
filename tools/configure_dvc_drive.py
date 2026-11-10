"""Read a Desktop OAuth JSON and save credentials only in .dvc/config.local."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(oauth_json, remote="drive"):
    from dvc.config import Config

    payload = json.loads(Path(oauth_json).read_text(encoding="utf-8-sig"))
    client = payload.get("installed", {})
    client_id, client_secret = client.get("client_id"), client.get("client_secret")
    if not isinstance(client_id, str) or not client_id.endswith(".apps.googleusercontent.com"):
        raise ValueError("Expected an OAuth Desktop app JSON downloaded from Google Cloud")
    if not isinstance(client_secret, str) or not client_secret:
        raise ValueError("OAuth JSON is missing client_secret")
    config = Config(dvc_dir=str(ROOT / ".dvc"))
    if remote not in config.get("remote", {}):
        raise ValueError("DVC remote must be configured first")
    with config.edit("local") as local:
        settings = local.setdefault("remote", {}).setdefault(remote, {})
        settings["gdrive_client_id"] = client_id
        settings["gdrive_client_secret"] = client_secret
    print(f"Configured OAuth for {remote} in .dvc/config.local; no secrets printed or committed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oauth-json", type=Path, required=True)
    parser.add_argument("--remote", default="drive")
    args = parser.parse_args()
    configure(args.oauth_json, args.remote)


if __name__ == "__main__":
    main()
