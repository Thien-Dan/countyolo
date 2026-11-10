"""
Theo dõi kernel Kaggle của dự án trên mọi tài khoản trong kaggle/*.json: in trạng thái, tải log + file nhỏ của kernel
đã xong về outputs/kaggle/<tài khoản>__<kernel>/ (bỏ thư mục lớn CountVid/, data/, countyolo/).

    python tools/kaggle_watch.py                     # trạng thái các kernel countyolo/countvid gần đây
    python tools/kaggle_watch.py --fetch nadthien/countyolo-density-vit-gate
"""
import argparse
from contextlib import contextmanager
from unittest.mock import patch
import glob
import json
import os
from pathlib import Path
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("CountVid/", "data/", "countyolo/", "vc/")


def accounts():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "kaggle", "*.json"))):
        k = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        out[k["username"]] = k["key"]
    return out


def api_for(user, key):
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    # The SDK consults ~/.kaggle/access_token even when explicit legacy
    # credentials were supplied. A stale global token can silently override
    # every account. Scope legacy authentication to this client's calls only.
    api.config_values[api.CONFIG_NAME_USER] = user
    api.config_values[api.CONFIG_NAME_KEY] = key
    api.config_values.pop(api.CONFIG_NAME_TOKEN, None)
    build = api.build_kaggle_client

    @contextmanager
    def legacy_client():
        with patch("kagglesdk.kaggle_http_client.get_access_token_from_env", return_value=(None, None)):
            with build() as client:
                yield client

    api.build_kaggle_client = legacy_client
    return api


def status(limit=6):
    for user, key in accounts().items():
        api = api_for(user, key)
        try:
            ks = api.kernels_list(mine=True, sort_by="dateRun", page_size=limit)
        except Exception as e:  # mạng chập chờn
            print(f"== {user}: lỗi {type(e).__name__}, HTTP {getattr(getattr(e, 'response', None), 'status_code', '?')}")
            continue
        print(f"== {user}")
        for k in ks:
            if not any(t in k.ref for t in ("countyolo", "countvid")):
                continue
            try:
                st = str(api.kernels_status(k.ref).status).split(".")[-1]
            except Exception:
                st = "?"
            print(f"   {k.ref:55s} {st:20s} {k.last_run_time}")


def fetch_log(ref):
    """Read the current session log without downloading model/data artifacts."""
    user, slug = ref.split("/")
    api = api_for(user, accounts()[user])
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = user, slug
    with api.build_kaggle_client() as kg:
        response = kg.kernels.kernels_api_client.list_kernel_session_output(req)
    raw = response.log or ""
    try:
        raw = "".join(e.get("data", "") for e in json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        pass
    out = os.path.join(ROOT, "outputs", "kaggle", f"{user}__{slug}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "live.log")
    if raw:
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
    return path, raw


def fetch(ref):
    user, slug = ref.split("/")
    key = accounts()[user]
    api = api_for(user, key)
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    out = os.path.join(ROOT, "outputs", "kaggle", f"{user}__{slug}")
    os.makedirs(out, exist_ok=True)
    token, got = None, []
    with api.build_kaggle_client() as kg:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = user, slug
            if token:
                req.page_token = token
            resp = kg.kernels.kernels_api_client.list_kernel_session_output(req)
            if resp.log:
                t = resp.log
                try:
                    t = "".join(e.get("data", "") for e in json.loads(t))
                except Exception:
                    pass
                open(os.path.join(out, "log.txt"), "w", encoding="utf-8").write(t)
                got.append("log.txt")
            for f in resp.files:
                if f.file_name.startswith(SKIP):
                    continue
                p = os.path.join(out, f.file_name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                response = requests.get(f.url, timeout=60)
                response.raise_for_status()
                open(p, "wb").write(response.content)
                got.append(f.file_name)
            token = resp.next_page_token
            if not token:
                break
    print(f"Đã tải {len(got)} file về {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", default=None, help="<tài khoản>/<kernel>")
    p.add_argument("--log", default=None, help="Read current session log only")
    a = p.parse_args()
    if a.log:
        path, log = fetch_log(a.log)
        print(path)
        print(log[-5000:])
    elif a.fetch:
        fetch(a.fetch)
    else:
        status()
    sys.exit(0)
