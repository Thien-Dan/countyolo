"""
Theo dõi kernel Kaggle của dự án trên mọi tài khoản trong kaggle/*.json: in trạng thái, tải log + file nhỏ của kernel
đã xong về outputs/kaggle/<tài khoản>__<kernel>/ (bỏ thư mục lớn CountVid/, data/, countyolo/).

    python tools/kaggle_watch.py                     # trạng thái các kernel countyolo/countvid gần đây
    python tools/kaggle_watch.py --fetch nadthien/countyolo-density-vit-gate
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("CountVid/", "data/", "countyolo/", "vc/")


def accounts():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "kaggle", "*.json"))):
        k = json.load(open(f))
        out[k["username"]] = k["key"]
    return out


def api_for(user, key):
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def status(limit=6):
    for user, key in accounts().items():
        api = api_for(user, key)
        try:
            ks = api.kernels_list(mine=True, sort_by="dateRun", page_size=limit)
        except Exception as e:  # mạng chập chờn
            print(f"== {user}: lỗi {type(e).__name__}")
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
                open(p, "wb").write(requests.get(f.url, timeout=300).content)
                got.append(f.file_name)
            token = resp.next_page_token
            if not token:
                break
    print(f"Đã tải {len(got)} file về {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", default=None, help="<tài khoản>/<kernel>")
    a = p.parse_args()
    fetch(a.fetch) if a.fetch else status()
    sys.exit(0)
