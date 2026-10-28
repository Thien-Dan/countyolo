"""
Tracker nhẹ để đếm vật duy nhất trong video từ điểm/box của CountYOLO (thay SAM 2 của CountVid).

Ghép phát hiện frame t với track đang sống bằng Hungarian. Chi phí = khoảng cách tâm / kích thước vật, cộng thêm
(1 - cosine) giữa feature DINOv2 của phát hiện và của track khi có feature. Feature giúp nối lại track khi vật bị
bỏ sót vài frame (CountYOLO chỉ có recall điểm ~0.64 trên FSC-147 val) thay vì sinh track mới -> đếm trùng.
Số đếm = số track xuất hiện ít nhất min_hits frame (lọc phát hiện giả thoáng qua, giống bộ lọc thời gian w=3
của CountVid).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self, gate=1.0, max_miss=3, min_hits=3, app_weight=0.0, min_cos=0.0, feat_momentum=0.8):
        """
        gate: khoảng cách tâm tối đa (tính theo kích thước vật) để được ghép
        max_miss: số frame liên tiếp không ghép được trước khi track kết thúc
        min_hits: số frame tối thiểu để track được đếm
        app_weight: trọng số (1 - cosine) trong chi phí; 0 = chỉ dùng vị trí
        min_cos: cosine tối thiểu để được ghép (chỉ khi có feature)
        feat_momentum: feature của track = EMA của feature các lần ghép
        """
        self.gate, self.max_miss, self.min_hits = gate, max_miss, min_hits
        self.app_weight, self.min_cos, self.momentum = app_weight, min_cos, feat_momentum
        self.active, self.done = [], []

    def update(self, centers, sides, feats=None):
        """centers (N, 2), sides (N,), feats (N, D) đã chuẩn hoá L2 hoặc None. Trả về id track của từng phát hiện."""
        n = len(centers)
        ids = [-1] * n
        matched_t, matched_d = set(), set()
        if self.active and n:
            tc = np.array([t["c"] for t in self.active])
            ts = np.array([t["s"] for t in self.active])
            dist = np.linalg.norm(tc[:, None] - centers[None], axis=2) / np.maximum(ts[:, None], sides[None])
            # Track mất dấu lâu được phép ghép xa hơn một chút (vật có thể đã di chuyển)
            miss = np.array([t["miss"] for t in self.active])
            ok = dist <= self.gate * (1 + 0.25 * miss[:, None])
            cost = dist.copy()
            if feats is not None and self.app_weight > 0 and all(t["f"] is not None for t in self.active):
                cos = np.array([t["f"] for t in self.active]) @ feats.T
                ok &= cos >= self.min_cos
                cost = cost + self.app_weight * (1 - cos)
            r, c = linear_sum_assignment(np.where(ok, cost, 1e6))
            for i, j in zip(r, c):
                if ok[i, j]:
                    t = self.active[i]
                    t["c"], t["s"] = centers[j], 0.5 * (t["s"] + sides[j])
                    if feats is not None and t["f"] is not None:
                        f = self.momentum * t["f"] + (1 - self.momentum) * feats[j]
                        t["f"] = f / max(np.linalg.norm(f), 1e-6)
                    t["hits"] += 1
                    t["miss"] = 0
                    ids[j] = t["id"]
                    matched_t.add(i)
                    matched_d.add(j)
        alive = []
        for i, t in enumerate(self.active):
            if i not in matched_t:
                t["miss"] += 1
            (alive if t["miss"] <= self.max_miss else self.done).append(t)
        next_id = len(self.done) + len(alive)
        for j in range(n):
            if j not in matched_d:
                alive.append({"id": next_id, "c": centers[j], "s": sides[j], "hits": 1, "miss": 0,
                              "f": feats[j] if feats is not None else None})
                ids[j] = next_id
                next_id += 1
        self.active = alive
        return ids

    def count(self):
        return sum(t["hits"] >= self.min_hits for t in self.done + self.active)
