"""Conservative, offline tracklet association. No detector or GT-count feedback.

SAM IDs remain untouched. Only the final counting identities are stitched.
Generic DINO appearance is evidence, not a trained ReID probability.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class IdentityConfig:
    short_gap: int = 8
    max_gap: int = 90
    confirmations: int = 3
    short_similarity: float = .95
    long_similarity: float = .98
    margin: float = .03
    motion_radius: float = 1.
    edge_fraction: float = .08

    def __post_init__(self):
        if not 0 < self.short_gap <= self.max_gap or self.confirmations < 2:
            raise ValueError('Invalid gap/confirmation limits')
        if not -1 <= self.short_similarity <= self.long_similarity <= 1:
            raise ValueError('Invalid similarity limits')
        if self.margin <= 0 or self.motion_radius <= 0 or not 0 < self.edge_fraction < .5:
            raise ValueError('Invalid identity geometry/margin')


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    return x / norm if x.ndim == 1 and np.isfinite(x).all() and norm > 1e-8 else None


class TemporalIdentity:
    def __init__(self, config=None):
        self.config = config or IdentityConfig()
        self.tracks = {}
        self.last_frame = -1
        self.shape = None

    def observe(self, frame, ids, boxes, features, shape):
        if frame <= self.last_frame:
            raise ValueError('Frames must increase strictly')
        if len(set(ids)) != len(ids) or not len(ids) == len(boxes) == len(features):
            raise ValueError('IDs/boxes/features must be aligned and unique per frame')
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError('Expected positive image height,width')
        if self.shape is not None and tuple(shape) != self.shape:
            raise ValueError('Image dimensions changed')
        self.shape = tuple(shape); self.last_frame = frame
        for raw_id, box, feature in zip(ids, boxes, features):
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] < box[:2]):
                raise ValueError('Invalid xyxy box')
            # A thin SAM mask can yield a zero-width/height box. Keep its native
            # sighting/count, but do not use this observation as identity evidence.
            feature = _unit(feature) if feature is not None and np.all(box[2:] > box[:2]) else None
            observations = self.tracks.setdefault(int(raw_id), [])
            if observations and feature is not None:
                previous = next((o[2] for o in observations if o[2] is not None), None)
                if previous is not None and previous.shape != feature.shape:
                    raise ValueError('Appearance dimension changed')
            observations.append((int(frame), box.copy(), feature))

    def _edge(self, box):
        h, w = self.shape
        f = self.config.edge_fraction
        return box[0] <= w*f or box[1] <= h*f or box[2] >= w*(1-f) or box[3] >= h*(1-f)

    def _candidate(self, old, new):
        c = self.config; k = c.confirmations
        gap = new[0][0] - old[-1][0]
        if not 1 <= gap <= c.max_gap or min(len(old), len(new)) < k:
            return None
        end, start = old[-k:], new[:k]
        # Multiple consecutive observations, not repeated evidence from one frame.
        if any(group[-1][0] - group[0][0] != k-1 for group in (end, start)):
            return None
        if any(o[2] is None for o in end+start):
            return None
        reference = _unit(np.mean([o[2] for o in end], axis=0))
        if reference is None:
            return None
        similarity = min(float(reference @ o[2]) for o in start)
        long_gap = gap > c.short_gap
        threshold = c.long_similarity if long_gap else c.short_similarity
        if similarity < threshold:
            return None
        old_box = end[-1][1]; scale = max(float(np.linalg.norm(old_box[2:] - old_box[:2])), 1.)
        old_center = (old_box[:2] + old_box[2:])/2
        if long_gap:
            # Long-range re-entry requires exit and entry near image boundaries.
            if not self._edge(old_box) or not self._edge(start[0][1]):
                return None
            distance = 0.
        else:
            previous = (end[-2][1][:2] + end[-2][1][2:])/2
            velocity = old_center - previous
            distances = [np.linalg.norm((o[1][:2]+o[1][2:])/2 -
                         (old_center + velocity*(o[0]-end[-1][0]))) / scale for o in start]
            distance = max(distances)
            if distance > c.motion_radius:
                return None
        ratios = [(np.prod(o[1][2:]-o[1][:2]) / np.prod(old_box[2:]-old_box[:2])) for o in start]
        if min(ratios) < .5 or max(ratios) > 2.:
            return None
        # Reject long-range appearance matches if a concurrent lookalike could
        # explain the appearance just as well, even if it has another live ID.
        if long_gap:
            for other in self.tracks.values():
                if other is old or other is new:
                    continue
                if other[0][0] <= start[0][0] <= other[-1][0]:
                    nearby = min(other, key=lambda o: abs(o[0]-start[0][0]))
                    if nearby[2] is not None and float(reference @ nearby[2]) >= similarity-c.margin:
                        return None
        return dict(cost=1-similarity + .02*distance + .01*min(gap, c.short_gap), similarity=similarity,
                    gap=gap, kind='reentry' if long_gap else 'occlusion')

    def finalize(self, min_hits=1):
        if min_hits < 1:
            raise ValueError('min_hits must be positive')
        candidates = []
        for old_id, old in self.tracks.items():
            for new_id, new in self.tracks.items():
                if old_id == new_id:
                    continue
                edge = self._candidate(old, new)
                if edge is not None:
                    candidates.append(dict(old=old_id, new=new_id, **edge))
        links = []; uncertain = []
        for edge in sorted(candidates, key=lambda e: (e['cost'], e['old'], e['new'])):
            alternatives = [e['cost'] for e in candidates if e is not edge and
                            (e['old'] == edge['old'] or e['new'] == edge['new'])]
            if alternatives and min(alternatives)-edge['cost'] < self.config.margin:
                uncertain.append(dict(edge, reason='ambiguous_competitor'))
                continue
            links.append(edge)
        parent = {i:i for i in self.tracks}
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for edge in links:
            parent[root(edge['new'])] = root(edge['old'])
        mapping = {i:root(i) for i in self.tracks}
        frames = {}
        for raw_id, observations in self.tracks.items():
            frames.setdefault(mapping[raw_id], set()).update(o[0] for o in observations)
        return dict(count=sum(len(f) >= min_hits for f in frames.values()),
                    baseline_count=sum(len(o) >= min_hits for o in self.tracks.values()),
                    raw_to_identity=mapping, identity_hits={i:len(f) for i,f in frames.items()},
                    links=links, uncertain=uncertain, config=asdict(self.config),
                    mode='offline_tracklet_stitching',
                    note='Unmatched/ambiguous tracklets remain distinct. Generic appearance is not trained ReID.')
