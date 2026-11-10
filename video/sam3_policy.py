"""SAM3 lifecycle policy, independent of torch and ground-truth counts.

Settings common to Meta's SAM3 temporal-disambiguation builder at
f1647e8d0c1d280991f4d1b7d974c2704537fe66. This remains an Ultralytics
implementation; matching exposed settings does not make it the Meta backend.
"""
from collections import deque

META_ALIGNED = dict(
    det_nms_thresh=0.1, assoc_iou_thresh=0.1, trk_assoc_iou_thresh=0.1,
    new_det_thresh=0.7, hotstart_delay=15, hotstart_unmatch_thresh=8,
    hotstart_dup_thresh=8, init_trk_keep_alive=30, max_trk_keep_alive=30,
    min_trk_keep_alive=-1,
    suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
    decrease_trk_keep_alive_for_empty_masklets=False,
    recondition_every_nth_frame=16, masklet_confirmation_enable=False,
)


class DelayedTrackFrames:
    """Delay output so early duplicate/unstable tracks can be revoked.

    Only pending frames are revised. Already emitted sightings remain valid
    when a track later expires or leaves the scene, as in Meta propagation.
    Payloads must be CPU records, not retained GPU masks or Results objects.
    """
    def __init__(self, delay):
        if delay < 0:
            raise ValueError('delay must be nonnegative')
        self.delay = delay
        self.pending = deque()
        self.removed = set()

    def push(self, record, removed=()):
        self.removed.update(int(i) for i in removed)
        self.pending.append(record)
        return [self._pop()] if len(self.pending) >= max(1, self.delay) else []

    def _pop(self):
        record = self.pending.popleft()
        indices = [i for i, obj in enumerate(record['ids']) if obj not in self.removed]
        return dict(record, ids=[record['ids'][i] for i in indices],
                    boxes=[record['boxes'][i] for i in indices])

    def finish(self):
        return [self._pop() for _ in range(len(self.pending))]


def add_object_batches(add, frame_idx, num_frames, obj_ids, masks, states, batch_size=0):
    """Bound per-state attention batches without capping detections or changing IDs."""
    if batch_size < 0:
        raise ValueError('object batch size must be nonnegative')
    if len(obj_ids) != len(masks):
        raise ValueError('Object ID/mask count mismatch')
    if not batch_size:
        return add(frame_idx, num_frames, obj_ids, masks, states)
    for start in range(0, len(obj_ids), batch_size):
        states = add(frame_idx, num_frames, obj_ids[start:start+batch_size],
                     masks[start:start+batch_size], states)
    return states
