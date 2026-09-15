from collections import Counter
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from surgint.model import Detections
from surgint.model.boxes import iou_matrix
from surgint.runtime.kalman import MEASUREMENT_DIM, KalmanFilter, to_box, to_measurement

# detections a track must collect before it is believed rather than suspected
CONFIRM_HITS = 3

# the rescue pass is stricter than the first. a low scoring box is weak evidence, so it
# is only allowed to continue a track it already sits almost on top of
RESCUE_MATCH_THRESH = 0.5


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class Track:
    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        score: float,
        class_id: int,
        kalman: KalmanFilter,
    ):
        self.track_id = track_id
        self.score = score
        self.kalman = kalman
        self.mean, self.covariance = kalman.initiate(to_measurement(box))

        self.votes = Counter([class_id])
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE

    @property
    def box(self) -> np.ndarray:
        """current estimate as xyxy"""
        return to_box(self.mean[:MEASUREMENT_DIM])

    @property
    def class_id(self) -> int:
        """majority class over matched detections; a per-frame flip must not split the track"""
        return self.votes.most_common(1)[0][0]

    def predict(self) -> None:
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float, class_id: int) -> None:
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, to_measurement(box)
        )
        self.score = score
        self.votes[class_id] += 1
        self.hits += 1
        self.time_since_update = 0

        # the state follows the hit count, so a lost track that is matched again
        # comes back confirmed instead of starting over as tentative
        self.state = TrackState.CONFIRMED if self.hits >= CONFIRM_HITS else TrackState.TENTATIVE

    def mark_lost(self) -> None:
        self.state = TrackState.LOST


def iou_distance(tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
    """1 - IoU, as a cost matrix"""
    # the track's own estimate, not the box it was last seen at, so a track that
    # was predicted through a gap is compared where it should be now
    estimates = np.array([track.box for track in tracks], dtype=float).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return 1.0 - iou_matrix(estimates, boxes)


def associate(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment; returns matches, unmatched track rows, unmatched detection columns"""
    tracks, detections = cost.shape
    if not tracks or not detections:
        return [], list(range(tracks)), list(range(detections))

    matches = []
    for track, detection in zip(*linear_sum_assignment(cost)):
        # the assignment is global, so it pairs everything it can. the threshold is
        # what decides whether a pairing is close enough to be the same object
        if cost[track, detection] <= threshold:
            matches.append((int(track), int(detection)))

    matched_tracks = {track for track, _ in matches}
    matched_detections = {detection for _, detection in matches}
    return (
        matches,
        [track for track in range(tracks) if track not in matched_tracks],
        [detection for detection in range(detections) if detection not in matched_detections],
    )


class ByteTrack:
    """association is class agnostic; each track votes its own class. see docs/decisions/0007"""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 0.0,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area

        self.kalman = KalmanFilter()
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: Detections) -> Detections:
        """one frame. high scoring boxes associate first, then low scoring ones rescue lost tracks"""
        for track in self.tracks:
            track.predict()

        boxes, scores, class_ids = self._usable(detections)
        high = np.flatnonzero(scores >= self.high_thresh)
        low = np.flatnonzero(scores < self.high_thresh)

        # the first pass sees every track, lost ones included, so an instrument that
        # comes back is recognised instead of being opened again under a new id
        pool = list(self.tracks)
        matched, missed, unclaimed = associate(iou_distance(pool, boxes[high]), self.match_thresh)
        for track, detection in matched:
            index = high[detection]
            pool[track].update(boxes[index], scores[index], class_ids[index])

        # the second pass sees only the tracks the first could not explain. an occluded
        # instrument usually still has a weak box on it, and that is enough to continue
        rescue = [pool[track] for track in missed]
        matched, missed, _ = associate(iou_distance(rescue, boxes[low]), RESCUE_MATCH_THRESH)
        for track, detection in matched:
            index = low[detection]
            rescue[track].update(boxes[index], scores[index], class_ids[index])

        for track in missed:
            rescue[track].mark_lost()

        # a high scoring box that no track claimed is an instrument we have not seen
        for detection in unclaimed:
            index = high[detection]
            self.tracks.append(
                Track(self.next_id, boxes[index], scores[index], class_ids[index], self.kalman)
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if self._alive(track)]
        return self._detections()

    def reset(self) -> None:
        """drop every track and reset the id counter; call between sequences or ids leak across them"""
        self.tracks = []
        self.next_id = 1

    def _usable(self, detections: Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """boxes worth tracking. a zero area box would divide by zero on the way to a measurement"""
        boxes = np.asarray(detections.boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(detections.scores, dtype=float).reshape(-1)
        class_ids = np.asarray(detections.class_ids, dtype=np.int64).reshape(-1)

        widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (
            (widths > 0)
            & (heights > 0)
            & (widths * heights >= self.min_box_area)
            & (scores >= self.low_thresh)
        )
        return boxes[keep], scores[keep], class_ids[keep]

    def _alive(self, track: Track) -> bool:
        # a track that was never confirmed has to be seen every frame, or it was noise
        if track.state is TrackState.TENTATIVE:
            return track.time_since_update == 0
        return track.time_since_update <= self.track_buffer

    def _detections(self) -> Detections:
        """confirmed tracks seen this frame. a predicted box is an estimate, not an observation"""
        visible = [
            track
            for track in self.tracks
            if track.state is TrackState.CONFIRMED and track.time_since_update == 0
        ]
        return Detections(
            np.array([track.box for track in visible], dtype=np.float32).reshape(-1, 4),
            np.array([track.score for track in visible], dtype=np.float32),
            np.array([track.class_id for track in visible], dtype=np.int64),
            np.array([track.track_id for track in visible], dtype=np.int64),
        )
