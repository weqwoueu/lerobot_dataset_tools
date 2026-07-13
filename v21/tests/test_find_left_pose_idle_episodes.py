from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "24_find_left_pose_idle_episodes.py"
SPEC = importlib.util.spec_from_file_location("find_left_pose_idle_episodes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DEFAULT_TARGET = np.asarray(MODULE.DEFAULT_TARGET_STATE, dtype=np.float64)


def make_states(frame_count: int) -> np.ndarray:
    states = np.tile(DEFAULT_TARGET, (frame_count, 1))
    states[:, 7:] += np.linspace(0.0, 1.0, frame_count)[:, None]
    return states


def find(states: np.ndarray) -> list[object]:
    return MODULE.find_qualifying_segments(
        states=states,
        target_left=DEFAULT_TARGET[:7],
        fps=30.0,
        joint_position_tolerance=0.15,
        gripper_position_tolerance=0.01,
        stationary_tolerance=0.001,
        min_duration_seconds=2.0,
    )


def test_exactly_two_seconds_is_excluded_and_longer_is_included() -> None:
    assert find(make_states(60)) == []

    segments = find(make_states(61))

    assert len(segments) == 1
    assert segments[0].start == 0
    assert segments[0].end == 61
    assert segments[0].duration_seconds == pytest.approx(61 / 30)


def test_right_arm_motion_does_not_affect_left_arm_idle_detection() -> None:
    states = make_states(90)
    states[:, 7:] = np.arange(90, dtype=np.float64)[:, None]

    segments = find(states)

    assert len(segments) == 1
    assert segments[0].frame_count == 90


@pytest.mark.parametrize(
    ("dimension", "offset"),
    [
        (0, 0.151),
        (5, -0.151),
        (6, 0.011),
    ],
)
def test_position_outside_tolerance_is_excluded(dimension: int, offset: float) -> None:
    states = make_states(90)
    states[:, dimension] += offset

    assert find(states) == []


def test_motion_frames_split_an_otherwise_long_interval() -> None:
    states = make_states(100)
    states[50, 0] += 0.002

    assert find(states) == []


def test_stationary_left_arm_near_target_is_detected() -> None:
    states = make_states(100)
    states[:, :6] += np.asarray([0.14, -0.14, 0.10, -0.10, 0.05, -0.05])
    states[:, 6] += 0.009

    segments = find(states)

    assert len(segments) == 1
    assert segments[0].frame_count == 100
