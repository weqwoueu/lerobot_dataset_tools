from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "25_plot_piper_fk_distribution.py"
SPEC = importlib.util.spec_from_file_location("plot_piper_fk_distribution", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_zero_joint_fk_matches_official_piper_sdk_reference() -> None:
    pose = MODULE.piper_fk_batch(np.zeros((1, 6), dtype=np.float64))[0]

    np.testing.assert_allclose(pose[:3], [0.056128, 0.0, 0.213266], atol=1e-6)
    np.testing.assert_allclose(
        pose[3:],
        np.radians([0.0, 85.0, 0.0]),
        atol=1e-6,
    )


def test_batch_fk_matches_single_sample_results() -> None:
    joints = np.asarray(
        [
            [0.1, 1.0, -0.7, 0.2, -0.3, 0.4],
            [-0.2, 2.0, -1.5, -0.4, 0.5, -0.6],
        ],
        dtype=np.float64,
    )

    batch_pose = MODULE.piper_fk_batch(joints)
    single_pose = np.concatenate([MODULE.piper_fk_batch(row[None, :]) for row in joints], axis=0)

    np.testing.assert_allclose(batch_pose, single_pose, atol=1e-12)


def test_base_translation_is_applied_in_analysis_frame() -> None:
    joints = np.zeros((1, 6), dtype=np.float64)
    local_pose = MODULE.piper_fk_batch(joints)[0]
    translated_pose = MODULE.piper_fk_batch(
        joints,
        base_pose_xyz_rpy_deg=(1.0, -2.0, 0.5, 0.0, 0.0, 0.0),
    )[0]

    np.testing.assert_allclose(translated_pose[:3] - local_pose[:3], [1.0, -2.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(translated_pose[3:], local_pose[3:], atol=1e-12)


def test_default_right_base_is_71_cm_along_negative_left_y() -> None:
    assert MODULE.DEFAULT_LEFT_BASE_POSE == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert MODULE.DEFAULT_RIGHT_BASE_POSE == (0.0, -0.71, 0.0, 0.0, 0.0, 0.0)


def test_fk_rejects_non_six_joint_input() -> None:
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        MODULE.piper_fk_batch(np.zeros((2, 7), dtype=np.float64))


def test_interactive_plot_is_self_contained(tmp_path: Path) -> None:
    output_path = tmp_path / "interactive.html"
    left = MODULE.VoxelDistribution(
        centers_m=np.asarray([[0.1, -0.2, 0.3], [0.2, -0.1, 0.4]]),
        counts=np.asarray([10, 20]),
    )
    right = MODULE.VoxelDistribution(
        centers_m=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]),
        counts=np.asarray([12, 24]),
    )

    MODULE.save_interactive_3d_plot(output_path, left, right, fps=30.0, max_plot_voxels=100, total_states=44)

    html = output_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert '<script src="https://cdn.plot.ly' not in html
    assert "Left arm dwell density" in html
