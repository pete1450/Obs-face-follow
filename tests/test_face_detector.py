"""
Unit tests for face_detector.FaceDetector.

These tests do not require a real camera or an ONVIF device.
They use synthetic numpy images to verify the detection logic and the
helper methods that compute offsets and draw annotations.
"""

import sys
import os

import numpy as np
import pytest

# Allow importing from the repository root regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from face_detector import FaceDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector():
    """Return a FaceDetector with default settings."""
    return FaceDetector()


def _blank_frame(width=640, height=480, channels=3, color=(0, 0, 0)):
    """Return a solid-colour BGR frame."""
    frame = np.zeros((height, width, channels), dtype=np.uint8)
    frame[:] = color
    return frame


# ---------------------------------------------------------------------------
# Cascade loading
# ---------------------------------------------------------------------------


class TestCascadeLoading:
    def test_cascade_loads_without_error(self, detector):
        """FaceDetector should initialise without raising."""
        assert detector._cascade is not None
        assert not detector._cascade.empty()

    def test_default_parameters(self, detector):
        assert detector.scale_factor == pytest.approx(1.3)
        assert detector.min_neighbors == 5
        assert detector.min_size == (30, 30)


# ---------------------------------------------------------------------------
# detect_faces
# ---------------------------------------------------------------------------


class TestDetectFaces:
    def test_returns_empty_list_on_blank_frame(self, detector):
        frame = _blank_frame()
        faces = detector.detect_faces(frame)
        assert isinstance(faces, list)
        assert faces == []

    def test_returns_empty_list_on_none_frame(self, detector):
        # Should not raise; should return an empty list.
        result = detector.detect_faces(None)
        assert result == []

    def test_returns_empty_list_on_empty_array(self, detector):
        result = detector.detect_faces(np.array([]))
        assert result == []

    def test_face_rects_are_tuples_of_four_ints(self, detector):
        """Any returned face rect must be a (x, y, w, h) tuple of ints."""
        frame = _blank_frame()
        faces = detector.detect_faces(frame)
        for face in faces:
            assert len(face) == 4
            assert all(isinstance(v, int) for v in face)


# ---------------------------------------------------------------------------
# get_primary_face_offset
# ---------------------------------------------------------------------------


class TestGetPrimaryFaceOffset:
    def test_returns_none_none_when_no_face(self, detector):
        frame = _blank_frame()
        offset, rect = detector.get_primary_face_offset(frame)
        assert offset is None
        assert rect is None

    def test_offset_range_is_within_minus_one_to_one(self, detector):
        """
        Construct a synthetic scenario by monkey-patching detect_faces to
        return a known face position and verify the normalised offset.
        """
        frame = _blank_frame(width=640, height=480)

        # Face centred at x=480 in a 640-pixel-wide frame.
        # frame_center_x = 320, face_center_x = 480
        # offset = (480 - 320) / 320 = 0.5
        detector.detect_faces = lambda _: [(440, 100, 80, 80)]  # x=440, w=80 → cx=480

        offset, rect = detector.get_primary_face_offset(frame)
        assert rect == (440, 100, 80, 80)
        assert offset == pytest.approx(0.5)

    def test_perfectly_centred_face_returns_zero_offset(self, detector):
        frame = _blank_frame(width=640, height=480)
        # cx should equal frame_center_x (320): x=280, w=80 → cx=320
        detector.detect_faces = lambda _: [(280, 100, 80, 80)]

        offset, _ = detector.get_primary_face_offset(frame)
        assert offset == pytest.approx(0.0)

    def test_left_of_centre_gives_negative_offset(self, detector):
        frame = _blank_frame(width=640, height=480)
        # cx = 80+40 = 120; frame_center = 320; offset = (120-320)/320 ≈ -0.625
        detector.detect_faces = lambda _: [(80, 100, 80, 80)]

        offset, _ = detector.get_primary_face_offset(frame)
        assert offset < 0

    def test_right_of_centre_gives_positive_offset(self, detector):
        frame = _blank_frame(width=640, height=480)
        # cx = 480+40 = 520; offset = (520-320)/320 = 0.625
        detector.detect_faces = lambda _: [(480, 100, 80, 80)]

        offset, _ = detector.get_primary_face_offset(frame)
        assert offset > 0

    def test_primary_face_is_the_largest(self, detector):
        frame = _blank_frame(width=640, height=480)
        small = (100, 100, 30, 30)  # area 900
        large = (300, 150, 100, 100)  # area 10000
        detector.detect_faces = lambda _: [small, large]

        _, rect = detector.get_primary_face_offset(frame)
        assert rect == large


# ---------------------------------------------------------------------------
# annotate_frame
# ---------------------------------------------------------------------------


class TestAnnotateFrame:
    def test_returns_same_shape_as_input(self, detector):
        frame = _blank_frame(width=320, height=240)
        annotated = detector.annotate_frame(frame, face_rect=None, deadzone=0.1)
        assert annotated.shape == frame.shape

    def test_does_not_modify_original_frame(self, detector):
        frame = _blank_frame(width=320, height=240)
        original = frame.copy()
        detector.annotate_frame(frame, face_rect=(50, 50, 60, 60), deadzone=0.1)
        assert np.array_equal(frame, original)

    def test_centre_line_is_drawn(self, detector):
        """The centre column of the annotated frame should contain blue pixels."""
        frame = _blank_frame(width=320, height=240, color=(0, 0, 0))
        # Use a non-zero deadzone so the orange deadzone lines don't
        # coincide with the blue centre line.
        annotated = detector.annotate_frame(frame, face_rect=None, deadzone=0.1)

        centre_col = annotated[:, 160]  # BGR – column at frame_w // 2
        # The blue channel (index 0 in BGR) of the centre line should be 255.
        assert centre_col[:, 0].max() == 255  # B channel

    def test_no_exception_with_none_face_rect(self, detector):
        frame = _blank_frame()
        detector.annotate_frame(frame, face_rect=None)  # should not raise
