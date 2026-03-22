"""
Face detection module using OpenCV Haar Cascades.

Detects faces in video frames and computes the horizontal offset
of the primary face relative to the frame center.
"""

import cv2
import numpy as np


class FaceDetector:
    """Detects faces in video frames using OpenCV Haar Cascades."""

    def __init__(self):
        self.scale_factor = 1.3
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self._cascade = None
        self._load_cascade()

    def _load_cascade(self):
        """Load the Haar cascade classifier for frontal face detection."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(
                f"Failed to load face cascade from: {cascade_path}"
            )

    def detect_faces(self, frame):
        """
        Detect all faces in a BGR frame.

        Args:
            frame: BGR numpy array (an OpenCV video frame).

        Returns:
            List of (x, y, w, h) tuples for each detected face, or an
            empty list when no faces are found.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(faces) == 0:
            return []

        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def get_primary_face_offset(self, frame):
        """
        Return the normalised horizontal offset of the largest detected face.

        The offset is measured from the frame centre:
          *  0.0  → face is perfectly centred
          * -1.0  → face centre is at the left edge of the frame
          * +1.0  → face centre is at the right edge of the frame

        Args:
            frame: BGR numpy array.

        Returns:
            (offset, face_rect) where *offset* is a float in [-1.0, 1.0]
            and *face_rect* is the (x, y, w, h) tuple of the primary face,
            or (None, None) when no face is found.
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None, None

        # Use the largest face (by area) as the primary face.
        primary = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = primary

        frame_w = frame.shape[1]
        if frame_w == 0:
            return None, None

        face_center_x = x + w / 2.0
        frame_center_x = frame_w / 2.0

        # Normalise so that the half-width equals 1.0.
        offset = (face_center_x - frame_center_x) / (frame_w / 2.0)
        return offset, primary

    def annotate_frame(self, frame, face_rect, deadzone=0.1):
        """
        Draw debug annotations on a copy of *frame*.

        Draws:
          * A green rectangle around the detected face (if any).
          * A blue vertical centre line.
          * Orange vertical deadzone boundary lines.

        Args:
            frame:      BGR numpy array.
            face_rect:  (x, y, w, h) tuple or None.
            deadzone:   Normalised deadzone half-width (fraction of
                        half-frame width).

        Returns:
            Annotated BGR numpy array.
        """
        annotated = frame.copy()
        frame_h, frame_w = annotated.shape[:2]

        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = x + w // 2
            cy = y + h // 2
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 2)
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 2)

        # Centre line (blue)
        centre_x = frame_w // 2
        cv2.line(annotated, (centre_x, 0), (centre_x, frame_h), (255, 0, 0), 1)

        # Deadzone boundaries (orange)
        dz_px = int(deadzone * frame_w / 2)
        cv2.line(
            annotated,
            (centre_x - dz_px, 0),
            (centre_x - dz_px, frame_h),
            (0, 140, 255),
            1,
        )
        cv2.line(
            annotated,
            (centre_x + dz_px, 0),
            (centre_x + dz_px, frame_h),
            (0, 140, 255),
            1,
        )

        return annotated
