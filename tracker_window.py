"""
OBS Face Tracker – Management Window.

This module provides the ``TrackerWindow`` Qt dialog that lets the user:
  * Start / stop face-tracking.
  * Select the video source (camera index or stream URL).
  * Tune the deadzone and tracking speed.
  * Configure the PTZ control method (URL / ONVIF / obs-ptz plugin).
  * View a live annotated preview of the video feed.

The heavy-lifting (face detection + PTZ commands) runs inside a ``QThread``
subclass (``TrackerThread``) so the GUI stays responsive.
"""

import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from face_detector import FaceDetector
from ptz_controller import PTZController


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class TrackerThread(QThread):
    """
    Runs face detection and PTZ control in a dedicated QThread.

    Signals
    -------
    face_detected(x_norm, y_norm, w_norm, h_norm)
        Emitted when a face is found; coordinates are normalised to [0, 1].
    no_face()
        Emitted when no face is detected in the current frame.
    frame_ready(ndarray)
        Emitted with an annotated BGR frame for the live preview.
    status_update(str)
        Informational message for the status bar.
    error_occurred(str)
        Emitted on unrecoverable errors; tracking stops automatically.
    """

    face_detected = pyqtSignal(float, float, float, float)
    no_face = pyqtSignal()
    frame_ready = pyqtSignal(object)  # numpy ndarray
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_source = 0
        self.detector = FaceDetector()
        self.ptz = PTZController()
        self.deadzone: float = 0.1
        self.speed: float = 0.5
        self._running = False
        self._last_direction: str | None = None

    # -- QThread entry point -----------------------------------------------

    def run(self):
        self._running = True
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                self.error_occurred.emit(
                    f"Cannot open video source: {self.camera_source}"
                )
                return

            self.status_update.emit("Tracking active")

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                self._process_frame(frame)
                time.sleep(0.033)  # target ~30 Hz

        except Exception as exc:
            self.error_occurred.emit(f"Tracking error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            try:
                self.ptz.stop()
            except Exception:
                pass
            self._last_direction = None

    def stop_tracking(self):
        """Signal the thread to stop (non-blocking)."""
        self._running = False

    # -- Internal helpers --------------------------------------------------

    def _process_frame(self, frame: np.ndarray):
        frame_h, frame_w = frame.shape[:2]

        offset, face_rect = self.detector.get_primary_face_offset(frame)

        # Annotate and emit preview (throttled via Qt's queued connection).
        annotated = self.detector.annotate_frame(frame, face_rect, self.deadzone)
        self.frame_ready.emit(annotated)

        if face_rect is not None:
            x, y, w, h = face_rect
            self.face_detected.emit(
                x / frame_w, y / frame_h, w / frame_w, h / frame_h
            )
            self._control_ptz(offset)
        else:
            self.no_face.emit()
            if self._last_direction is not None:
                self.ptz.stop()
                self._last_direction = None

    def _control_ptz(self, offset: float):
        """Send pan commands based on normalised face offset."""
        if abs(offset) <= self.deadzone:
            # Face is inside the deadzone – stop any ongoing movement.
            if self._last_direction is not None:
                self.ptz.stop()
                self._last_direction = None
        elif offset > self.deadzone:
            # Face is to the right.
            effective = offset - self.deadzone
            pan_speed = min(1.0, effective * self.speed * 2.0)
            if self._last_direction != "right":
                self.ptz.pan_right(pan_speed)
                self._last_direction = "right"
        else:
            # Face is to the left.
            effective = abs(offset) - self.deadzone
            pan_speed = min(1.0, effective * self.speed * 2.0)
            if self._last_direction != "left":
                self.ptz.pan_left(pan_speed)
                self._last_direction = "left"


# ---------------------------------------------------------------------------
# Management window
# ---------------------------------------------------------------------------


class TrackerWindow(QDialog):
    """
    Standalone management window for OBS Face Tracker.

    Can be used standalone (e.g. launched from the OBS Scripts dialog) or
    as an embedded widget.
    """

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._thread: TrackerThread | None = None

        self.setWindowTitle("OBS Face Tracker")
        self.setMinimumSize(640, 580)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ---- Status bar ----
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        sf_layout = QHBoxLayout(status_frame)
        sf_layout.setContentsMargins(6, 4, 6, 4)

        self._status_lbl = QLabel("Status: Inactive")
        self._status_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        sf_layout.addWidget(self._status_lbl)

        self._face_lbl = QLabel("No face detected")
        self._face_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sf_layout.addWidget(self._face_lbl)

        root.addWidget(status_frame)

        # ---- Live preview ----
        self._preview_lbl = QLabel(
            "Live preview will appear here once tracking starts."
        )
        self._preview_lbl.setMinimumHeight(220)
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setStyleSheet("background-color: #1a1a1a; color: #888;")
        root.addWidget(self._preview_lbl)

        # ---- Tab widget ----
        tabs = QTabWidget()
        tabs.addTab(self._build_control_tab(), "Control")
        tabs.addTab(self._build_tracking_tab(), "Tracking")
        tabs.addTab(self._build_ptz_tab(), "PTZ Settings")
        root.addWidget(tabs)

        # ---- Bottom buttons ----
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self._apply_all_settings)
        btn_row.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    # -- Control tab -------------------------------------------------------

    def _build_control_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Activate / deactivate button
        self._track_btn = QPushButton("▶  Start Tracking")
        self._track_btn.setMinimumHeight(54)
        self._track_btn.setFont(QFont("Arial", 12))
        self._set_track_btn_active(False)
        self._track_btn.clicked.connect(self.toggle_tracking)
        layout.addWidget(self._track_btn)

        # Video source group
        src_group = QGroupBox("Video Source")
        src_form = QFormLayout(src_group)

        self._src_type_combo = QComboBox()
        self._src_type_combo.addItems(["Camera Device Index", "Video URL / RTSP Stream"])
        self._src_type_combo.currentIndexChanged.connect(self._on_src_type_changed)
        src_form.addRow("Source Type:", self._src_type_combo)

        self._cam_index_spin = QSpinBox()
        self._cam_index_spin.setRange(0, 20)
        self._cam_index_spin.setValue(0)
        self._cam_index_spin.setToolTip(
            "OpenCV camera device index (0 = first webcam, 1 = second, …)"
        )
        src_form.addRow("Camera Index:", self._cam_index_spin)

        self._cam_url_edit = QLineEdit()
        self._cam_url_edit.setPlaceholderText(
            "rtsp://192.168.1.100/stream  or  http://192.168.1.100/video"
        )
        self._cam_url_edit.setVisible(False)
        src_form.addRow("Stream URL:", self._cam_url_edit)

        layout.addWidget(src_group)
        layout.addStretch()
        return tab

    # -- Tracking settings tab ---------------------------------------------

    def _build_tracking_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Tracking Parameters")
        form = QFormLayout(group)

        self._deadzone_spin = QDoubleSpinBox()
        self._deadzone_spin.setRange(0.01, 0.5)
        self._deadzone_spin.setSingleStep(0.01)
        self._deadzone_spin.setValue(0.1)
        self._deadzone_spin.setDecimals(2)
        self._deadzone_spin.setToolTip(
            "Fraction of half-frame width in which no pan command is issued.\n"
            "0.10 means 10 % of the half-width on each side of centre.\n"
            "Increase to reduce jitter; decrease for tighter tracking."
        )
        form.addRow("Deadzone:", self._deadzone_spin)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 3.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setValue(0.5)
        self._speed_spin.setDecimals(1)
        self._speed_spin.setToolTip(
            "Pan speed multiplier.  Higher = faster camera movement."
        )
        form.addRow("Speed Multiplier:", self._speed_spin)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.05, 2.0)
        self._scale_spin.setSingleStep(0.05)
        self._scale_spin.setValue(1.3)
        self._scale_spin.setDecimals(2)
        self._scale_spin.setToolTip(
            "Haar cascade scale factor.\n"
            "Higher = faster detection but less accurate."
        )
        form.addRow("Detection Scale:", self._scale_spin)

        self._min_neighbors_spin = QSpinBox()
        self._min_neighbors_spin.setRange(1, 15)
        self._min_neighbors_spin.setValue(5)
        self._min_neighbors_spin.setToolTip(
            "Minimum Haar cascade neighbours.\n"
            "Higher = fewer false positives but may miss faces."
        )
        form.addRow("Min Neighbours:", self._min_neighbors_spin)

        layout.addWidget(group)
        layout.addStretch()
        return tab

    # -- PTZ settings tab --------------------------------------------------

    def _build_ptz_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Method selection
        method_group = QGroupBox("PTZ Control Method")
        method_layout = QVBoxLayout(method_group)

        self._method_url_radio = QRadioButton("URL-based  (HTTP GET requests)")
        self._method_onvif_radio = QRadioButton("ONVIF  (standard PTZ protocol)")
        self._method_obsptz_radio = QRadioButton("obs-ptz Plugin  (OBS source property)")
        self._method_url_radio.setChecked(True)

        for rb in (
            self._method_url_radio,
            self._method_onvif_radio,
            self._method_obsptz_radio,
        ):
            method_layout.addWidget(rb)
            rb.toggled.connect(self._on_method_changed)

        layout.addWidget(method_group)

        # -- URL settings --
        self._url_group = QGroupBox("URL Settings")
        url_form = QFormLayout(self._url_group)

        self._pan_left_url = QLineEdit()
        self._pan_left_url.setPlaceholderText(
            "http://192.168.1.100/ptz?cmd=left"
        )
        url_form.addRow("Pan Left URL:", self._pan_left_url)

        self._pan_right_url = QLineEdit()
        self._pan_right_url.setPlaceholderText(
            "http://192.168.1.100/ptz?cmd=right"
        )
        url_form.addRow("Pan Right URL:", self._pan_right_url)

        self._stop_url = QLineEdit()
        self._stop_url.setPlaceholderText("http://192.168.1.100/ptz?cmd=stop")
        url_form.addRow("Stop URL:", self._stop_url)

        self._speed_param = QLineEdit()
        self._speed_param.setPlaceholderText(
            "speed  (URL query-string key, leave blank if unused)"
        )
        url_form.addRow("Speed Param Name:", self._speed_param)

        layout.addWidget(self._url_group)

        # -- ONVIF settings --
        self._onvif_group = QGroupBox("ONVIF Settings")
        onvif_form = QFormLayout(self._onvif_group)

        self._onvif_host = QLineEdit()
        self._onvif_host.setPlaceholderText("192.168.1.100")
        onvif_form.addRow("Camera IP:", self._onvif_host)

        self._onvif_port = QSpinBox()
        self._onvif_port.setRange(1, 65535)
        self._onvif_port.setValue(80)
        onvif_form.addRow("Port:", self._onvif_port)

        self._onvif_user = QLineEdit()
        self._onvif_user.setPlaceholderText("admin")
        onvif_form.addRow("Username:", self._onvif_user)

        self._onvif_pass = QLineEdit()
        self._onvif_pass.setEchoMode(QLineEdit.Password)
        self._onvif_pass.setPlaceholderText("password")
        onvif_form.addRow("Password:", self._onvif_pass)

        self._onvif_profile = QLineEdit()
        self._onvif_profile.setPlaceholderText(
            "Profile token (auto-detected if left blank)"
        )
        onvif_form.addRow("Profile Token:", self._onvif_profile)

        test_btn = QPushButton("Test ONVIF Connection")
        test_btn.clicked.connect(self._test_onvif)
        onvif_form.addRow("", test_btn)

        self._onvif_group.setVisible(False)
        layout.addWidget(self._onvif_group)

        # -- obs-ptz settings --
        self._obsptz_group = QGroupBox("obs-ptz Plugin Settings")
        obsptz_form = QFormLayout(self._obsptz_group)

        self._obsptz_source = QLineEdit()
        self._obsptz_source.setPlaceholderText(
            "Exact OBS source name for the PTZ camera"
        )
        obsptz_form.addRow("PTZ Source Name:", self._obsptz_source)

        note = QLabel(
            "Requires the obs-ptz plugin to be installed and a PTZ camera "
            "source to be added in OBS.  The source name must match exactly."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        obsptz_form.addRow("", note)

        self._obsptz_group.setVisible(False)
        layout.addWidget(self._obsptz_group)

        layout.addStretch()

        apply_ptz_btn = QPushButton("Apply PTZ Settings")
        apply_ptz_btn.clicked.connect(self._apply_ptz_to_thread)
        layout.addWidget(apply_ptz_btn)

        return tab

    # ------------------------------------------------------------------
    # Slot helpers
    # ------------------------------------------------------------------

    def _set_track_btn_active(self, active: bool):
        if active:
            self._track_btn.setText("⏹  Stop Tracking")
            self._track_btn.setStyleSheet(
                "QPushButton { background-color: #7a2d2d; color: white;"
                " border-radius: 5px; }"
                "QPushButton:hover { background-color: #9a3a3a; }"
            )
        else:
            self._track_btn.setText("▶  Start Tracking")
            self._track_btn.setStyleSheet(
                "QPushButton { background-color: #2d7a2d; color: white;"
                " border-radius: 5px; }"
                "QPushButton:hover { background-color: #3a9a3a; }"
            )

    def _on_src_type_changed(self, index: int):
        use_url = index == 1
        self._cam_index_spin.setVisible(not use_url)
        self._cam_url_edit.setVisible(use_url)

    def _on_method_changed(self):
        self._url_group.setVisible(self._method_url_radio.isChecked())
        self._onvif_group.setVisible(self._method_onvif_radio.isChecked())
        self._obsptz_group.setVisible(self._method_obsptz_radio.isChecked())

    # ------------------------------------------------------------------
    # PTZ config helpers
    # ------------------------------------------------------------------

    def _ptz_config(self) -> tuple[str, dict]:
        """Return (method, config_dict) from the current UI state."""
        if self._method_url_radio.isChecked():
            return PTZController.METHOD_URL, {
                "pan_left_url": self._pan_left_url.text(),
                "pan_right_url": self._pan_right_url.text(),
                "stop_url": self._stop_url.text(),
                "speed_param": self._speed_param.text(),
            }
        if self._method_onvif_radio.isChecked():
            return PTZController.METHOD_ONVIF, {
                "host": self._onvif_host.text(),
                "port": self._onvif_port.value(),
                "username": self._onvif_user.text(),
                "password": self._onvif_pass.text(),
                "profile": self._onvif_profile.text(),
            }
        # obs-ptz
        return PTZController.METHOD_OBSPTZ, {
            "source_name": self._obsptz_source.text(),
        }

    def _apply_ptz_to_thread(self):
        """Push current PTZ settings to the running thread (if any)."""
        if self._thread is not None:
            method, cfg = self._ptz_config()
            self._thread.ptz.configure(method, cfg)

    def _apply_all_settings(self):
        """Apply all current settings to the running thread."""
        if self._thread is not None:
            self._thread.deadzone = self._deadzone_spin.value()
            self._thread.speed = self._speed_spin.value()
            self._thread.detector.scale_factor = self._scale_spin.value()
            self._thread.detector.min_neighbors = self._min_neighbors_spin.value()
            self._apply_ptz_to_thread()
        QMessageBox.information(self, "Settings Applied", "Settings have been applied.")

    def _test_onvif(self):
        ptz = PTZController()
        method, cfg = self._ptz_config()
        if method != PTZController.METHOD_ONVIF:
            return
        ptz.configure(method, cfg)
        try:
            token = ptz.test_connection()
            QMessageBox.information(
                self,
                "ONVIF Connection OK",
                f"Connected successfully.\nProfile token: {token}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "ONVIF Connection Failed", f"Error:\n{exc}"
            )

    def _camera_source(self):
        """Return the camera source value (int index or str URL)."""
        if self._src_type_combo.currentIndex() == 0:
            return self._cam_index_spin.value()
        url = self._cam_url_edit.text().strip()
        return url if url else 0

    # ------------------------------------------------------------------
    # Tracking control
    # ------------------------------------------------------------------

    def toggle_tracking(self):
        """Start tracking if inactive, stop it if active."""
        if self._thread is None or not self._thread.isRunning():
            self.start_tracking()
        else:
            self.stop_tracking()

    def start_tracking(self):
        """Start the face tracking thread."""
        self._thread = TrackerThread(parent=self)
        self._thread.deadzone = self._deadzone_spin.value()
        self._thread.speed = self._speed_spin.value()
        self._thread.detector.scale_factor = self._scale_spin.value()
        self._thread.detector.min_neighbors = self._min_neighbors_spin.value()
        self._thread.camera_source = self._camera_source()

        method, cfg = self._ptz_config()
        self._thread.ptz.configure(method, cfg)

        self._thread.face_detected.connect(self._on_face_detected)
        self._thread.no_face.connect(self._on_no_face)
        self._thread.frame_ready.connect(self._on_frame_ready)
        self._thread.status_update.connect(self._on_status_update)
        self._thread.error_occurred.connect(self._on_error)

        self._thread.start()

        self._set_track_btn_active(True)
        self._status_lbl.setText("Status: ACTIVE – Tracking")
        self._status_lbl.setStyleSheet("color: #2d9a2d; font-weight: bold;")

    def stop_tracking(self):
        """Stop the face tracking thread (non-blocking).

        Disconnects all signals before signalling the thread to stop so that
        no queued callbacks arrive after the UI has been reset.  The thread
        cleans itself up via Qt's ``deleteLater`` mechanism.
        """
        if self._thread is not None:
            thread = self._thread
            self._thread = None

            # Stop PTZ movement immediately before the thread winds down.
            try:
                thread.ptz.stop()
            except Exception:
                pass

            # Disconnect all signals so no callbacks reach this window
            # after we return.
            for sig in (
                thread.face_detected,
                thread.no_face,
                thread.frame_ready,
                thread.status_update,
                thread.error_occurred,
            ):
                try:
                    sig.disconnect()
                except Exception:
                    pass

            # Signal the thread loop to exit; it will call deleteLater once
            # finished so Qt can reclaim the object safely.
            thread.stop_tracking()
            thread.finished.connect(thread.deleteLater)

        self._set_track_btn_active(False)
        self._status_lbl.setText("Status: Inactive")
        self._status_lbl.setStyleSheet("")
        self._face_lbl.setText("No face detected")
        self._face_lbl.setStyleSheet("")
        self._preview_lbl.setText(
            "Live preview will appear here once tracking starts."
        )
        self._preview_lbl.setStyleSheet("background-color: #1a1a1a; color: #888;")

    # ------------------------------------------------------------------
    # Thread signal handlers
    # ------------------------------------------------------------------

    def _on_face_detected(self, x: float, y: float, w: float, h: float):
        self._face_lbl.setText(f"Face at ({x:.2f}, {y:.2f})  size {w:.2f}×{h:.2f}")
        self._face_lbl.setStyleSheet("color: #2d9a2d;")

    def _on_no_face(self):
        self._face_lbl.setText("No face detected")
        self._face_lbl.setStyleSheet("color: #9a2d2d;")

    def _on_frame_ready(self, frame: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img).scaled(
                self._preview_lbl.width(),
                self._preview_lbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._preview_lbl.setPixmap(pixmap)
        except Exception:
            pass

    def _on_status_update(self, msg: str):
        self._status_lbl.setText(f"Status: {msg}")

    def _on_error(self, msg: str):
        self.stop_tracking()
        QMessageBox.critical(self, "Tracking Error", msg)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def reject(self):
        """Override QDialog.reject() so the Escape key never hides the window.

        By default QDialog hides itself when Escape is pressed (via reject →
        hide).  In OBS, global keyboard shortcuts can forward Escape to active
        Qt windows, which would silently hide our tracker window and leave the
        background thread running.  Overriding reject() as a no-op prevents
        this; the window can only be dismissed via its own Close button.
        """
        pass  # intentional no-op

    def closeEvent(self, event):
        self.stop_tracking()
        event.accept()
