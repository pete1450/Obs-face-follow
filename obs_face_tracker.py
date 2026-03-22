"""
OBS Face Tracker – OBS Python Script Entry Point.

Load this file via **Tools → Scripts** in OBS Studio.

Prerequisites
-------------
Install Python dependencies before loading the script::

    pip install -r requirements.txt

The script adds a single button to the Scripts dialog.
Clicking **"Open Tracker Window"** opens the management dialog where you can:

  * Start / stop face tracking.
  * Configure the video source (camera index or RTSP/HTTP stream URL).
  * Tune the deadzone and pan speed.
  * Choose a PTZ control method (URL, ONVIF, or obs-ptz plugin).
  * View a live annotated camera preview.

If PyQt5 is not available the script logs a clear error message and directs
the user to install the dependencies.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Make sure the directory containing this script is on sys.path so that
# tracker_window, face_detector, and ptz_controller can be imported.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# obspython is supplied by OBS; it will not be present outside OBS.
try:
    import obspython as obs
except ImportError:  # allow importing this module in unit-test context
    obs = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_tracker_window = None  # TrackerWindow instance (created lazily)
_settings = None  # obs_data_t reference from OBS


# ---------------------------------------------------------------------------
# OBS script callbacks
# ---------------------------------------------------------------------------


def script_description() -> str:
    return (
        "<b>OBS Face Tracker</b><br>"
        "Detects a face in a video source and pans a PTZ camera to keep it "
        "centred horizontally.<br><br>"
        "Click <b>Open Tracker Window</b> to configure and start tracking.<br><br>"
        "Requires: <code>pip install -r requirements.txt</code>"
    )


def script_load(settings) -> None:
    """Called by OBS when the script is loaded."""
    global _settings
    _settings = settings
    if obs is not None:
        obs.timer_add(_periodic_check, 2000)


def script_unload() -> None:
    """Called by OBS when the script is unloaded."""
    global _tracker_window
    if obs is not None:
        obs.timer_remove(_periodic_check)
    if _tracker_window is not None:
        try:
            _tracker_window.stop_tracking()
            _tracker_window.close()
        except Exception:
            pass
        _tracker_window = None


def script_defaults(settings) -> None:
    """Set default property values."""
    if obs is None:
        return
    obs.obs_data_set_default_double(settings, "deadzone", 0.1)
    obs.obs_data_set_default_double(settings, "speed", 0.5)
    obs.obs_data_set_default_int(settings, "camera_index", 0)
    obs.obs_data_set_default_string(settings, "ptz_method", "url")


def script_update(settings) -> None:
    """Called by OBS when any script property changes."""
    global _settings
    _settings = settings


def script_properties():
    """Define the properties shown in the OBS Scripts dialog."""
    if obs is None:
        return None
    props = obs.obs_properties_create()
    obs.obs_properties_add_button(
        props, "open_window", "Open Tracker Window", _open_tracker_window
    )
    return props


# ---------------------------------------------------------------------------
# Button callback
# ---------------------------------------------------------------------------


def _open_tracker_window(props, prop) -> bool:
    """Open (or bring to front) the tracker management window."""
    global _tracker_window

    try:
        from PyQt5.QtWidgets import QApplication  # noqa: PLC0415

        from tracker_window import TrackerWindow  # noqa: PLC0415

        app = QApplication.instance()
        if app is None:
            # Should not happen inside OBS, but handle it gracefully.
            app = QApplication(sys.argv)  # noqa: F841

        if _tracker_window is None or not _tracker_window.isVisible():
            _tracker_window = TrackerWindow(settings=_settings)
            _tracker_window.show()
        else:
            _tracker_window.raise_()
            _tracker_window.activateWindow()

    except ImportError as exc:
        _log_error(
            f"Cannot open tracker window: {exc}\n"
            "Please install dependencies:  pip install -r requirements.txt"
        )
    except Exception as exc:
        _log_error(f"Unexpected error opening tracker window: {exc}")

    return True


# ---------------------------------------------------------------------------
# Periodic housekeeping
# ---------------------------------------------------------------------------


def _periodic_check() -> None:
    """Clean up the tracker-window reference when the window has been closed.

    ``stop_tracking()`` is already called by the window's own ``closeEvent``,
    so we only need to clear the module-level reference here.  We never call
    ``stop_tracking()`` from this function because doing so would block
    OBS's main thread while waiting for the worker thread to finish.
    """
    global _tracker_window
    if _tracker_window is None:
        return
    try:
        visible = _tracker_window.isVisible()
    except RuntimeError:
        # The underlying C++ Qt object was already destroyed.
        _tracker_window = None
        return
    if not visible:
        # The window was closed via its close button (closeEvent already
        # stopped tracking).  Clear our reference.
        _tracker_window = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_error(message: str) -> None:
    if obs is not None:
        obs.script_log(obs.LOG_ERROR, message)
    else:
        print(f"[OBS Face Tracker] ERROR: {message}", file=sys.stderr)
