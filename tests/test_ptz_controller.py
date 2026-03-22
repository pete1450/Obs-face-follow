"""
Unit tests for ptz_controller.PTZController.

These tests use mocking to avoid real network or OBS calls.
"""

import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptz_controller import PTZController


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ptz():
    return PTZController()


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_default_method_is_url(self, ptz):
        assert ptz.method == PTZController.METHOD_URL

    def test_configure_sets_method_and_config(self, ptz):
        ptz.configure(PTZController.METHOD_ONVIF, {"host": "10.0.0.1"})
        assert ptz.method == PTZController.METHOD_ONVIF
        assert ptz.config["host"] == "10.0.0.1"

    def test_configure_resets_onvif_profile_cache(self, ptz):
        ptz._onvif_profile = "cached_token"
        ptz.configure(PTZController.METHOD_ONVIF, {})
        assert ptz._onvif_profile is None

    def test_configure_makes_a_copy_of_config(self, ptz):
        original = {"key": "value"}
        ptz.configure(PTZController.METHOD_URL, original)
        original["key"] = "changed"
        assert ptz.config["key"] == "value"  # copy not reference


# ---------------------------------------------------------------------------
# Speed clamping
# ---------------------------------------------------------------------------


class TestSpeedClamping:
    def test_pan_left_clamps_speed_above_one(self, ptz):
        ptz.configure(PTZController.METHOD_URL, {})
        with patch.object(ptz, "_url_pan") as mock_pan:
            ptz.pan_left(5.0)
            mock_pan.assert_called_once_with("left", 1.0)

    def test_pan_right_clamps_speed_below_zero(self, ptz):
        ptz.configure(PTZController.METHOD_URL, {})
        with patch.object(ptz, "_url_pan") as mock_pan:
            ptz.pan_right(-0.5)
            mock_pan.assert_called_once_with("right", 0.0)


# ---------------------------------------------------------------------------
# URL method
# ---------------------------------------------------------------------------


class TestURLMethod:
    def _url_ptz(self, **kwargs):
        p = PTZController()
        config = {
            "pan_left_url": "http://cam/left",
            "pan_right_url": "http://cam/right",
            "stop_url": "http://cam/stop",
        }
        config.update(kwargs)
        p.configure(PTZController.METHOD_URL, config)
        return p

    @patch("ptz_controller.requests.get")
    def test_pan_left_calls_get_with_left_url(self, mock_get, ptz):
        ptz.configure(
            PTZController.METHOD_URL,
            {"pan_left_url": "http://cam/left"},
        )
        ptz.pan_left(0.5)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "http://cam/left"

    @patch("ptz_controller.requests.get")
    def test_pan_right_calls_get_with_right_url(self, mock_get, ptz):
        ptz.configure(
            PTZController.METHOD_URL,
            {"pan_right_url": "http://cam/right"},
        )
        ptz.pan_right(0.3)
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "http://cam/right"

    @patch("ptz_controller.requests.get")
    def test_stop_calls_get_with_stop_url(self, mock_get, ptz):
        ptz.configure(
            PTZController.METHOD_URL,
            {"stop_url": "http://cam/stop"},
        )
        ptz.stop()
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "http://cam/stop"

    @patch("ptz_controller.requests.get")
    def test_speed_param_is_appended_when_configured(self, mock_get, ptz):
        ptz.configure(
            PTZController.METHOD_URL,
            {
                "pan_right_url": "http://cam/right",
                "speed_param": "speed",
            },
        )
        ptz.pan_right(0.5)
        _, kwargs = mock_get.call_args
        assert "params" in kwargs
        assert "speed" in kwargs["params"]
        assert kwargs["params"]["speed"] == 50  # round(0.5 * 100)

    @patch("ptz_controller.requests.get")
    def test_no_request_when_url_not_configured(self, mock_get, ptz):
        ptz.configure(PTZController.METHOD_URL, {})
        ptz.pan_left(0.5)
        mock_get.assert_not_called()

    @patch("ptz_controller.requests.get", side_effect=ConnectionError("timeout"))
    def test_request_exception_does_not_propagate(self, mock_get, ptz):
        ptz.configure(
            PTZController.METHOD_URL,
            {"pan_left_url": "http://cam/left"},
        )
        ptz.pan_left(0.5)  # should not raise


# ---------------------------------------------------------------------------
# ONVIF method – profile detection
# ---------------------------------------------------------------------------


class TestONVIFProfile:
    def test_configured_profile_is_returned_without_http_call(self, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "MainStream"},
        )
        with patch("ptz_controller.requests.post") as mock_post:
            token = ptz.test_connection()
            mock_post.assert_not_called()
        assert token == "MainStream"

    def test_cached_profile_is_reused(self, ptz):
        ptz.configure(PTZController.METHOD_ONVIF, {"host": "10.0.0.1"})
        ptz._onvif_profile = "cached"
        with patch("ptz_controller.requests.post") as mock_post:
            token = ptz.test_connection()
            mock_post.assert_not_called()
        assert token == "cached"

    @patch("ptz_controller.requests.post")
    def test_auto_detection_parses_token_from_soap_response(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80},
        )
        mock_response = MagicMock()
        mock_response.text = (
            '<Profiles token="Profile_000">...</Profiles>'
        )
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        token = ptz.test_connection()
        assert token == "Profile_000"


# ---------------------------------------------------------------------------
# ONVIF method – ContinuousMove / Stop
# ---------------------------------------------------------------------------


class TestONVIFControl:
    @patch("ptz_controller.requests.post")
    def test_pan_right_sends_positive_pan_value(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        ptz.pan_right(0.7)

        mock_post.assert_called_once()
        body = mock_post.call_args[1]["data"]
        assert "ContinuousMove" in body
        assert "0.7000" in body  # positive pan speed

    @patch("ptz_controller.requests.post")
    def test_pan_left_sends_negative_pan_value(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        ptz.pan_left(0.4)

        body = mock_post.call_args[1]["data"]
        assert "-0.4000" in body  # negative pan speed

    @patch("ptz_controller.requests.post")
    def test_stop_sends_stop_command(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        ptz.stop()

        mock_post.assert_called_once()
        body = mock_post.call_args[1]["data"]
        assert "Stop" in body

    @patch("ptz_controller.requests.post", side_effect=ConnectionError("refused"))
    def test_onvif_exception_does_not_propagate(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        ptz.pan_right(0.5)  # should not raise


# ---------------------------------------------------------------------------
# ONVIF – tilt is always zero (horizontal-only requirement)
# ---------------------------------------------------------------------------


class TestHorizontalOnlyTracking:
    @patch("ptz_controller.requests.post")
    def test_tilt_value_is_zero_when_panning_right(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        ptz.pan_right(0.5)
        body = mock_post.call_args[1]["data"]
        # PanTilt y attribute must be 0.0000
        assert 'y="0.0000"' in body

    @patch("ptz_controller.requests.post")
    def test_tilt_value_is_zero_when_panning_left(self, mock_post, ptz):
        ptz.configure(
            PTZController.METHOD_ONVIF,
            {"host": "10.0.0.1", "port": 80, "profile": "P1"},
        )
        ptz.pan_left(0.5)
        body = mock_post.call_args[1]["data"]
        assert 'y="0.0000"' in body
