"""
PTZ Camera Controller.

Supports three control methods:
  * URL-based  – HTTP GET to configurable pan-left / pan-right / stop URLs.
  * ONVIF      – Standard SOAP-based PTZ control (ContinuousMove / Stop).
  * obs-ptz    – Interact with the obs-ptz OBS plugin via source properties.

Usage example::

    ptz = PTZController()
    ptz.configure(PTZController.METHOD_URL, {
        "pan_left_url":  "http://192.168.1.100/ptz?cmd=left",
        "pan_right_url": "http://192.168.1.100/ptz?cmd=right",
        "stop_url":      "http://192.168.1.100/ptz?cmd=stop",
        "speed_param":   "speed",
    })
    ptz.pan_right(0.6)
    time.sleep(0.5)
    ptz.stop()
"""

import base64
import datetime
import hashlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class PTZController:
    """Control a PTZ camera via URL, ONVIF, or the obs-ptz OBS plugin."""

    METHOD_URL = "url"
    METHOD_ONVIF = "onvif"
    METHOD_OBSPTZ = "obs-ptz"

    def __init__(self):
        self.method = self.METHOD_URL
        self.config: dict = {}
        self._onvif_profile: str | None = None
        self._move_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, method: str, config: dict) -> None:
        """
        Set the active PTZ control method and its configuration.

        Args:
            method: One of ``METHOD_URL``, ``METHOD_ONVIF``, or
                    ``METHOD_OBSPTZ``.
            config: Dictionary of method-specific settings (see module
                    docstring for examples).
        """
        self.method = method
        self.config = dict(config)
        self._onvif_profile = None  # reset cached profile on re-configure

    def test_connection(self) -> str:
        """
        Verify that the current PTZ configuration can reach the camera.

        For ONVIF, this fetches the media profile list and returns the
        detected (or configured) profile token.  For other methods there is
        no round-trip check and an empty string is returned immediately.

        Returns:
            A human-readable string describing the connection result.

        Raises:
            RuntimeError: If the connection attempt fails.
        """
        if self.method == self.METHOD_ONVIF:
            token = self._onvif_get_profile()
            return token
        return ""

    def pan_left(self, speed: float = 0.5) -> None:
        """Pan the camera to the left at the given speed (0.0–1.0)."""
        speed = max(0.0, min(1.0, float(speed)))
        with self._move_lock:
            if self.method == self.METHOD_URL:
                self._url_pan("left", speed)
            elif self.method == self.METHOD_ONVIF:
                self._onvif_continuous_move(-speed, 0.0)
            elif self.method == self.METHOD_OBSPTZ:
                self._obsptz_pan(-speed)

    def pan_right(self, speed: float = 0.5) -> None:
        """Pan the camera to the right at the given speed (0.0–1.0)."""
        speed = max(0.0, min(1.0, float(speed)))
        with self._move_lock:
            if self.method == self.METHOD_URL:
                self._url_pan("right", speed)
            elif self.method == self.METHOD_ONVIF:
                self._onvif_continuous_move(speed, 0.0)
            elif self.method == self.METHOD_OBSPTZ:
                self._obsptz_pan(speed)

    def stop(self) -> None:
        """Stop all PTZ movement."""
        with self._move_lock:
            if self.method == self.METHOD_URL:
                self._url_stop()
            elif self.method == self.METHOD_ONVIF:
                self._onvif_stop()
            elif self.method == self.METHOD_OBSPTZ:
                self._obsptz_stop()

    # ------------------------------------------------------------------
    # URL-based control
    # ------------------------------------------------------------------

    def _url_pan(self, direction: str, speed: float) -> None:
        if not _HAS_REQUESTS:
            logger.error("'requests' library is required for URL-based PTZ control.")
            return

        url_key = "pan_left_url" if direction == "left" else "pan_right_url"
        url = self.config.get(url_key, "").strip()
        if not url:
            logger.debug("Pan URL not configured (direction=%s).", direction)
            return

        params = {}
        speed_param = self.config.get("speed_param", "").strip()
        if speed_param:
            params[speed_param] = round(speed * 100)

        try:
            requests.get(url, params=params, timeout=2)
        except Exception as exc:
            logger.warning("PTZ URL request failed (%s): %s", url, exc)

    def _url_stop(self) -> None:
        if not _HAS_REQUESTS:
            return

        url = self.config.get("stop_url", "").strip()
        if not url:
            return

        try:
            requests.get(url, timeout=2)
        except Exception as exc:
            logger.warning("PTZ stop URL request failed (%s): %s", url, exc)

    # ------------------------------------------------------------------
    # ONVIF control
    # ------------------------------------------------------------------

    def _onvif_service_url(self, service: str = "ptz") -> str:
        host = self.config.get("host", "").strip()
        port = int(self.config.get("port", 80))
        paths = {
            "ptz": "/onvif/ptz_service",
            "media": "/onvif/media_service",
            "device": "/onvif/device_service",
        }
        return f"http://{host}:{port}{paths.get(service, '/onvif/device_service')}"

    def _onvif_auth_header(self) -> str:
        """Build a WS-UsernameToken security header (digest authentication)."""
        username = self.config.get("username", "").strip()
        password = self.config.get("password", "")

        if not username:
            return ""

        nonce_bytes = os.urandom(16)
        nonce_b64 = base64.b64encode(nonce_bytes).decode()
        created = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )

        # PasswordDigest = Base64(SHA1(nonce_raw + created_utf8 + password_utf8))
        raw = nonce_bytes + created.encode() + password.encode()
        digest = base64.b64encode(hashlib.sha1(raw).digest()).decode()  # noqa: S324

        return (
            "<soap:Header>"
            '<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-wssecurity-secext-1.0.xsd">'
            "<wsse:UsernameToken>"
            f"<wsse:Username>{username}</wsse:Username>"
            '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
            f"{digest}</wsse:Password>"
            '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-wssecurity-utility-1.0.xsd#Base64Binary">'
            f"{nonce_b64}</wsse:Nonce>"
            '<wsu:Created xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-wssecurity-utility-1.0.xsd">'
            f"{created}</wsu:Created>"
            "</wsse:UsernameToken>"
            "</wsse:Security>"
            "</soap:Header>"
        )

    def _onvif_get_profile(self) -> str:
        """Return the configured profile token, auto-detecting if necessary."""
        if self._onvif_profile:
            return self._onvif_profile

        configured = self.config.get("profile", "").strip()
        if configured:
            self._onvif_profile = configured
            return self._onvif_profile

        if not _HAS_REQUESTS:
            raise RuntimeError("'requests' library required for ONVIF auto-detection.")

        auth = self._onvif_auth_header()
        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"'
            ' xmlns:media="http://www.onvif.org/ver10/media/wsdl">'
            f"{auth}"
            "<soap:Body><media:GetProfiles/></soap:Body>"
            "</soap:Envelope>"
        )

        resp = requests.post(
            self._onvif_service_url("media"),
            data=body,
            headers={"Content-Type": "application/soap+xml"},
            timeout=5,
        )
        resp.raise_for_status()

        import re

        # Token is typically an attribute on Profiles elements.
        match = re.search(r'<[^>]*Profiles[^>]+\btoken="([^"]+)"', resp.text)
        if not match:
            match = re.search(r'\btoken="([^"]+)"', resp.text)
        if not match:
            raise RuntimeError(
                "Could not detect a profile token from the ONVIF camera response."
            )

        self._onvif_profile = match.group(1)
        return self._onvif_profile

    def _onvif_continuous_move(self, pan: float, tilt: float = 0.0) -> None:
        if not _HAS_REQUESTS:
            logger.error("'requests' library is required for ONVIF control.")
            return

        try:
            profile = self._onvif_get_profile()
            auth = self._onvif_auth_header()
            body = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"'
                ' xmlns:ptz="http://www.onvif.org/ver20/ptz/wsdl"'
                ' xmlns:tt="http://www.onvif.org/ver10/schema">'
                f"{auth}"
                "<soap:Body>"
                "<ptz:ContinuousMove>"
                f"<ptz:ProfileToken>{profile}</ptz:ProfileToken>"
                "<ptz:Velocity>"
                f'<tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>'
                "</ptz:Velocity>"
                "</ptz:ContinuousMove>"
                "</soap:Body>"
                "</soap:Envelope>"
            )
            requests.post(
                self._onvif_service_url("ptz"),
                data=body,
                headers={"Content-Type": "application/soap+xml"},
                timeout=3,
            )
        except Exception as exc:
            logger.warning("ONVIF ContinuousMove failed: %s", exc)

    def _onvif_stop(self) -> None:
        if not _HAS_REQUESTS:
            return

        try:
            profile = self._onvif_get_profile()
            auth = self._onvif_auth_header()
            body = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"'
                ' xmlns:ptz="http://www.onvif.org/ver20/ptz/wsdl">'
                f"{auth}"
                "<soap:Body>"
                "<ptz:Stop>"
                f"<ptz:ProfileToken>{profile}</ptz:ProfileToken>"
                "<ptz:PanTilt>true</ptz:PanTilt>"
                "<ptz:Zoom>false</ptz:Zoom>"
                "</ptz:Stop>"
                "</soap:Body>"
                "</soap:Envelope>"
            )
            requests.post(
                self._onvif_service_url("ptz"),
                data=body,
                headers={"Content-Type": "application/soap+xml"},
                timeout=3,
            )
        except Exception as exc:
            logger.warning("ONVIF Stop failed: %s", exc)

    # ------------------------------------------------------------------
    # obs-ptz plugin control
    # ------------------------------------------------------------------

    def _obsptz_pan(self, speed: float) -> None:
        """
        Update the ``pan_speed`` property on an obs-ptz source.

        The obs-ptz plugin (https://github.com/glikely/obs-ptz) exposes PTZ
        cameras as OBS sources.  We update the ``pan_speed`` property so that
        the plugin drives the physical camera.
        """
        try:
            import obspython as obs  # only available inside OBS

            source_name = self.config.get("source_name", "").strip()
            if not source_name:
                logger.warning("obs-ptz source name is not configured.")
                return

            source = obs.obs_get_source_by_name(source_name)
            if source is None:
                logger.warning("obs-ptz source not found: %s", source_name)
                return

            try:
                settings = obs.obs_data_create()
                obs.obs_data_set_double(settings, "pan_speed", speed)
                obs.obs_source_update(source, settings)
                obs.obs_data_release(settings)
            finally:
                obs.obs_source_release(source)

        except ImportError:
            logger.error("obspython is not available; obs-ptz control requires OBS.")
        except Exception as exc:
            logger.warning("obs-ptz pan failed: %s", exc)

    def _obsptz_stop(self) -> None:
        self._obsptz_pan(0.0)
