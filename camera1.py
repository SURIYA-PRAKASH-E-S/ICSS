"""
camera1.py — Mobile Camera Handler for AI Crowd Surveillance System

SETUP INSTRUCTIONS FOR USER:
─────────────────────────────────────────────────────────────
Android:
  1. Install "IP Webcam" from Google Play Store (free)
  2. Open app → scroll down → tap "Start server"
  3. Note the IP shown on screen (e.g., http://192.168.1.5:8080)
  4. Enter this IP in the Streamlit sidebar under "📱 Mobile Camera"

iOS:
  1. Install "EpocCam" or "IP Camera Lite" from App Store
  2. Follow app instructions to get stream URL
  3. Enter stream URL in Streamlit sidebar

Both phone and PC must be connected to the SAME WiFi network.
─────────────────────────────────────────────────────────────
"""

import cv2
import threading
import time
import requests
import numpy as np


class MobileCameraStream:
    """
    Handles mobile camera connection via IP Webcam over WiFi.
    Uses threading for non-blocking frame capture with thread-safe frame access.
    """

    def __init__(self, ip_address: str, port: int = 8080):
        """
        ip_address: Phone IP (e.g., '192.168.1.5')
        port: IP Webcam default port is 8080
        """
        self.ip       = ip_address.strip()
        self.port     = port
        self.base_url = f"http://{self.ip}:{self.port}"

        # Try /video FIRST — most reliable for IP Webcam app
        self.stream_urls = [
            f"http://{self.ip}:{self.port}/video",
            f"http://{self.ip}:{self.port}/videofeed",
            f"http://{self.ip}:{self.port}/mjpeg/1",
        ]

        self.cap          = None
        self.frame        = None      # latest captured frame (BGR)
        self.running      = False
        self.connected    = False
        self.error_msg    = ""
        self.fps          = 0
        self._lock        = threading.Lock()   # ← thread-safe frame access
        self._thread      = None
        self._frame_count = 0
        self._start_time  = time.time()

    # ── CONNECTION ──────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        """
        Test if phone is reachable before starting stream.
        Returns (success: bool, message: str)
        """
        try:
            resp = requests.get(
                f"{self.base_url}/status.json",
                timeout=3
            )
            if resp.status_code == 200:
                return True, f"✅ Connected to {self.base_url}"
        except requests.exceptions.ConnectionError:
            return False, (
                f"❌ Cannot reach {self.base_url}\n"
                f"Check: phone & PC on same WiFi, IP Webcam app running"
            )
        except requests.exceptions.Timeout:
            return False, f"❌ Timeout — phone not responding at {self.base_url}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
        return False, "❌ Unknown connection error"

    def connect(self) -> tuple[bool, str]:
        """
        Try each stream URL until one works.
        Returns (success, message)
        """
        for url in self.stream_urls:
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer lag
                cap.set(cv2.CAP_PROP_FPS, 30)

                if not cap.isOpened():
                    cap.release()
                    continue

                # Flush stale buffer frames
                for _ in range(3):
                    cap.grab()

                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.cap       = cap
                    self.connected = True
                    self.error_msg = ""
                    return True, f"✅ Connected: {url}"

                cap.release()

            except Exception as e:
                continue

        self.error_msg = (
            "❌ Cannot read frames from any stream URL.\n"
            "Check: IP Webcam app is running, same WiFi network."
        )
        return False, self.error_msg

    # ── STREAMING ───────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Connect and start background capture thread."""
        ok, msg = self.connect()
        if not ok:
            return False, msg

        # Wait up to 2 seconds for first frame
        self.running     = True
        self._start_time = time.time()
        self._thread     = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

        # Block until first frame arrives or timeout
        timeout = time.time() + 2.0
        while time.time() < timeout:
            if self.get_frame() is not None:
                return True, msg
            time.sleep(0.05)

        return True, msg + " (waiting for first frame...)"

    def _capture_loop(self):
        """Background thread — continuously reads frames into self.frame."""
        consecutive_failures = 0

        while self.running:
            if self.cap and self.cap.isOpened():
                # Grab + retrieve (faster than read())
                grabbed = self.cap.grab()
                if grabbed:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        # Resize to standard size for faster processing
                        frame = cv2.resize(frame, (640, 480))

                        with self._lock:
                            self.frame = frame  # store latest frame

                        self._frame_count += 1
                        elapsed  = time.time() - self._start_time
                        self.fps = round(
                            self._frame_count / elapsed
                            if elapsed > 0 else 0, 1
                        )
                        consecutive_failures = 0
                        continue

                consecutive_failures += 1
                if consecutive_failures > 30:
                    # Too many failures → reconnect
                    self.connected = False
                    consecutive_failures = 0
                    time.sleep(1)
                    self.connect()
            else:
                time.sleep(0.1)

    def get_frame(self) -> np.ndarray | None:
        """
        Thread-safe frame getter.
        Returns latest BGR frame or None if not ready.
        """
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()   # return copy to avoid race condition

    def stop(self):
        self.running   = False
        self.connected = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        with self._lock:
            self.frame = None

    def get_status(self) -> dict:
        """Return connection status dict for Streamlit display."""
        return {
            'connected' : self.connected,
            'ip'        : self.ip,
            'port'      : self.port,
            'fps'       : self.fps,
            'url'       : self.base_url,
            'error'     : self.error_msg
        }


# ── STREAMLIT HELPER ────────────────────────────────────────────

def render_mobile_camera_sidebar(session_state) -> MobileCameraStream | None:
    """
    Renders mobile camera controls in Streamlit sidebar.
    Call this from app.py sidebar section.
    Returns MobileCameraStream instance or None.
    """
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.subheader("📱 Mobile Camera (IP Webcam)")

    # Setup instructions expander
    with st.sidebar.expander("📖 Setup Instructions", expanded=False):
        st.markdown("""
**Android:**
1. Install **IP Webcam** from Play Store
2. Open app → tap **Start server**
3. Note the IP address shown (e.g., `192.168.1.5`)

**iOS:**
1. Install **IP Camera Lite** from App Store
2. Start server, note the IP address

⚠️ Phone & PC must be on the **same WiFi network**
        """)

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        ip_input = st.text_input(
            "Phone IP Address",
            value=session_state.get('mobile_ip', '192.168.1.5'),
            placeholder="192.168.x.x",
            key="mobile_ip_input"
        )
    with col2:
        port_input = st.number_input(
            "Port", value=8080, min_value=1000,
            max_value=9999, key="mobile_port_input"
        )

    col_a, col_b = st.sidebar.columns(2)

    with col_a:
        if st.button("🔗 Connect", key="mobile_connect_btn"):
            if 'mobile_stream' in session_state:
                session_state.mobile_stream.stop()

            stream = MobileCameraStream(ip_input, port_input)
            ok, msg = stream.start()

            if ok:
                session_state['mobile_stream']    = stream
                session_state['mobile_ip']        = ip_input
                session_state['use_mobile_camera'] = True
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

    with col_b:
        if st.button("⛔ Disconnect", key="mobile_disconnect_btn"):
            if 'mobile_stream' in session_state:
                session_state.mobile_stream.stop()
                del session_state['mobile_stream']
            session_state['use_mobile_camera'] = False
            st.sidebar.info("📱 Mobile camera disconnected")

    # Status display
    if 'mobile_stream' in session_state:
        status = session_state.mobile_stream.get_status()
        if status['connected']:
            st.sidebar.success(
                f"📡 Live | {status['ip']}:{status['port']} "
                f"| {status['fps']} FPS"
            )
        else:
            st.sidebar.error("📡 Disconnected — check phone app")

    return session_state.get('mobile_stream', None)
