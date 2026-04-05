"""
alert_manager.py — Real-Time Alert System
Handles: Dashboard alerts, Sound alerts, SMS/Email notifications
"""

import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ── ALERT TYPES ─────────────────────────────────────────────────

class AlertType(Enum):
    ZONE_OVERCROWDED  = "Zone Overcrowded"
    HIGH_RISK         = "High Risk Level"
    CROWD_SURGE       = "Sudden Crowd Surge"


class AlertSeverity(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    alert_type  : AlertType
    severity    : AlertSeverity
    message     : str
    zone_id     : Optional[str]   = None
    count       : int             = 0
    density     : float           = 0.0
    timestamp   : float           = field(default_factory=time.time)
    acknowledged: bool            = False

    def formatted_time(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(
            self.timestamp
        ).strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            'type'     : self.alert_type.value,
            'severity' : self.severity.value,
            'message'  : self.message,
            'zone_id'  : self.zone_id,
            'count'    : self.count,
            'density'  : self.density,
            'time'     : self.formatted_time(),
        }


# ── SURGE DETECTOR ───────────────────────────────────────────────

class SurgeDetector:
    """
    Detects sudden rapid increase in crowd count.
    Triggers if count increases by surge_threshold% within
    surge_window_seconds seconds.
    """

    def __init__(self,
                 surge_threshold_pct: float = 50.0,
                 surge_window_seconds: int  = 10):
        self.threshold   = surge_threshold_pct   # % increase
        self.window      = surge_window_seconds
        self._history    = []                    # [(timestamp, count)]

    def update(self, current_count: int) -> tuple[bool, float]:
        """
        Returns (surge_detected: bool, pct_change: float)
        """
        now = time.time()
        self._history.append((now, current_count))

        # Keep only entries within the window
        self._history = [
            (t, c) for t, c in self._history
            if now - t <= self.window
        ]

        if len(self._history) < 2:
            return False, 0.0

        oldest_count = self._history[0][1]
        if oldest_count == 0:
            return False, 0.0

        pct_change = ((current_count - oldest_count) / oldest_count) * 100

        if pct_change >= self.threshold:
            return True, round(pct_change, 1)

        return False, round(pct_change, 1)


# ── EMAIL/SMS NOTIFIER ───────────────────────────────────────────

class AlertNotifier:
    """Sends Email and SMS (via email-to-SMS gateway) notifications."""

    # Common carrier email-to-SMS gateways
    SMS_GATEWAYS = {
        'AT&T'       : 'txt.att.net',
        'T-Mobile'   : 'tmomail.net',
        'Verizon'    : 'vtext.com',
        'Airtel'     : 'airtelap.com',
        'Jio'        : 'jioworldservice.com',
        'BSNL'       : 'bsnlmobile.com',
        'VI (Vodafone)': 'vimail.in',
    }

    def __init__(self,
                 smtp_host    : str = "smtp.gmail.com",
                 smtp_port    : int = 587,
                 sender_email : str = "",
                 sender_password: str = "",
                 recipient_email: str = "",
                 phone_number : str = "",
                 carrier      : str = ""):

        self.smtp_host   = smtp_host
        self.smtp_port   = smtp_port
        self.sender      = sender_email
        self.password    = sender_password
        self.recipient   = recipient_email
        self.phone       = phone_number.replace("-","").replace(" ","")
        self.carrier     = carrier
        self.enabled     = bool(sender_email and sender_password)

    def _send_email(self, subject: str, body: str,
                    to_email: str) -> tuple[bool, str]:
        try:
            msg = MIMEMultipart()
            msg['From']    = self.sender
            msg['To']      = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, to_email, msg.as_string())
            return True, "Sent"
        except smtplib.SMTPAuthenticationError:
            return False, "Auth failed — check email/password or App Password"
        except Exception as e:
            return False, str(e)

    def send_email_alert(self, alert: Alert) -> tuple[bool, str]:
        if not self.enabled or not self.recipient:
            return False, "Email not configured"
        subject = f"🚨 Crowd Alert [{alert.severity.value}] — {alert.alert_type.value}"
        body = (
            f"CROWD SURVEILLANCE ALERT\n"
            f"{'='*40}\n"
            f"Type     : {alert.alert_type.value}\n"
            f"Severity : {alert.severity.value}\n"
            f"Time     : {alert.formatted_time()}\n"
            f"Zone     : {alert.zone_id or 'N/A'}\n"
            f"Count    : {alert.count} people\n"
            f"Density  : {alert.density:.3f} p/m²\n"
            f"\nMessage:\n{alert.message}\n"
            f"{'='*40}\n"
            f"AI Crowd Surveillance System"
        )
        # Run in thread to avoid blocking
        threading.Thread(
            target=self._send_email,
            args=(subject, body, self.recipient),
            daemon=True
        ).start()
        return True, "Email sending in background"

    def send_sms_alert(self, alert: Alert) -> tuple[bool, str]:
        if not self.enabled or not self.phone or not self.carrier:
            return False, "SMS not configured"
        gateway = self.SMS_GATEWAYS.get(self.carrier)
        if not gateway:
            return False, f"Unknown carrier: {self.carrier}"
        sms_email = f"{self.phone}@{gateway}"
        subject   = "Crowd Alert"
        body      = (
            f"[{alert.severity.value}] {alert.alert_type.value}\n"
            f"{alert.message}\n"
            f"Time: {alert.formatted_time()}"
        )
        threading.Thread(
            target=self._send_email,
            args=(subject, body, sms_email),
            daemon=True
        ).start()
        return True, "SMS sending in background"


# ── MAIN ALERT MANAGER ───────────────────────────────────────────

class AlertManager:
    """
    Central alert manager.
    Detects conditions, creates alerts, sends notifications,
    manages alert log and cooldowns.
    """

    def __init__(self,
                 cooldown_seconds      : int   = 60,
                 surge_threshold_pct   : float = 50.0,
                 surge_window_seconds  : int   = 10,
                 high_risk_count_thresh: int   = 15,
                 zone_density_thresh   : float = 0.5,
                 enable_email          : bool  = False,
                 enable_sms            : bool  = False,
                 notifier_config       : dict  = None):

        self.cooldown          = cooldown_seconds
        self.high_risk_thresh  = high_risk_count_thresh
        self.zone_dens_thresh  = zone_density_thresh
        self.enable_email      = enable_email
        self.enable_sms        = enable_sms

        self._alert_log        = []          # all alerts ever
        self._last_alert_time  = {}          # key → timestamp
        self._surge_detector   = SurgeDetector(
            surge_threshold_pct, surge_window_seconds
        )

        # Setup notifier
        cfg = notifier_config or {}
        self._notifier = AlertNotifier(**cfg) if cfg else AlertNotifier()

    # ── COOLDOWN CHECK ───────────────────────────────────────────

    def _can_alert(self, key: str) -> bool:
        last = self._last_alert_time.get(key, 0)
        return (time.time() - last) >= self.cooldown

    def _record_alert(self, key: str, alert: Alert):
        self._last_alert_time[key] = time.time()
        self._alert_log.append(alert)
        # Keep last 100 alerts
        if len(self._alert_log) > 100:
            self._alert_log = self._alert_log[-100:]

        # Send notifications in background
        if self.enable_email:
            self._notifier.send_email_alert(alert)
        if self.enable_sms:
            self._notifier.send_sms_alert(alert)

    # ── DETECTION METHODS ────────────────────────────────────────

    def check_high_risk(self, risk_level: str,
                        count: int, density: float) -> list[Alert]:
        """Trigger if risk_level == HIGH"""
        alerts = []
        if risk_level == "HIGH" and self._can_alert("high_risk"):
            alert = Alert(
                alert_type = AlertType.HIGH_RISK,
                severity   = AlertSeverity.CRITICAL,
                message    = (
                    f"🚨 HIGH RISK detected! "
                    f"{count} people, density {density:.3f} p/m²"
                ),
                count   = count,
                density = density
            )
            self._record_alert("high_risk", alert)
            alerts.append(alert)
        return alerts

    def check_zone_overcrowding(self, zones: list) -> list[Alert]:
        """Trigger for each overcrowded zone"""
        alerts = []
        for zone in zones:
            if not zone.get('is_overcrowded'):
                continue
            key = f"zone_{zone['zone_id']}"
            if self._can_alert(key):
                alert = Alert(
                    alert_type = AlertType.ZONE_OVERCROWDED,
                    severity   = AlertSeverity.HIGH,
                    message    = (
                        f"⚠️ Zone {zone['zone_id']} overcrowded! "
                        f"{zone['people_count']} people, "
                        f"{zone['density']:.3f} p/m²"
                    ),
                    zone_id = zone['zone_id'],
                    count   = zone['people_count'],
                    density = zone['density']
                )
                self._record_alert(key, alert)
                alerts.append(alert)
        return alerts

    def check_surge(self, current_count: int) -> list[Alert]:
        """Trigger on sudden crowd increase"""
        alerts = []
        surge, pct = self._surge_detector.update(current_count)
        if surge and self._can_alert("surge"):
            alert = Alert(
                alert_type = AlertType.CROWD_SURGE,
                severity   = AlertSeverity.CRITICAL,
                message    = (
                    f"📈 CROWD SURGE! Count increased {pct:.1f}% "
                    f"rapidly → {current_count} people now"
                ),
                count = current_count
            )
            self._record_alert("surge", alert)
            alerts.append(alert)
        return alerts

    def process_all(self, risk_level: str, count: int,
                    density: float, zones: list) -> list[Alert]:
        """
        Run all checks in one call.
        Returns list of NEW alerts triggered this frame.
        """
        new_alerts = []
        new_alerts += self.check_high_risk(risk_level, count, density)
        new_alerts += self.check_zone_overcrowding(zones)
        new_alerts += self.check_surge(count)
        return new_alerts

    # ── LOG ACCESS ───────────────────────────────────────────────

    def get_recent_alerts(self, n: int = 20) -> list[Alert]:
        return list(reversed(self._alert_log[-n:]))

    def get_active_alerts(self) -> list[Alert]:
        """Alerts from the last 5 minutes"""
        cutoff = time.time() - 300
        return [a for a in self._alert_log if a.timestamp >= cutoff]

    def clear_alerts(self):
        self._alert_log.clear()
        self._last_alert_time.clear()

    def get_stats(self) -> dict:
        active = self.get_active_alerts()
        return {
            'total_alerts'   : len(self._alert_log),
            'active_alerts'  : len(active),
            'high_risk_count': sum(
                1 for a in active
                if a.alert_type == AlertType.HIGH_RISK
            ),
            'surge_count'    : sum(
                1 for a in active
                if a.alert_type == AlertType.CROWD_SURGE
            ),
            'zone_alert_count': sum(
                1 for a in active
                if a.alert_type == AlertType.ZONE_OVERCROWDED
            ),
        }
