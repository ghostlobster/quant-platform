"""Email alert adapter via SMTP."""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional


class EmailAlertAdapter:
    def __init__(self) -> None:
        self._smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self._smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self._smtp_user = os.environ.get("SMTP_USER", "")
        self._smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self._from_addr = os.environ.get("ALERT_FROM_EMAIL", self._smtp_user)
        self._to_addr = os.environ.get("ALERT_TO_EMAIL", "")

    def send(self, message: str, *, level: str = "info", channel: Optional[str] = None) -> bool:
        to_addr = channel or self._to_addr
        if not to_addr:
            return False
        msg = MIMEText(message)
        msg["Subject"] = f"[{level.upper()}] Quant Platform Alert"
        msg["From"] = self._from_addr
        msg["To"] = to_addr
        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                if self._smtp_user and self._smtp_password:
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._from_addr, [to_addr], msg.as_string())
            return True
        except Exception:
            return False
