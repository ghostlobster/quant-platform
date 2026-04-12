"""
alerts/channels.py — Pluggable notification channels for the alert engine.

Supported channels
------------------
  TelegramChannel  — POST to Telegram Bot API
  EmailChannel     — SMTP STARTTLS (e.g. Gmail app password)
  WebhookChannel   — POST JSON payload to an arbitrary URL

Public API
----------
  get_configured_channels() -> list
      Inspect environment variables and return ready-to-use channel instances.
      Safe to call even when no env vars are set.

  broadcast(subject, body, channels=None) -> dict
      Send to every configured channel (or a supplied list).
      Returns {'sent': int, 'failed': int, 'channels': list[str]}.
"""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Protocol

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

class AlertChannel(Protocol):
    def send(self, subject: str, body: str) -> bool:
        """Send an alert. Returns True if successful, False on failure."""
        ...


# ── Channel implementations ───────────────────────────────────────────────────

@dataclass
class TelegramChannel:
    bot_token: str  # from TELEGRAM_BOT_TOKEN
    chat_id: str    # from TELEGRAM_CHAT_ID

    def send(self, subject: str, body: str) -> bool:
        """POST to Telegram Bot API. Never raises; returns False on failure."""
        try:
            import json
            import urllib.parse
            import urllib.request

            text = f"*{subject}*\n{body}"
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Telegram alert sent: %s", subject)
                    return True
                logger.error("Telegram API returned status %d", resp.status)
                return False
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


@dataclass
class EmailChannel:
    smtp_host: str   # from EMAIL_SMTP_HOST (default smtp.gmail.com)
    smtp_port: int   # from EMAIL_SMTP_PORT (default 587)
    username: str    # from EMAIL_USERNAME
    password: str    # from EMAIL_PASSWORD (app password)
    to_address: str  # from EMAIL_TO

    def send(self, subject: str, body: str) -> bool:
        """Send via SMTP STARTTLS. Never raises; returns False on failure."""
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"] = self.username
            msg["To"] = self.to_address

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.username, self.password)
                smtp.sendmail(self.username, [self.to_address], msg.as_string())

            logger.info("Email alert sent to %s: %s", self.to_address, subject)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False


@dataclass
class WebhookChannel:
    url: str  # from WEBHOOK_URL

    def send(self, subject: str, body: str) -> bool:
        """POST JSON payload to webhook URL. Never raises; returns False on failure."""
        try:
            import json
            import urllib.request

            payload = json.dumps({
                "subject": subject,
                "body": body,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }).encode()
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status < 300:
                    logger.info("Webhook alert sent to %s: %s", self.url, subject)
                    return True
                logger.error("Webhook returned status %d", resp.status)
                return False
        except Exception as exc:
            logger.error("Webhook send failed: %s", exc)
            return False


# ── Factory ───────────────────────────────────────────────────────────────────

def get_configured_channels() -> list:
    """
    Read environment variables and return a list of configured channel instances.

    A channel is only included when all its required env vars are present and
    non-empty.  Safe to call with no env vars set — returns an empty list.
    """
    channels: list = []

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if bot_token and chat_id:
        channels.append(TelegramChannel(bot_token=bot_token, chat_id=chat_id))
        logger.debug("TelegramChannel configured.")

    email_user = os.getenv("EMAIL_USERNAME", "").strip()
    email_pass = os.getenv("EMAIL_PASSWORD", "").strip()
    email_to   = os.getenv("EMAIL_TO", "").strip()
    if email_user and email_pass and email_to:
        channels.append(EmailChannel(
            smtp_host  = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com").strip(),
            smtp_port  = int(os.getenv("EMAIL_SMTP_PORT", "587")),
            username   = email_user,
            password   = email_pass,
            to_address = email_to,
        ))
        logger.debug("EmailChannel configured.")

    webhook_url = os.getenv("WEBHOOK_URL", "").strip()
    if webhook_url:
        channels.append(WebhookChannel(url=webhook_url))
        logger.debug("WebhookChannel configured.")

    return channels


# ── Broadcast ─────────────────────────────────────────────────────────────────

def broadcast(subject: str, body: str, channels: list = None) -> dict:
    """
    Send *subject* / *body* to every channel in *channels* (or all configured
    channels when *channels* is None).

    Returns
    -------
    dict with keys:
        sent     : int   — number of channels that returned True
        failed   : int   — number of channels that returned False
        channels : list[str] — channel class names attempted
    """
    if channels is None:
        channels = get_configured_channels()

    sent = 0
    failed = 0
    attempted: list[str] = []

    for ch in channels:
        name = type(ch).__name__
        attempted.append(name)
        try:
            ok = ch.send(subject, body)
        except Exception as exc:
            # Belt-and-suspenders: channel implementations should never raise,
            # but we guard here so one broken channel can't abort the rest.
            logger.error("Unexpected error from %s.send(): %s", name, exc)
            ok = False

        if ok:
            sent += 1
        else:
            failed += 1

    return {"sent": sent, "failed": failed, "channels": attempted}
