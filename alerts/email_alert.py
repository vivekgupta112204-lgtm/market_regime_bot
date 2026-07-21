from __future__ import annotations

import smtplib
from email.message import EmailMessage
from loguru import logger
from pathlib import Path

class EmailAlert:
    """Email integration for dispatching reports and system alerts."""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_address: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address

    def send_email(self, to_addresses: list[str], subject: str, body: str, attachments: list[Path] | None = None) -> bool:
        """Sends an email with optional file attachments (e.g. charts/PDFs)."""
        if not self.smtp_server or not self.username:
            logger.warning("EmailAlert: SMTP config missing.")
            return False

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.from_address
        msg['To'] = ", ".join(to_addresses)
        msg.set_content(body)

        if attachments:
            for filepath in attachments:
                if filepath.exists():
                    try:
                        with open(filepath, 'rb') as f:
                            data = f.read()
                            msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=filepath.name)
                    except Exception as e:
                        logger.error("Failed to attach {}: {}", filepath, e)
                else:
                    logger.warning("Attachment blocked, file not found: {}", filepath)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            logger.debug("Email alert sent successfully.")
            return True
        except Exception as e:
            logger.error("Failed to send email alert: {}", e)
            return False
