"""
backend/test_email.py
=====================
Standalone SMTP test script — run this DIRECTLY in Windows cmd
WITHOUT needing the FastAPI server to be running.

Usage (from the backend/ folder):
    python test_email.py

What it does:
    1. Reads your .env file and shows what was loaded
    2. Tests the TCP connection to Gmail
    3. Tests EHLO, STARTTLS, LOGIN step by step
    4. If all pass, sends a real test email to your SMTP_USER address
       so you can confirm delivery end-to-end
"""
import os
import smtplib
import socket
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Load .env manually (no FastAPI needed) ────────────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"

def load_env(path: Path) -> dict:
    """Read key=value pairs from a .env file."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env

env   = load_env(_ENV_FILE)

HOST  = env.get("SMTP_HOST", "")
PORT  = int(env.get("SMTP_PORT", 587))
USER  = env.get("SMTP_USER", "")
PASS  = env.get("SMTP_PASSWORD", "")
FROM  = env.get("EMAIL_FROM", "WebMine BI")

W = 62
SEP = "═" * W

def ok(msg):  print(f"  ✓  {msg}")
def err(msg): print(f"  ✗  {msg}")
def hdr(msg): print(f"\n  {msg}\n  {'─'*W}")

# ── Step 0: Show what was read ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("  WebMine BI — SMTP Email Test")
print(SEP)
print(f"\n  .env file : {_ENV_FILE}")
print(f"  Exists    : {'YES' if _ENV_FILE.exists() else 'NO  ← CREATE backend/.env'}")

hdr("Settings loaded from .env")
print(f"  SMTP_HOST     : {HOST or '(empty) ← must be smtp.gmail.com'}")
print(f"  SMTP_PORT     : {PORT}")
print(f"  SMTP_USER     : {USER or '(empty) ← your Gmail address'}")
print(f"  SMTP_PASSWORD : {'*' * len(PASS) if PASS else '(empty) ← App Password'}")
print(f"  EMAIL_FROM    : {FROM}")

if not all([HOST, USER, PASS]):
    print(f"\n  ✗  Cannot continue — one or more settings are empty.")
    print(f"     Edit {_ENV_FILE} and add the missing values.")
    print(f"\n  Template:\n")
    print("     SMTP_HOST=smtp.gmail.com")
    print("     SMTP_PORT=587")
    print("     SMTP_USER=your-address@gmail.com")
    print("     SMTP_PASSWORD=abcdefghijklmnop   # 16-char App Password, no spaces")
    print("     EMAIL_FROM=WebMine BI")
    print(f"\n{SEP}\n")
    raise SystemExit(1)

# ── Step 1: TCP connection ─────────────────────────────────────────────────────
hdr(f"Step 1 — TCP connection to {HOST}:{PORT}")
try:
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.close()
    ok(f"TCP socket opened to {HOST}:{PORT}")
except socket.timeout:
    err(f"Connection timed out — firewall may be blocking port {PORT}")
    print(f"\n  Try on a different network (home vs university).")
    raise SystemExit(1)
except OSError as e:
    err(f"Network error: {e}")
    raise SystemExit(1)

# ── Steps 2-4: SMTP handshake ──────────────────────────────────────────────────
hdr("Steps 2-4 — SMTP EHLO → STARTTLS → LOGIN")
ctx = ssl.create_default_context()
try:
    with smtplib.SMTP(HOST, PORT, timeout=20) as smtp:

        # Step 2 — EHLO
        code, banner = smtp.ehlo()
        if code != 250:
            err(f"EHLO rejected: {code} {banner}")
            raise SystemExit(1)
        ok(f"EHLO accepted  (server said: {banner[:60].decode(errors='ignore') if isinstance(banner,bytes) else str(banner)[:60]})")

        # Step 3 — STARTTLS  (upgrade to encrypted connection)
        smtp.starttls(context=ctx)
        smtp.ehlo()                # ← required second EHLO after TLS upgrade
        ok("STARTTLS negotiated and second EHLO sent")

        # Step 4 — LOGIN
        smtp.login(USER, PASS)
        ok(f"LOGIN successful  ({USER})")

        # ── Step 5: Send a real test email ─────────────────────────────────────
        hdr("Step 5 — Sending test email")
        now     = datetime.utcnow().strftime("%d %b %Y %H:%M UTC")
        subject = f"WebMine BI — SMTP Test  ({now})"
        msg     = MIMEMultipart("alternative")
        msg["Subject"]  = subject
        msg["From"]     = f"{FROM} <{USER}>"
        msg["To"]       = USER          # send the test to yourself
        msg["Reply-To"] = USER

        plain = f"WebMine BI SMTP test — sent at {now}\nIf you see this, email is working!"
        html  = f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:20px auto;
                    background:#f9fafb;border-radius:12px;overflow:hidden">
          <div style="background:#1e3a8a;padding:24px;color:#fff">
            <div style="font-size:20px;font-weight:700">📡 WebMine BI</div>
            <div style="opacity:.8;margin-top:4px">SMTP Connection Test</div>
          </div>
          <div style="padding:24px">
            <div style="font-size:32px;text-align:center;margin:16px 0">✅</div>
            <p style="text-align:center;color:#111827;font-weight:600">
              Email is working correctly!
            </p>
            <p style="color:#6b7280;text-align:center;font-size:14px">
              Sent at <strong>{now}</strong><br>
              From: <strong>{USER}</strong>
            </p>
          </div>
        </div>"""
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html,  "html",  "utf-8"))

        smtp.send_message(msg)
        ok(f"Test email sent → check {USER} inbox (also check Spam folder)")

except smtplib.SMTPAuthenticationError as e:
    code   = e.smtp_code
    detail = e.smtp_error.decode(errors="ignore") if isinstance(e.smtp_error, bytes) else str(e.smtp_error)
    hdr("✗  Authentication failed")
    print(f"  Error code : {code}")
    print(f"  Detail     : {detail}")
    print()
    print("  Most common causes:")
    print("  1. You used your Gmail login password instead of an App Password")
    print("     → Go to myaccount.google.com → Security → App passwords")
    print("     → Generate a 16-character code for 'Mail'")
    print("     → Paste it into SMTP_PASSWORD in backend/.env  (no spaces)")
    print()
    print("  2. There are spaces in the 16-character code")
    print("     → Remove all spaces: 'abcd efgh ijkl mnop' → 'abcdefghijklmnop'")
    print()
    print("  3. 2-Factor Authentication is not enabled on your Google account")
    print("     → App Passwords only appear after 2FA is turned on")
    raise SystemExit(1)

except smtplib.SMTPException as e:
    err(f"SMTP error: {e}")
    raise SystemExit(1)

except ssl.SSLError as e:
    err(f"TLS error: {e}")
    raise SystemExit(1)

# ── Done ───────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ALL STEPS PASSED — email is configured correctly.")
print(f"  Check the inbox of {USER} for the test message.")
print(f"  (If not in inbox, check the Spam / Promotions folder.)")
print(f"{SEP}\n")
