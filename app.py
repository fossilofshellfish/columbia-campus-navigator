from __future__ import annotations

import base64
import json
import os
import re
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

from dateutil import parser as dtparse
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GRequest
from googleapiclient.discovery import build

from openai import OpenAI


# ---------------- Config ----------------
APP_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")  # set to https://<render-domain>
REDIRECT_URI = f"{BASE_URL}/auth/callback"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# OAuth client JSON comes from env var (base64-encoded), so you don't need to ship a file
OAUTH_CLIENT_JSON_B64 = os.getenv("GOOGLE_OAUTH_CLIENT_JSON_B64", "")

SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# For “Columbia-related”: Gmail search can match body/subject/snippet.
# We'll require "columbia" keyword + recent window.
GMAIL_QUERY = 'columbia newer_than:30d'

templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET)


# ---------------- Helpers: OAuth ----------------
def oauth_client_config() -> dict:
    if not OAUTH_CLIENT_JSON_B64:
        raise RuntimeError("Missing env var GOOGLE_OAUTH_CLIENT_JSON_B64")
    raw = base64.b64decode(OAUTH_CLIENT_JSON_B64).decode("utf-8")
    return json.loads(raw)

def build_flow() -> Flow:
    return Flow.from_client_config(
        oauth_client_config(),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

def creds_from_session(sess: dict) -> Optional[Credentials]:
    data = sess.get("google_creds")
    if not data:
        return None
    return Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
    )

def creds_to_session(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

def ensure_valid_creds(creds: Credentials) -> Credentials:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GRequest())
    return creds


# ---------------- Helpers: Google APIs ----------------
def gmail_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds)

def cal_service(creds: Credentials):
    return build("calendar", "v3", credentials=creds)

def get_user_email(creds: Credentials) -> str:
    # Uses OAuth2 API to fetch email
    oauth2 = build("oauth2", "v2", credentials=creds)
    info = oauth2.userinfo().get().execute()
    return info.get("email", "")

def calendar_freebusy(cal, time_min: datetime, time_max: datetime, calendar_id: str="primary") -> list[dict]:
    body = {"timeMin": time_min.isoformat(), "timeMax": time_max.isoformat(), "items": [{"id": calendar_id}]}
    resp = cal.freebusy().query(body=body).execute()
    return resp.get("calendars", {}).get(calendar_id, {}).get("busy", [])

def calendar_insert(cal, event_body: dict, calendar_id: str="primary") -> dict:
    return cal.events().insert(calendarId=calendar_id, body=event_body).execute()


# ---------------- Helpers: Gmail parsing ----------------
def _b64url_decode(data: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(data + "==")

def _extract_text(payload: dict) -> str:
    # Best-effort: gather text/plain and text/html, fallback to snippet if missing
    parts = []
    stack = [payload]
    text_plain = []
    text_html = []

    while stack:
        node = stack.pop()
        if node.get("parts"):
            stack.extend(node["parts"])
            continue
        mime = node.get("mimeType", "")
        data = node.get("body", {}).get("data")
        if not data:
            continue
        raw = _b64url_decode(data).decode("utf-8", errors="ignore")
        if mime == "text/plain":
            text_plain.append(raw)
        elif mime == "text/html":
            text_html.append(raw)

    if text_plain:
        return "\n".join(text_plain)

    if text_html:
        # very light strip
        html = "\n".join(text_html)
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        html = re.sub(r"(?is)<br\s*/?>", "\n", html)
        html = re.sub(r"(?is)</p>", "\n", html)
        html = re.sub(r"(?is)<.*?>", " ", html)
        html = re.sub(r"\s+", " ", html).strip()
        return html

    return ""

def fetch_full_email(gmail, msg_id: str) -> dict:
    msg = gmail.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    body_text = _extract_text(payload)
    return {
        "id": msg_id,
        "threadId": msg.get("threadId",""),
        "from": headers.get("From",""),
        "subject": headers.get("Subject",""),
        "date": headers.get("Date",""),
        "snippet": msg.get("snippet",""),
        "body_text": (body_text or msg.get("snippet",""))[:12000],
    }

def list_message_ids(gmail, q: str, max_results: int=30) -> list[str]:
    resp = gmail.users().messages().list(userId="me", q=q, maxResults=max_results).execute()
    return [m["id"] for m in resp.get("messages", [])]


# ---------------- LLM: classification + extraction ----------------
EMAIL_SYSTEM = """
You are an assistant for Columbia University students.
You will classify a Columbia-related email and extract structured information.

Return ONLY valid JSON (no markdown).
Schema:
{
  "category": "course_requirement|campus_event|lecture_talk|school_notice|newsletter|other",
  "confidence": 0-1,
  "title": "...",
  "start_time": "ISO8601 with timezone offset or null",
  "end_time": "ISO8601 with timezone offset or null",
  "timezone": "America/New_York",
  "location": "string or null",
  "online_or_in_person": "online|in_person|hybrid|unknown",
  "rsvp_url": "string or null",
  "summary_en": "concise English summary (<=70 words)",
  "why_recommended_en": "short reason (<=45 words)"
}

Rules:
- If this is NOT an event/lecture, set start_time/end_time/location/rsvp_url to null.
- If start_time exists but end_time is missing, set end_time = start_time + 60 minutes.
- Extract RSVP/registration link if present.
"""

def llm_json(system: str, payload: dict) -> dict:
    text = json.dumps(payload, ensure_ascii=False)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":text}],
        temperature=0.2,
    )
    out = resp.choices[0].message.content.strip()
    return json.loads(out)


# ---------------- Ranking / constraints ----------------
def parse_dt(s: Optional[str]):
    if not s:
        return None
    try:
        return dtparse.isoparse(s)
    except Exception:
        return None

def overlaps(a_start, a_end, b_start, b_end) -> bool:
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return max(a_start, b_start) < min(a_end, b_end)

def has_conflict(st, et, busy_blocks: list[dict]) -> bool:
    if not st or not et:
        return False
    for b in busy_blocks:
        bs = parse_dt(b.get("start"))
        be = parse_dt(b.get("end"))
        if overlaps(st, et, bs, be):
            return True
    return False

def is_past_or_yesterday(st: Optional[datetime]) -> bool:
    if not st:
        return False
    now_ny = datetime.now(timezone.utc).astimezone(dtparse.tz.gettz("America/New_York"))
    st_ny = st.astimezone(dtparse.tz.gettz("America/New_York"))
    # Reject yesterday or earlier
    return st_ny.date() <= (now_ny.date() - timedelta(days=1))

def score_event(ex: dict, prefs: dict, conflict: bool) -> float:
    s = 0.0
    if conflict:
        s -= 5.0
    else:
        s += 3.0
    if ex.get("rsvp_url"):
        s += 1.0
    if ex.get("location"):
        s += 0.5
    if ex.get("start_time"):
        s += 1.0
    inc = prefs.get("include_keywords", []) or []
    exc = prefs.get("exclude_keywords", []) or []
    text = (ex.get("title","") + " " + ex.get("summary_en","")).lower()
    for k in inc:
        k2 = str(k).lower().strip()
        if k2 and k2 in text:
            s += 0.7
    for k in exc:
        k2 = str(k).lower().strip()
        if k2 and k2 in text:
            s -= 1.0
    try:
        s += float(ex.get("confidence", 0.5))
    except Exception:
        pass
    return s

def build_calendar_event(ex: dict) -> dict:
    tz = ex.get("timezone") or "America/New_York"
    return {
        "summary": ex.get("title") or "(No title)",
        "location": ex.get("location") or "",
        "description": f"{ex.get('summary_en','')}\n\nRSVP: {ex.get('rsvp_url') or ''}".strip(),
        "start": {"dateTime": ex.get("start_time"), "timeZone": tz},
        "end": {"dateTime": ex.get("end_time"), "timeZone": tz},
    }


# ---------------- In-memory user store (internal demo) ----------------
# For a class demo, session storage is enough. If you want persistence, swap to SQLite.
def get_prefs(request: Request) -> dict:
    return request.session.get("prefs") or {"include_keywords": [], "exclude_keywords": []}

def set_prefs(request: Request, inc: list[str], exc: list[str]):
    request.session["prefs"] = {"include_keywords": inc, "exclude_keywords": exc}


# ---------------- Routes ----------------
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home(request: Request):
    logged_in = bool(request.session.get("google_creds"))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "logged_in": logged_in
    })

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/login")
def login(request: Request):
    flow = build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    request.session["oauth_state"] = state
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
def callback(request: Request, code: str, state: str):
    if state != request.session.get("oauth_state"):
        return HTMLResponse("OAuth state mismatch", status_code=400)
    flow = build_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    request.session["google_creds"] = creds_to_session(creds)
    return RedirectResponse("/prefs")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

@app.get("/prefs", response_class=HTMLResponse)
def prefs_page(request: Request):
    if not request.session.get("google_creds"):
        return RedirectResponse("/login")
    prefs = get_prefs(request)
    return templates.TemplateResponse("prefs.html", {"request": request, "prefs": prefs})

@app.post("/prefs", response_class=HTMLResponse)
def prefs_save(request: Request, include_keywords: str = Form(""), exclude_keywords: str = Form("")):
    inc = [x.strip() for x in include_keywords.split(",") if x.strip()]
    exc = [x.strip() for x in exclude_keywords.split(",") if x.strip()]
    set_prefs(request, inc, exc)
    return RedirectResponse("/dashboard", status_code=303)



KEYWORDS = ("columbia", "sipa", "canvas")

def is_columbia_related(email: dict) -> bool:
    """
    True if any part of the email indicates Columbia relevance:
    - keyword appears in subject/snippet/body (columbia/sipa/canvas)
    - or sender domain ends with columbia.edu (including subdomains)
    - or recipient/cc contains @columbia.edu
    """
    subject = (email.get("subject") or "")
    snippet = (email.get("snippet") or "")
    body = (email.get("body_text") or "")

    # Keyword scan (subject/snippet/body)
    blob = f"{subject}\n{snippet}\n{body}".lower()
    if any(k in blob for k in KEYWORDS):
        return True

    # Sender domain check
    frm = (email.get("from") or "").lower()
    # extracts any emails found in the From header
    addrs = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", frm, flags=re.I)
    for a in addrs:
        domain = a.split("@", 1)[1]
        if domain == "columbia.edu" or domain.endswith(".columbia.edu"):
            return True

    # Optional: To/Cc domain check if you store these fields
    to_cc = (email.get("to", "") + " " + email.get("cc", "")).lower()
    if "@columbia.edu" in to_cc:
        return True

    return False

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if not request.session.get("google_creds"):
        return RedirectResponse("/login")

    creds = creds_from_session(request.session)
    creds = ensure_valid_creds(creds)
    request.session["google_creds"] = creds_to_session(creds)

    prefs = get_prefs(request)

    gmail = gmail_service(creds)
    cal = cal_service(creds)

    # schedule window
    now = datetime.now(timezone.utc)
    time_min = now
    time_max = now + timedelta(days=30)
    busy = calendar_freebusy(cal, time_min, time_max)

    # fetch columbia-related messages
    ids = list_message_ids(gmail, GMAIL_QUERY, max_results=30)
    logging.warning(f"Fetched message ids: {len(ids)}")
    emails = [fetch_full_email(gmail, mid) for mid in ids]
    logging.warning(f"Emails fetched: {len(emails)}")

    extracted = []
    for e in emails:
        # hard guarantee: must contain "columbia" anywhere
        if not is_columbia_related(e):
            logging.warning(f"Email {e['id']} is not Columbia-related")
            continue
        try:
            ex = llm_json(EMAIL_SYSTEM, {"email": e, "preferences": prefs, "busy_blocks": busy})
            extracted.append(ex)
        except Exception:
            continue
    logging.warning(f"Extracted items: {len(extracted)}")
    # split outputs
    notices = [x for x in extracted if x.get("category") in ("course_requirement","school_notice","newsletter")]
    events = [x for x in extracted if x.get("category") in ("campus_event","lecture_talk")]
    logging.warning(f"Events: {len(events)} | Notices: {len(notices)}")   
    # filter events: not past/yesterday, not conflicting
    candidates = []
    for ex in events:
        st = parse_dt(ex.get("start_time"))
        et = parse_dt(ex.get("end_time"))
        if not st or not et:
            continue
        if is_past_or_yesterday(st):
            continue
        conflict = has_conflict(st, et, busy)
        if conflict:
            continue
        candidates.append({"extracted": ex, "score": score_event(ex, prefs, conflict)})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top5 = candidates[:5]

    # Save top5 in session for add-to-calendar
    request.session["last_top5"] = top5

    # Create a compact notice summary (English) via LLM (optional but useful)
    notice_summary = ""
    if notices:
        try:
            summary_payload = {"notices": [{"category":n["category"],"title":n["title"],"summary":n["summary_en"]} for n in notices[:20]]}
            notice_summary = llm_json(
                "Summarize the following items into a concise English bulletin with headings. Return JSON: {\"bulletin\":\"...\"}",
                summary_payload
            ).get("bulletin","")
        except Exception:
            notice_summary = ""

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "prefs": prefs,
            "top5": top5,
            "notice_bulletin": notice_summary,
            "notice_items": notices[:15],
        },
    )

@app.post("/calendar/add")
def add_to_calendar(request: Request, idx: int = Form(...)):
    if not request.session.get("google_creds"):
        return RedirectResponse("/login", status_code=303)

    top5 = request.session.get("last_top5") or []
    logging.warning(f"calendar/add idx={idx} last_top5_len={len(top5)}")

    if not top5:
        return HTMLResponse(
            "Your session does not have a generated Top 5 list. Please go back to Dashboard and generate again.",
            status_code=400,
        )

    if idx < 1 or idx > len(top5):
        return HTMLResponse("Invalid selection", status_code=400)

    creds = ensure_valid_creds(creds_from_session(request.session))
    request.session["google_creds"] = creds_to_session(creds)
    cal = cal_service(creds)

    ex = top5[idx - 1]["extracted"]
    body = build_calendar_event(ex)
    _created = calendar_insert(cal, body)

    return RedirectResponse("/dashboard", status_code=303)