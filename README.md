
````md
# Columbia Campus Navigator

A lightweight FastAPI web app for Columbia/SIPA students: it scans recent Columbia-related Gmail messages, uses an LLM to classify/extract events & deadlines, then shows a clean dashboard with one-click “open registration” and “add to Google Calendar”.

## What it does

- **Gmail scan (recent)**: pulls recent emails matching a query (default: `columbia newer_than:30d`)
- **AI extraction**: classifies each email and extracts structured fields:
  - Events / talks (title, time, location, RSVP link, summary)
  - Notices (course / professor / school / newsletter)
  - Deadlines (date + optional time)
- **Calendar-aware ranking**: checks Google Calendar free/busy and labels conflicting events
- **One-click actions**
  - Open registration link
  - Add event to Google Calendar
  - Add deadline reminder as **non-blocking** (transparent) calendar item

## Tech stack

- FastAPI + Jinja2 templates
- Google OAuth (Gmail readonly + Calendar events)
- OpenAI (LLM extraction)

---

## Quick start (local)

### 1) Create a virtual env & install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Environment variables

Create a `.env` (or export in your shell):

Required:

* `SESSION_SECRET` – random string for session middleware
* `BASE_URL` – e.g. `http://localhost:8000` locally, or your Render URL in prod
* `GOOGLE_OAUTH_CLIENT_JSON_B64` – base64 of Google OAuth client JSON
* `OPENAI_API_KEY` – your OpenAI key

Optional:

* `OPENAI_MODEL` – default `gpt-4.1-mini`
* `DASH_CACHE_TTL_SECONDS` – server cache TTL (if enabled), default 1200

Example:

```bash
export SESSION_SECRET="dev-secret-change-me"
export BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
export GOOGLE_OAUTH_CLIENT_JSON_B64="BASE64_ENCODED_JSON_HERE"
```

> **How to create GOOGLE_OAUTH_CLIENT_JSON_B64**
>
> Download your Google OAuth client JSON (Web application) and base64 encode it:
>
> ```bash
> base64 -i client_secret.json | tr -d '\n'
> ```
>
> Paste the output into `GOOGLE_OAUTH_CLIENT_JSON_B64`.

### 3) Run the server

```bash
uvicorn app:app --reload --port 8000
```

Open: `http://localhost:8000`

---

## Google OAuth setup notes

In Google Cloud Console:

* Create OAuth client: **Web application**
* Add Authorized redirect URI:

  * Local: `http://localhost:8000/auth/callback`
  * Prod: `https://<your-render-domain>/auth/callback`
* Enable APIs:

  * Gmail API
  * Google Calendar API
  * (Optional) OAuth2 API / People API depending on your implementation

Scopes used:

* Gmail readonly
* Calendar readonly + Calendar events
* Basic user info (openid/email/profile)

---

## App routes

* `/` – landing page
* `/login` – start OAuth flow
* `/auth/callback` – OAuth redirect
* `/prefs` – set include/exclude keywords
* `/dashboard` – main dashboard (ranked events + grouped notices)
* `/calendar/add` – add selected event to Google Calendar
* `/calendar/add_deadline` – add deadline reminder (transparent)

---

## Ranking logic (high level)

Events are scored using:

* conflict vs non-conflict handling
* presence of RSVP link / location / start time
* keyword matches from user preferences

Conflicting events are labeled and shown last.

---

## Troubleshooting

### “Invalid selection” when adding to calendar

Usually means the server couldn’t find the cached dashboard state (or session expired).

* Refresh `/dashboard` and try again.
* If deployed with multiple instances, use a shared store (Redis) for cached state.

### RSVP link not extracted

Some emails hide links inside HTML `<a href=...>`.
Fix by extracting `href` values from HTML parts and/or injecting them into text before stripping tags.

### Score seems too low

Make sure your UI percentage mapping matches your `score_event()` scale:

* If `score_event()` returns **0–100**, display it directly.
* If it returns a raw scale (e.g., 0–10), map it properly.

### No events on dashboard

Common causes:

* No matching emails in the last 30 days
* Time extraction failed (email didn’t contain a clear date/time)
* Everything filtered out as past/conflicting/excluded

---

## Security & privacy

This is an internal demo project:

* Gmail is read only for matching messages
* Calendar changes happen **only** when the user clicks “Add”
* OAuth credentials are stored in session for demo purposes

---

## License

Internal / class demo.

