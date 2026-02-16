from io import BytesIO
import os
import time
import json
import re
import logging
import secrets
import uuid
import hashlib
import threading
import ipaddress
from pathlib import Path
from collections import defaultdict, deque

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from engine import RecruitmentEngine
import uvicorn

DISABLE_DOCS = (os.getenv("DISABLE_DOCS", "true").strip().lower() in {"1", "true", "yes"})
app = FastAPI(
    docs_url=None if DISABLE_DOCS else "/docs",
    redoc_url=None if DISABLE_DOCS else "/redoc",
    openapi_url=None if DISABLE_DOCS else "/openapi.json",
)

LOG_LEVEL = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("career_agent.api")

cors_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_env:
    ALLOW_ORIGINS = [x.strip() for x in cors_env.split(",") if x.strip()]
else:
    ALLOW_ORIGINS = [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",  # allows opening index.html via file:// (Origin: null) during local dev
    ]

# Allow file:// origin during local dev only when enabled.
ALLOW_NULL_ORIGIN = (os.getenv("ALLOW_NULL_ORIGIN", "true").strip().lower() in {"1", "true", "yes"})
if ALLOW_NULL_ORIGIN and "null" not in ALLOW_ORIGINS:
    ALLOW_ORIGINS.append("null")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Session-Id", "X-Monitor-Key"],
)

# Base engine holds immutable-ish config + KB; per-session engines hold user state.
_base_engine = RecruitmentEngine()

API_KEY = os.getenv("APP_API_KEY", "").strip()
API_KEY_REQUIRED = (os.getenv("API_KEY_REQUIRED", "false").strip().lower() in {"1", "true", "yes"})
MONITORING_API_KEY = os.getenv("MONITORING_API_KEY", "").strip()
MONITORING_KEY_REQUIRED = (os.getenv("MONITORING_KEY_REQUIRED", "true").strip().lower() in {"1", "true", "yes"})
MONITORING_MAX_QUERY_EVENTS = int(os.getenv("MONITORING_MAX_QUERY_EVENTS", "2000"))
MONITORING_MAX_RESUME_UPLOADS = int(os.getenv("MONITORING_MAX_RESUME_UPLOADS", "800"))
MONITORING_MAX_RESUME_BUILDS = int(os.getenv("MONITORING_MAX_RESUME_BUILDS", "800"))
MONITORING_MAX_CAPTURE_CHARS = int(os.getenv("MONITORING_MAX_CAPTURE_CHARS", "16000"))
MONITORING_RETENTION_SEC = int(os.getenv("MONITORING_RETENTION_SEC", "259200"))  # 72h default
MONITORING_ALLOWED_IPS_RAW = os.getenv("MONITORING_ALLOWED_IPS", "").strip()
MONITORING_ALLOWED_IPS = [x.strip() for x in MONITORING_ALLOWED_IPS_RAW.split(",") if x.strip()]
MONITORING_CAPTURE_QUERY_TEXT = (os.getenv("MONITORING_CAPTURE_QUERY_TEXT", "false").strip().lower() in {"1", "true", "yes"})
MONITORING_CAPTURE_RESUME_TEXT = (os.getenv("MONITORING_CAPTURE_RESUME_TEXT", "false").strip().lower() in {"1", "true", "yes"})
MONITORING_CAPTURE_RESUME_BUILD_TEXT = (os.getenv("MONITORING_CAPTURE_RESUME_BUILD_TEXT", "false").strip().lower() in {"1", "true", "yes"})
MONITORING_CAPTURE_CLIENT_METADATA = (os.getenv("MONITORING_CAPTURE_CLIENT_METADATA", "false").strip().lower() in {"1", "true", "yes"})
AGENT_ID = os.getenv("AGENT_ID", "career-agent").strip() or "career-agent"
AGENT_NAME = os.getenv("AGENT_NAME", "Lin.O").strip() or "Lin.O"
AGENT_ENV = os.getenv("AGENT_ENV", "production").strip() or "production"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "5242880"))  # 5MB default
MAX_QUERY_BYTES = int(os.getenv("MAX_QUERY_BYTES", "20000"))  # 20KB default
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_QUERY_PER_WINDOW = int(os.getenv("RATE_LIMIT_QUERY_PER_WINDOW", "30"))
RATE_LIMIT_UPLOAD_PER_WINDOW = int(os.getenv("RATE_LIMIT_UPLOAD_PER_WINDOW", "10"))
TRUST_X_FORWARDED_FOR = (os.getenv("TRUST_X_FORWARDED_FOR", "false").strip().lower() in {"1", "true", "yes"})

SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", "3600"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))

_rate_buckets = defaultdict(deque)
_sessions = {}  # session_key -> (RecruitmentEngine, last_seen_epoch)
_index_html = Path(__file__).resolve().with_name("index.html")
_assets_dir = Path(__file__).resolve().with_name("assets")
_assets_dir.mkdir(exist_ok=True)
_app_started_at = time.time()
_monitor_lock = threading.Lock()
_monitor_unique_visitors = set()
_monitor_visitors = {}
_monitor_query_events = deque(maxlen=MONITORING_MAX_QUERY_EVENTS)
_monitor_resume_upload_events = deque(maxlen=MONITORING_MAX_RESUME_UPLOADS)
_monitor_resume_build_events = deque(maxlen=MONITORING_MAX_RESUME_BUILDS)

app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


def _client_ip(request: Request):
    if TRUST_X_FORWARDED_FOR:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(bucket_key: str, limit: int):
    now = time.time()
    q = _rate_buckets[bucket_key]
    while q and (now - q[0]) > RATE_LIMIT_WINDOW_SEC:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def _require_api_key(request: Request):
    if not API_KEY_REQUIRED:
        return None
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "Server auth misconfiguration."})
    token = (request.headers.get("X-API-Key") or "").strip()
    if not secrets.compare_digest(token, API_KEY):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return None


def _require_monitoring_key(request: Request):
    if MONITORING_ALLOWED_IPS and not _is_monitor_ip_allowed(request):
        return JSONResponse(status_code=403, content={"error": "Monitoring source IP not allowed"})
    if not MONITORING_KEY_REQUIRED:
        return None
    required_key = MONITORING_API_KEY
    if not required_key:
        return JSONResponse(status_code=500, content={"error": "Monitoring auth misconfiguration."})
    token = (request.headers.get("X-Monitor-Key") or "").strip()
    if not secrets.compare_digest(token, required_key):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return None


_session_id_re = re.compile(r"^[A-Za-z0-9_-]{6,128}$")


def _session_key(request: Request) -> str:
    # Prefer explicit per-browser session id from frontend.
    sid = (request.headers.get("X-Session-Id") or "").strip()
    if sid and _session_id_re.fullmatch(sid):
        return f"sid:{sid}"

    # Fallback: IP + User-Agent hash (best-effort; not ideal for shared IPs).
    ip = _client_ip(request)
    ua = (request.headers.get("user-agent") or "").strip()
    return f"ipua:{ip}|{ua}"


def _cleanup_sessions(now: float):
    if not _sessions:
        return
    expired = [k for k, (_, seen) in _sessions.items() if (now - seen) > SESSION_TTL_SEC]
    for k in expired:
        _sessions.pop(k, None)

    # Hard cap to avoid unbounded memory. Drop oldest.
    if len(_sessions) > MAX_SESSIONS:
        items = sorted(_sessions.items(), key=lambda kv: kv[1][1])
        for k, _ in items[: max(0, len(_sessions) - MAX_SESSIONS)]:
            _sessions.pop(k, None)


def _engine_for_request(request: Request) -> RecruitmentEngine:
    now = time.time()
    _cleanup_sessions(now)
    key = _session_key(request)
    eng, _seen = _sessions.get(key, (None, None))
    if eng is None:
        eng = RecruitmentEngine.from_base(_base_engine)
    _sessions[key] = (eng, now)
    return eng


def _visitor_id_from_request(request: Request) -> str:
    session_key = _session_key(request)
    return hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:24]


def _truncate_value(value: str, max_len: int = MONITORING_MAX_CAPTURE_CHARS) -> str:
    text = (value or "").strip()
    return text[:max_len]


def _sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _safe_capture(value: str, enabled: bool, max_len: int = MONITORING_MAX_CAPTURE_CHARS) -> str:
    if not enabled:
        return ""
    return _truncate_value(value, max_len=max_len)


def _is_ip_in_allowlist(ip_text: str) -> bool:
    if not MONITORING_ALLOWED_IPS:
        return True
    ip_text = (ip_text or "").strip()
    if not ip_text:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    for token in MONITORING_ALLOWED_IPS:
        if token == "*":
            return True
        try:
            if "/" in token:
                if ip_obj in ipaddress.ip_network(token, strict=False):
                    return True
            elif ip_obj == ipaddress.ip_address(token):
                return True
        except ValueError:
            continue
    return False


def _is_monitor_ip_allowed(request: Request) -> bool:
    # Prefer x-forwarded-for when present to support reverse-proxy deployments.
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        src = xff.split(",")[0].strip()
        return _is_ip_in_allowlist(src)
    return _is_ip_in_allowlist(_client_ip(request))


def _cleanup_monitoring(now_ts: int):
    if MONITORING_RETENTION_SEC <= 0:
        return
    threshold = now_ts - MONITORING_RETENTION_SEC

    while _monitor_query_events and int(_monitor_query_events[0].get("ts", 0)) < threshold:
        _monitor_query_events.popleft()
    while _monitor_resume_upload_events and int(_monitor_resume_upload_events[0].get("ts", 0)) < threshold:
        _monitor_resume_upload_events.popleft()
    while _monitor_resume_build_events and int(_monitor_resume_build_events[0].get("ts", 0)) < threshold:
        _monitor_resume_build_events.popleft()

    expired_visitors = [vid for vid, row in _monitor_visitors.items() if int(row.get("last_seen_ts", 0)) < threshold]
    for vid in expired_visitors:
        _monitor_visitors.pop(vid, None)
        _monitor_unique_visitors.discard(vid)


def _record_visitor_seen(request: Request):
    now = int(time.time())
    vid = _visitor_id_from_request(request)
    ip = _client_ip(request)
    ua = (request.headers.get("user-agent") or "").strip()[:300]
    with _monitor_lock:
        _monitor_unique_visitors.add(vid)
        row = _monitor_visitors.get(vid)
        if not row:
            row = {
                "visitor_id": vid,
                "first_seen_ts": now,
                "last_seen_ts": now,
                "ip": (ip if MONITORING_CAPTURE_CLIENT_METADATA else ""),
                "user_agent": (ua if MONITORING_CAPTURE_CLIENT_METADATA else ""),
                "query_count": 0,
                "resume_upload_count": 0,
                "resume_build_count": 0,
                "last_query": "",
                "last_resume_name": "",
            }
        else:
            row["last_seen_ts"] = now
            if MONITORING_CAPTURE_CLIENT_METADATA:
                row["ip"] = ip
            if MONITORING_CAPTURE_CLIENT_METADATA and ua:
                row["user_agent"] = ua
        _monitor_visitors[vid] = row
    return vid


def _record_query_event(request: Request, query_text: str, use_profile_context: bool, resume_builder: bool):
    now = int(time.time())
    vid = _record_visitor_seen(request)
    q = _truncate_value(query_text, max_len=MONITORING_MAX_CAPTURE_CHARS)
    with _monitor_lock:
        _cleanup_monitoring(now)
        _monitor_query_events.append({
            "ts": now,
            "visitor_id": vid,
            "query": _safe_capture(q, MONITORING_CAPTURE_QUERY_TEXT),
            "query_len": len(q),
            "query_sha256": _sha256_text(q),
            "use_profile_context": bool(use_profile_context),
            "resume_builder": bool(resume_builder),
        })
        row = _monitor_visitors.get(vid) or {}
        row["query_count"] = int(row.get("query_count", 0)) + 1
        row["last_query"] = (_safe_capture(q, MONITORING_CAPTURE_QUERY_TEXT, max_len=280))
        row["last_seen_ts"] = now
        _monitor_visitors[vid] = row


def _record_resume_upload_event(request: Request, file_name: str, extracted_text: str, profile_name: str):
    now = int(time.time())
    vid = _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        _monitor_resume_upload_events.append({
            "ts": now,
            "visitor_id": vid,
            "file_name": (file_name or "")[:300],
            "profile_name": (profile_name or "")[:150],
            "resume_text": _safe_capture(extracted_text, MONITORING_CAPTURE_RESUME_TEXT),
            "resume_text_len": len(extracted_text or ""),
            "resume_text_sha256": _sha256_text(_truncate_value(extracted_text)),
        })
        row = _monitor_visitors.get(vid) or {}
        row["resume_upload_count"] = int(row.get("resume_upload_count", 0)) + 1
        row["last_resume_name"] = (profile_name or "")[:150]
        row["last_seen_ts"] = now
        _monitor_visitors[vid] = row


def _record_resume_build_event(request: Request, resume_name: str, content_markdown: str, query_text: str):
    now = int(time.time())
    vid = _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        _monitor_resume_build_events.append({
            "ts": now,
            "visitor_id": vid,
            "resume_name": (resume_name or "")[:150],
            "resume_markdown": _safe_capture(content_markdown, MONITORING_CAPTURE_RESUME_BUILD_TEXT),
            "resume_markdown_len": len(content_markdown or ""),
            "resume_markdown_sha256": _sha256_text(_truncate_value(content_markdown)),
            "trigger_query": _safe_capture(query_text, MONITORING_CAPTURE_QUERY_TEXT, max_len=5000),
        })
        row = _monitor_visitors.get(vid) or {}
        row["resume_build_count"] = int(row.get("resume_build_count", 0)) + 1
        row["last_seen_ts"] = now
        _monitor_visitors[vid] = row


def _extract_resume_text(file: UploadFile, content: bytes):
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        pages_text = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages_text.append(page_text)
        return "\n".join(pages_text).strip()

    if name.endswith(".docx"):
        doc = Document(BytesIO(content))
        parts = [(p.text or "").strip() for p in doc.paragraphs]
        return "\n".join([p for p in parts if p]).strip()

    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = (request.headers.get("X-Request-ID") or str(uuid.uuid4())).strip()[:128]
    request.state.request_id = request_id
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled_exception request_id=%s method=%s path=%s ip=%s",
            request_id,
            request.method,
            request.url.path,
            _client_ip(request),
        )
        response = JSONResponse(status_code=500, content={"error": "Request processing failed.", "request_id": request_id})
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    logger.info(
        "request request_id=%s method=%s path=%s status=%s duration_ms=%s ip=%s",
        request_id,
        request.method,
        request.url.path,
        getattr(response, "status_code", "?"),
        elapsed_ms,
        _client_ip(request),
    )
    return response


@app.get("/status")
async def get_status(request: Request):
    auth_err = _require_api_key(request)
    if auth_err:
        return auth_err
    return _base_engine.get_status_info()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/monitoring/summary")
async def monitoring_summary(request: Request):
    auth_err = _require_monitoring_key(request)
    if auth_err:
        return auth_err

    now = int(time.time())
    _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        visitors = list(_monitor_visitors.values())
        total_visitors = len(_monitor_unique_visitors)
        query_events = len(_monitor_query_events)
        resume_uploads = len(_monitor_resume_upload_events)
        resume_builds = len(_monitor_resume_build_events)

    return {
        "agent": AGENT_ID,
        "uptime_sec": int(time.time() - _app_started_at),
        "total_visitors": total_visitors,
        "total_queries_logged": query_events,
        "total_resume_uploads_logged": resume_uploads,
        "total_resume_builds_logged": resume_builds,
        "active_sessions": len(_sessions),
        "visitors": visitors[-200:],
        "monitoring_security": {
            "query_text_capture": MONITORING_CAPTURE_QUERY_TEXT,
            "resume_text_capture": MONITORING_CAPTURE_RESUME_TEXT,
            "resume_build_text_capture": MONITORING_CAPTURE_RESUME_BUILD_TEXT,
            "client_metadata_capture": MONITORING_CAPTURE_CLIENT_METADATA,
            "retention_sec": MONITORING_RETENTION_SEC,
            "ip_allowlist_enabled": bool(MONITORING_ALLOWED_IPS),
        },
        "llm_status": _base_engine.get_status_info(),
    }


@app.get("/monitoring/queries")
async def monitoring_queries(request: Request, limit: int = 200):
    auth_err = _require_monitoring_key(request)
    if auth_err:
        return auth_err
    limit = max(1, min(limit, 2000))
    now = int(time.time())
    _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        rows = list(_monitor_query_events)[-limit:]
    return {"count": len(rows), "items": rows}


@app.get("/monitoring/resumes/uploads")
async def monitoring_resume_uploads(request: Request, limit: int = 100):
    auth_err = _require_monitoring_key(request)
    if auth_err:
        return auth_err
    limit = max(1, min(limit, 1000))
    now = int(time.time())
    _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        rows = list(_monitor_resume_upload_events)[-limit:]
    return {"count": len(rows), "items": rows}


@app.get("/monitoring/resumes/built")
async def monitoring_resume_built(request: Request, limit: int = 100):
    auth_err = _require_monitoring_key(request)
    if auth_err:
        return auth_err
    limit = max(1, min(limit, 1000))
    now = int(time.time())
    _record_visitor_seen(request)
    with _monitor_lock:
        _cleanup_monitoring(now)
        rows = list(_monitor_resume_build_events)[-limit:]
    return {"count": len(rows), "items": rows}


@app.get("/monitoring/dashboard")
async def monitoring_dashboard(
    request: Request,
    visitor_limit: int = 200,
    query_limit: int = 200,
    upload_limit: int = 100,
    build_limit: int = 100,
):
    auth_err = _require_monitoring_key(request)
    if auth_err:
        return auth_err

    visitor_limit = max(1, min(visitor_limit, 2000))
    query_limit = max(1, min(query_limit, 2000))
    upload_limit = max(1, min(upload_limit, 1000))
    build_limit = max(1, min(build_limit, 1000))
    now = int(time.time())
    _record_visitor_seen(request)

    with _monitor_lock:
        _cleanup_monitoring(now)
        visitors = list(_monitor_visitors.values())[-visitor_limit:]
        queries = list(_monitor_query_events)[-query_limit:]
        uploads = list(_monitor_resume_upload_events)[-upload_limit:]
        builds = list(_monitor_resume_build_events)[-build_limit:]
        total_visitors = len(_monitor_unique_visitors)

    return {
        "schema_version": "monitoring.v1",
        "generated_at_ts": int(time.time()),
        "agent": {
            "id": AGENT_ID,
            "name": AGENT_NAME,
            "env": AGENT_ENV,
            "uptime_sec": int(time.time() - _app_started_at),
            "active_sessions": len(_sessions),
        },
        "totals": {
            "visitors": total_visitors,
            "queries_logged": len(_monitor_query_events),
            "resume_uploads_logged": len(_monitor_resume_upload_events),
            "resume_builds_logged": len(_monitor_resume_build_events),
        },
        "windows": {
            "visitors_returned": len(visitors),
            "queries_returned": len(queries),
            "uploads_returned": len(uploads),
            "builds_returned": len(builds),
        },
        "visitors": visitors,
        "queries": queries,
        "resume_uploads": uploads,
        "resume_builds": builds,
        "llm_status": _base_engine.get_status_info(),
    }


@app.get("/")
async def serve_ui_root():
    if _index_html.exists():
        return FileResponse(str(_index_html))
    return JSONResponse(status_code=404, content={"detail": "UI file not found"})


@app.get("/resume/status")
async def resume_status(request: Request):
    auth_err = _require_api_key(request)
    if auth_err:
        return auth_err
    engine = _engine_for_request(request)
    return engine.get_resume_status()


@app.post("/resume/clear")
async def resume_clear(request: Request):
    auth_err = _require_api_key(request)
    if auth_err:
        return auth_err
    engine = _engine_for_request(request)
    profile = engine.clear_resume_profile()
    return JSONResponse(content={"ok": True, "message": profile["message"]})


@app.post("/resume/upload")
async def resume_upload(request: Request, file: UploadFile = File(...)):
    try:
        auth_err = _require_api_key(request)
        if auth_err:
            return auth_err
        ip = _client_ip(request)
        if not _check_rate_limit(f"upload:{ip}", RATE_LIMIT_UPLOAD_PER_WINDOW):
            return JSONResponse(status_code=429, content={"ok": False, "message": "Rate limit exceeded. Try again later."})

        if not file.filename:
            return JSONResponse(status_code=400, content={"ok": False, "message": "Missing file name."})

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"ok": False, "message": "Uploaded file is too large."})
        if not content:
            return JSONResponse(status_code=400, content={"ok": False, "message": "Uploaded file is empty."})

        text = _extract_resume_text(file, content)
        engine = _engine_for_request(request)
        profile = engine.set_resume_profile(text, file.filename)

        if not profile.get("uploaded"):
            return JSONResponse(status_code=400, content={"ok": False, "message": profile.get("message", "Failed to parse resume.")})

        _record_resume_upload_event(
            request=request,
            file_name=file.filename or "",
            extracted_text=text,
            profile_name=profile.get("name", ""),
        )

        return JSONResponse(
            content={
                "ok": True,
                "name": profile["name"],
                "message": profile["message"],
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "message": str(e)})
    except Exception:
        return JSONResponse(status_code=500, content={"ok": False, "message": "Resume processing failed."})


@app.post("/query")
async def query_endpoint(request: Request):
    try:
        auth_err = _require_api_key(request)
        if auth_err:
            return auth_err
        ip = _client_ip(request)
        if not _check_rate_limit(f"query:{ip}", RATE_LIMIT_QUERY_PER_WINDOW):
            return JSONResponse(status_code=429, content={"answer": "Rate limit exceeded. Try again shortly.", "sources": []})

        raw = await request.body()
        if len(raw) > MAX_QUERY_BYTES:
            return JSONResponse(status_code=413, content={"answer": "Query payload too large.", "sources": []})
        data = json.loads(raw.decode("utf-8", errors="strict"))

        user_query = data.get("query", "")
        if user_query is None:
            user_query = ""
        if not isinstance(user_query, str):
            return JSONResponse(status_code=400, content={"answer": "Invalid query type.", "sources": []})

        use_profile_context = bool(data.get("use_profile_context", False))
        resume_builder = bool(data.get("resume_builder", False))
        _record_query_event(request, user_query, use_profile_context=use_profile_context, resume_builder=resume_builder)

        engine = _engine_for_request(request)
        result = engine.get_ai_response(
            user_query,
            use_profile_context=use_profile_context,
            resume_builder=resume_builder,
        )
        if resume_builder and isinstance(result, dict):
            rb = result.get("resume_builder") or {}
            resume_md = rb.get("content_markdown") or ""
            if resume_md:
                _record_resume_build_event(
                    request=request,
                    resume_name=rb.get("name", ""),
                    content_markdown=resume_md,
                    query_text=user_query,
                )
        return JSONResponse(content=jsonable_encoder(result))
    except ValueError:
        return JSONResponse(status_code=400, content={"answer": "Invalid JSON payload.", "sources": []})
    except Exception:
        req_id = getattr(request.state, "request_id", "-")
        logger.exception("query_endpoint_failed request_id=%s", req_id)
        return JSONResponse(
            status_code=500,
            content={"answer": "Request processing failed.", "sources": [], "request_id": req_id},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

