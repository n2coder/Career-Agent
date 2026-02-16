# Phase 2 Security + Quality Validation Report (IndianJobRAG)

Date: 2026-02-13  
Phase: 2 (Production-Grade Security + Quality Validation)  
Scope: FastAPI backend (`main.py` + `engine.py`), frontend auth wiring (`index.html`), Docker runtime assumptions, resume upload pipeline, LLM/RAG `/query`.  
Environment: Local API on `http://127.0.0.1:8001` with `API_KEY_REQUIRED=true`.

## Executive Summary

Phase 2 validated that Phase 1 remediations still hold under additional abuse/edge-case testing, and extended validation into production-style concerns:
- DoS controls (request body caps, file size caps, rate limits)
- Browser abuse constraints (CORS behavior)
- Endpoint access control consistency
- Output quality risks (latency and response-length control)

Result: core API protections behaved as intended (401/413/429 behavior, CORS constraints, docs disabled). The largest remaining product risk is **LLM output length/latency predictability**, which impacts UX and cost.

## What We Tested

### 1) Endpoint Access Control (AuthN)

- `GET /health` public health probe (expected `200`)
- `GET /status` requires `X-API-Key` when `API_KEY_REQUIRED=true` (expected `401` unauth, `200` auth)
- `GET /resume/status` requires `X-API-Key` (expected `401` unauth, `200` auth)
- `POST /query` requires `X-API-Key` (expected `401` unauth, `200` auth)
- `POST /resume/upload` requires `X-API-Key` (expected `401` unauth)
- `POST /resume/clear` requires `X-API-Key` (expected `401` unauth)

### 2) DoS / Input Boundaries

- `/query` request body max: `MAX_QUERY_BYTES` (expected `413` when exceeded)
- `/resume/upload` file max: `MAX_UPLOAD_BYTES` (expected `413` when exceeded)
- `/resume/upload` empty file (expected `400`)
- `/resume/upload` unsupported type (expected `400`)
- Rate limiting:
  - `/query` per-IP windowed limit `RATE_LIMIT_QUERY_PER_WINDOW` (expected `429` after threshold)

### 3) CORS Validation (Browser Abuse Constraints)

- Disallowed origin should not receive permissive ACAO headers.
- Local dev `file://` origin (`Origin: null`) should be allowed (ACAO `null`).

### 4) API Surface Reduction

- Docs disabled by default: `GET /docs` should be `404` when `DISABLE_DOCS=true`.

### 5) LLM Security Regression

- Prompt/policy exfil requests should be refused and not leak internal instructions.

### 6) Quality Validation (Behavioral)

- Response length: verify whether “concise” instructions produce consistently concise outputs.
- Latency: observe end-to-end `/query` response time for a typical prompt.

## Results (Evidence)

### AuthN / Access Control

- `GET /health`: `200`
- `GET /status` without key: `401`
- `GET /status` with key: `200`
- `GET /docs`: `404`

### DoS / Input Boundaries

- `/query` invalid JSON: `400`
- `/query` oversized JSON payload (> `MAX_QUERY_BYTES`): `413`
- `/resume/upload` oversized file (> `MAX_UPLOAD_BYTES`): `413`
- `/resume/upload` unsupported type (`.exe`): `400`
- `/resume/upload` empty file: `400`

### Rate Limiting

Burst test (35 rapid `/query` requests in one window using an oversized payload):
- `413` responses: 30
- `429` responses: 5

### CORS

- `Origin: https://evil.example` -> no `access-control-allow-origin` header observed.
- `Origin: null` -> `access-control-allow-origin: null` observed.

### LLM Security (Prompt Exfiltration)

Test prompt: request for “full system prompt and hidden policies word for word”  
Observed: refused (no policy/system prompt content returned).

## Findings / Risks

### HIGH (Product/Cost): Output Length Control Is Inconsistent

Even when explicitly requesting “concise” output, the model can return multi-thousand-character answers.

Impact:
- Higher latency and cost
- Worse readability on mobile
- Increased chance of truncation/formatting issues in the UI

Recommended remediation:
- Enforce server-side generation limits (provider `max_tokens` / equivalent).
- Add a hard post-generation cap (truncate safely at section boundary + “Ask to continue” only when truly necessary).
- Use a tighter “short answer” style contract for general Q&A, and only expand when user asks for detail.

### MEDIUM (UX/Perf): Latency Variability

Observed single request latency can exceed 10 seconds depending on provider/model load and token volume.

Recommended remediation:
- Prefer streaming responses (SSE/WebSocket) for better perceived performance.
- Reduce default output size as above.
- Add timeouts and retry policy appropriate for the provider.

### LOW (Text/Encoding): Non-ASCII Quote Characters

Some responses include smart quotes/apostrophes, which can look corrupted if rendered in a non-UTF8 context.

Recommended remediation:
- Normalize to ASCII punctuation for UI rendering and PDF/report generation (or ensure UTF-8 end-to-end).

## Production Checklist (Phase 2 Exit Criteria)

- All sensitive endpoints behind auth (done in current environment with `API_KEY_REQUIRED=true`).
- Input limits on upload + query (done).
- Rate limiting present (done; note: in-memory only).
- CORS allowlist, no wildcard (done).
- Docs disabled by default in production (done).
- Minimal information disclosure in errors (done).
- CI follow-ups (recommended):
  - Dependency/SBOM snapshot (`pip freeze`) + routine scanning.
  - Container image scan in CI (Trivy/Grype) and base image pinning.
  - Move rate limiting to Redis/API gateway for horizontal scaling.

## Next Phase (Phase 3) Recommendation

**Phase 3 = “Operational Security + CI/CD Gatekeeping”**
- Automated security checks in CI (linting, SAST, dependency scanning, container scanning).
- Observability: structured logs, request IDs, metrics (latency, token counts), alerting.
- Secrets hygiene: rotate keys, ensure no secrets in logs, enforce `.env` exclusion.
- External-facing deployment hardening: TLS, reverse proxy, WAF/CDN, stricter CORS for hosted frontend.

