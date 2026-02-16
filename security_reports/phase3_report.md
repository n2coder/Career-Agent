# Phase 3 Operational Security + Regression Report (IndianJobRAG)

Date: 2026-02-16 16:45:30
Phase: 3 (Operational Security + Reliability Hardening)
Scope: `main.py`, `engine.py`, runtime API behavior on `http://127.0.0.1:8001`

## Executive Summary

Phase 3 continued from Phase 1/2 and focused on operational hardening and regression cleanup:
- Request traceability + runtime observability
- Security header baseline
- Auth comparison hardening
- LLM source attribution correctness when provider fallback occurs
- Regression checks for auth, query, CORS, and hallucination guards

Result: The app remains functional and now has better runtime traceability and safer defaults for production operations.

## Changes Implemented

### 1) Request ID + Operational Middleware
- Added HTTP middleware that:
  - Creates/propagates `X-Request-ID`
  - Adds `X-Process-Time-Ms`
  - Adds baseline security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`
  - Logs method/path/status/duration/client IP for each request
- File: `main.py`

### 2) API Key Comparison Hardening
- Replaced direct string compare with `secrets.compare_digest` for `X-API-Key` checks.
- File: `main.py`

### 3) CORS Dev/Prod Control for `Origin: null`
- Replaced unconditional allow of `null` origin with env-controlled behavior:
  - `ALLOW_NULL_ORIGIN=true|false` (default true for local dev compatibility)
- File: `main.py`

### 4) LLM Source Attribution Accuracy
- Added runtime tracking of the actual provider/model used for the latest response (`last_response_source`), and used it in response sources.
- This avoids misleading source labels when provider fallback is used.
- File: `engine.py`

## Validation Results (Evidence)

### Runtime + Auth
- `GET /health` -> `200`
- `GET /status` without key -> `401`
- `GET /status` with key -> `200` and provider/source returned as configured

### Middleware/Security Headers
- `GET /health` now includes:
  - `X-Request-ID` present
  - `X-Process-Time-Ms` present
  - `X-Content-Type-Options: nosniff`

### Query Path
- `POST /query` with authenticated request returned `200` and non-error roadmap answer.
- `sources` reflected OpenAI model in active OpenAI mode.

### CORS
- `Origin: https://evil.example` -> no permissive ACAO header
- `Origin: null` -> `access-control-allow-origin: null` (with `ALLOW_NULL_ORIGIN=true`)

### Existing Guard Regression
- `scripts/hallucination_guards_smoketest.py` -> `{"ok": true}`

## Residual Risks / Next Steps

- Rate limiting is still in-memory and per-process; move to Redis/API-gateway for horizontal scale.
- Session storage is in-memory; add external session/cache strategy if running multiple app instances.
- Add CI gates for lint + unit + security checks (`pip-audit`, image scanning) on every merge.
- Consider stricter production CORS (`ALLOW_NULL_ORIGIN=false`) outside local dev.

## Phase 4 Recommendation

Phase 4 should focus on CI/CD security gatekeeping + deployment controls:
- Automated dependency and container vulnerability scans
- Production-safe config profile (strict CORS, TLS/proxy assumptions, log shipping)
- Operational alerting on elevated 401/429/500 rates and long-tail latency
