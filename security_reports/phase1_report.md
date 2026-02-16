# Phase 1 Security Assessment Report (IndianJobRAG)

Date: 2026-02-13  
Scope: FastAPI backend (`main.py` + `engine.py`), Docker deployment, resume upload, LLM/RAG query endpoint.  
Environment: Local Docker container, API exposed on `http://127.0.0.1:8001`.

## Executive Summary

Phase 1 focused on high-signal security smoke tests and targeted manual abuse cases against the live API. We found multiple high/critical issues typical of early-stage AI apps (prompt exfiltration, wildcard CORS, verbose error leakage, unbounded upload). These were remediated and re-tested.

## Findings (Before Fixes)

### CRITICAL: Prompt/Policy Exfiltration via `/query`
- Endpoint: `POST /query`
- Test: user prompt asking for “full hidden system prompt / policy text”
- Observed behavior: model returned internal-style system/policy text.
- Impact: policy leakage, prompt injection success, potential exposure of sensitive internal context and future secrets.

### HIGH: Wildcard CORS
- Evidence: `access-control-allow-origin: *`
- Impact: any origin could call API from a browser context, enabling cross-site abuse.

### HIGH: No Authentication on Sensitive Endpoints
- Endpoints: `/query`, `/resume/upload`, `/resume/clear`, `/status`
- Impact: cost burn (LLM), abuse of file upload, untrusted usage.

### HIGH: No Upload Size Limit
- Endpoint: `POST /resume/upload`
- Test: large file upload accepted.
- Impact: DoS/memory pressure and storage abuse.

### MEDIUM: Verbose Exception Messages Returned
- Endpoint: `POST /query`
- Tests: malformed JSON / wrong content type
- Observed behavior: HTTP 500 with raw exception string in response.
- Impact: information disclosure, easier attacker tuning.

### MEDIUM: Unsupported Upload Type Returned 500
- Endpoint: `POST /resume/upload`
- Test: upload `.exe`
- Observed behavior: HTTP 500
- Impact: noisy errors, inconsistent input handling.

### MEDIUM: `/docs` and `/openapi.json` Publicly Exposed
- Endpoints: `GET /docs`, `GET /openapi.json`
- Impact: attack surface discovery.

## Remediations Implemented

### Prompt Exfiltration Block
- Implemented detection for prompt/policy exfiltration attempts.
- Added response guard to suppress “prompt leak” style outputs if the model tries to comply.
- Files: `engine.py`

### CORS Hardening
- CORS now defaults to known origins, and can be set via `CORS_ORIGINS` env.
- Added support for `Origin: null` during local dev when opening `index.html` via `file://`.
- Files: `main.py`

### Optional API Key Authentication
- Added `X-API-Key` gate controlled by env var `API_KEY_REQUIRED`.
- Added frontend support for `X-API-Key` via localStorage, plus an in-app API key panel.
- Files: `main.py`, `index.html`

### Rate Limiting (In-Memory)
- Added basic per-IP windowed rate limits for `/query` and `/resume/upload`.
- Files: `main.py`

### Upload Controls
- Added upload max size (default 5MB) with HTTP 413.
- Unsupported formats now return HTTP 400 (not 500).
- Files: `main.py`

### Safer Error Handling
- Invalid JSON returns HTTP 400 with generic message.
- Internal errors return generic 500 responses (no raw exception strings).
- Files: `main.py`

### Disable API Docs By Default
- `/docs` and `/openapi.json` disabled by default (404) unless enabled explicitly.
- Files: `main.py`

### Docker Secret Hygiene (Post Phase 1 Hardening)
- Prevent `.env` from being copied into images by adding it to `.dockerignore`.
- Run container as non-root.
- Files: `.dockerignore`, `Dockerfile`

## Re-Test Results (After Fixes)

### Prompt Exfiltration
- Result: blocked with safe message and `SafetyPolicy` source.

### CORS
- Result: wildcard removed; disallowed origins do not get permissive headers.

### Upload Size
- Result: oversize uploads return `413 Request Entity Too Large`.

### Error Verbosity
- Result: malformed JSON returns `400 Invalid JSON payload.`; no raw stack/exception details.

### Docs/OpenAPI
- Result: `/docs` and `/openapi.json` return 404 by default.

### Rate Limiting
- Result: bursts triggered `429` responses as expected.

## Residual Risks / Notes
- Rate limiting is in-memory; horizontally scaled deployments will need a shared limiter (Redis) or gateway/WAF.
- LLM safety requires continuous testing (prompt injection, RAG poisoning, data exfil from context).
- Rotate any exposed provider keys and avoid storing secrets in repo or images.

## Next Phase (Phase 2) Plan
- Deeper API security testing: authz/IDOR patterns, business-logic abuse cases, request fuzzing.
- LLM-specific red-teaming: prompt injection with tool/context abuse, data exfil attempts, output policy checks.
- Supply chain: dependency inventory + SBOM, container image scanning workflow.
- Performance + DoS tests: sustained concurrency, timeout behavior, upload parser hardening.

