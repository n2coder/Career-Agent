# Lin.O - Indian Job RAG Agent

AI-powered career assistant for India-focused job guidance, resume analysis, fresher resume creation, roadmap planning, and monitoring-ready operations.
<img width="1954" height="1264" alt="1" src="https://github.com/user-attachments/assets/ea1aba95-2362-4b4f-84d7-e883ed660dd7" />
<img width="1375" height="1250" alt="2" src="https://github.com/user-attachments/assets/837b8bdd-c202-43bd-bc3c-91f2f30656cd" />
<img width="614" height="928" alt="3" src="https://github.com/user-attachments/assets/f2447419-1e94-4912-b2a1-5cef84351c02" />
<img width="2476" height="1234" alt="4" src="https://github.com/user-attachments/assets/957ccd54-dda6-4c80-b240-50d4204b04b1" />

## Highlights

- `Lin.O` branded career agent UI (responsive desktop/mobile)
- Resume upload and profile-aware responses
- Fresher guided resume flow with PDF generation
- Quick roadmap actions for top IT roles
- OpenAI-backed LLM routing (`gpt-4o-mini` by default)
- API key enforcement for app and separate key for monitoring
- Production Docker setup with Nginx reverse proxy
- Monitoring endpoints for visitors, queries, and resume activity

## Screens and UX

- Sidebar: status, roadmap shortcuts, profile tools
- Chat area: user/assistant conversation, markdown rendering
- Resume tools: upload, analyze, builder, skill upgrades, salary insights
- Downloadable resume PDF from generated draft

## Architecture

```text
Browser
  -> Nginx (reverse proxy :80)
      -> FastAPI app (Uvicorn :8001)
          -> RecruitmentEngine (RAG + provider routing)
          -> Knowledge base docs (knowledge_base/)
          -> Session state + monitoring event stores
```

## Tech Stack

- Backend: `FastAPI`, `Uvicorn`
- LLM: `OpenAI` (via `OPENAI_API_KEY`, model configurable)
- Parsing: `pdfplumber`, `python-docx`
- Frontend: single-page `index.html` (Tailwind/CDN + custom JS/CSS)
- Infra: `Docker`, `docker-compose`, `Nginx`

## Project Structure

```text
.
|- main.py
|- engine.py
|- index.html
|- requirements.txt
|- Dockerfile
|- docker-compose.prod.yml
|- ops/nginx/
|- knowledge_base/
|- scripts/
|- security_reports/
`- assets/
```

## API Endpoints

### App

- `GET /` - UI
- `GET /health` - liveness
- `GET /status` - readiness/provider/docs count
- `POST /query` - ask Lin.O
- `GET /resume/status` - resume profile state
- `POST /resume/upload` - upload resume file
- `POST /resume/clear` - clear profile

### Monitoring

- `GET /monitoring/summary`
- `GET /monitoring/queries`
- `GET /monitoring/resumes/uploads`
- `GET /monitoring/resumes/built`
- `GET /monitoring/dashboard`

## Local Run (Dev)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Open:

- App: `http://localhost:8001`

## Production Run (Docker)

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

Check:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production ps
curl http://localhost/health
```

## Environment Configuration

Start from `.env.production.example` and create `.env.production`.

### Core Security

- `API_KEY_REQUIRED=true`
- `APP_API_KEY=<long_random_secret>`
- `DISABLE_DOCS=true`
- `ALLOW_NULL_ORIGIN=false`
- `CORS_ORIGINS=https://your-domain`

### Runtime Limits

- `TRUST_X_FORWARDED_FOR=true`
- `RATE_LIMIT_WINDOW_SEC=60`
- `RATE_LIMIT_QUERY_PER_WINDOW=30`
- `RATE_LIMIT_UPLOAD_PER_WINDOW=10`
- `MAX_UPLOAD_BYTES=5242880`
- `MAX_QUERY_BYTES=20000`
- `SESSION_TTL_SEC=3600`
- `MAX_SESSIONS=500`

### LLM Provider

- `LLM_PROVIDER=openai`
- `OPENAI_MODEL=gpt-4o-mini`
- `OPENAI_API_KEY=<openai_key>`

### Monitoring Security

- `MONITORING_KEY_REQUIRED=true`
- `MONITORING_API_KEY=<separate_secret>`
- `MONITORING_ALLOWED_IPS=<optional_allowlist>`
- `MONITORING_RETENTION_SEC=259200`
- `MONITORING_CAPTURE_QUERY_TEXT=false`
- `MONITORING_CAPTURE_RESUME_TEXT=false`
- `MONITORING_CAPTURE_RESUME_BUILD_TEXT=false`
- `MONITORING_CAPTURE_CLIENT_METADATA=false`

## Security Notes

- Keep `.env` and `.env.production` out of Git (already ignored).
- Rotate keys if exposed:
  - `OPENAI_API_KEY`
  - `APP_API_KEY`
  - `MONITORING_API_KEY`
- Use separate secrets for app traffic and monitoring access.
- Restrict monitoring via IP allowlist where possible.

## Validation Scripts

```bash
python scripts/validate_production_config.py --env-file .env.production
python scripts/validate_deployment_profile.py --env-file .env.production
python scripts/hallucination_guards_smoketest.py
```

## Roadmap / Reports

Security and hardening reports are available in:

- `security_reports/phase1_report.md`
- `security_reports/phase2_report.md`
- `security_reports/phase3_report.md`
- `security_reports/phase4_report.md`
- `security_reports/phase5_report.md`
- `security_reports/production_runbook.md`

## Branding Assets

- Avatar image path: `assets/lino-face.png`

## License

Private.
