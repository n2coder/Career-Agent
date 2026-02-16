# Production Deployment Runbook (Phase 5)

## 1) Prepare Secrets

- Copy `.env.production.example` -> `.env.production`
- Set strong values for:
  - `APP_API_KEY`
  - `OPENAI_API_KEY` (or `HUGGINGFACE_API_KEY` if using HF)
- Keep `.env.production` out of source control.

## 2) Validate Production Profile

- Run:
  - `python scripts/validate_deployment_profile.py --env-file .env.production`

Expected: `Deployment profile validation passed.`

## 3) Deploy Behind Reverse Proxy

- Use:
  - `docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build`
- Public entrypoint is nginx (`reverse-proxy`) on port 80.
- API container is internal-only (`expose: 8001`).

## 4) TLS Termination

- Recommended: terminate TLS at external LB/ingress (Nginx ingress, Cloudflare, ALB, etc.)
- Forward headers:
  - `X-Forwarded-For`
  - `X-Forwarded-Proto=https`
  - `X-Request-ID`

## 5) Post-Deploy Checks

- `GET /health` returns `200`
- `GET /status` without key returns `401`
- `GET /status` with key returns `200`
- Confirm response headers include:
  - `X-Request-ID`
  - `X-Process-Time-Ms`
  - `X-Content-Type-Options`

## 6) Backup + Recovery Checklist

- Store config backups securely (excluding secrets in plain text).
- Keep reproducible image tags/releases for rollback.
- Document rollback command:
  - `docker compose -f docker-compose.prod.yml down`
  - redeploy last known-good tag/config
- Verify data/state implications:
  - Session state is in-memory and ephemeral.
  - No persistent DB backup required for app state in current architecture.

## 7) Secret Rotation Policy

- Rotate `APP_API_KEY` and provider keys regularly (e.g., every 30-90 days).
- Rotate immediately on any suspected exposure.
- Validate auth flow after rotation.

