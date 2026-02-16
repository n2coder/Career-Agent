import argparse
from pathlib import Path


def parse_env(path: Path) -> dict:
    values = {}
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip().lstrip("\ufeff")
        values[key] = v.strip().strip('"').strip("'")
    return values


def truthy(v: str) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def validate(env: dict) -> list[str]:
    errors = []

    if not truthy(env.get("API_KEY_REQUIRED", "")):
        errors.append("API_KEY_REQUIRED must be true in production.")
    if not truthy(env.get("DISABLE_DOCS", "")):
        errors.append("DISABLE_DOCS must be true in production.")
    if truthy(env.get("ALLOW_NULL_ORIGIN", "false")):
        errors.append("ALLOW_NULL_ORIGIN must be false in production.")
    if not truthy(env.get("TRUST_X_FORWARDED_FOR", "")):
        errors.append("TRUST_X_FORWARDED_FOR should be true behind reverse proxy.")

    cors = env.get("CORS_ORIGINS", "")
    origins = [x.strip() for x in cors.split(",") if x.strip()]
    if not origins:
        errors.append("CORS_ORIGINS must be set.")
    else:
        lowered = {o.lower() for o in origins}
        if "null" in lowered or "*" in lowered:
            errors.append("CORS_ORIGINS must not contain null or * in production.")
        if any(o.startswith("http://") for o in origins):
            errors.append("CORS_ORIGINS must use https:// origins.")

    api_key = env.get("APP_API_KEY", "")
    if len(api_key) < 24:
        errors.append("APP_API_KEY must be at least 24 chars.")

    if not truthy(env.get("MONITORING_KEY_REQUIRED", "true")):
        errors.append("MONITORING_KEY_REQUIRED must be true in production.")
    mon_key = env.get("MONITORING_API_KEY", "")
    if len(mon_key.strip()) < 24:
        errors.append("MONITORING_API_KEY must be at least 24 chars.")
    if truthy(env.get("MONITORING_CAPTURE_QUERY_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_QUERY_TEXT must be false in production.")
    if truthy(env.get("MONITORING_CAPTURE_RESUME_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_RESUME_TEXT must be false in production.")
    if truthy(env.get("MONITORING_CAPTURE_RESUME_BUILD_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_RESUME_BUILD_TEXT must be false in production.")

    provider = (env.get("LLM_PROVIDER") or "").strip().lower()
    if provider not in {"openai", "hf"}:
        errors.append("LLM_PROVIDER must be one of: openai, hf.")

    if provider == "openai" and not env.get("OPENAI_API_KEY", "").strip():
        errors.append("OPENAI_API_KEY must be set when LLM_PROVIDER=openai.")
    if provider == "hf" and not env.get("HUGGINGFACE_API_KEY", "").strip():
        errors.append("HUGGINGFACE_API_KEY must be set when LLM_PROVIDER=hf.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production deployment env profile.")
    parser.add_argument("--env-file", default=".env.production", help="Path to production env file")
    args = parser.parse_args()

    env = parse_env(Path(args.env_file))
    errors = validate(env)
    if errors:
        print("Deployment profile validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Deployment profile validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
