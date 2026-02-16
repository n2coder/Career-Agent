import argparse
from pathlib import Path


def parse_env_file(path: Path) -> dict:
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


def validate(env: dict, strict: bool = False) -> list[str]:
    errors = []

    if not truthy(env.get("API_KEY_REQUIRED", "")):
        errors.append("API_KEY_REQUIRED must be true.")
    if not truthy(env.get("DISABLE_DOCS", "")):
        errors.append("DISABLE_DOCS must be true.")
    if truthy(env.get("ALLOW_NULL_ORIGIN", "false")):
        errors.append("ALLOW_NULL_ORIGIN must be false for production.")

    cors = env.get("CORS_ORIGINS", "").strip()
    if not cors:
        errors.append("CORS_ORIGINS must be set in production.")
    else:
        origins = [x.strip() for x in cors.split(",") if x.strip()]
        lowered = {o.lower() for o in origins}
        if "*" in lowered:
            errors.append("CORS_ORIGINS must not include '*'.")
        if "null" in lowered:
            errors.append("CORS_ORIGINS must not include 'null' in production.")
        if strict and any(o.startswith("http://") for o in origins):
            errors.append("CORS_ORIGINS should use https:// only in strict mode.")

    app_key = env.get("APP_API_KEY", "").strip()
    if not app_key:
        errors.append("APP_API_KEY must be set.")
    elif strict and len(app_key) < 24:
        errors.append("APP_API_KEY must be at least 24 chars in strict mode.")

    if not truthy(env.get("MONITORING_KEY_REQUIRED", "true")):
        errors.append("MONITORING_KEY_REQUIRED should be true in production.")
    mon_key = env.get("MONITORING_API_KEY", "").strip()
    if not mon_key:
        errors.append("MONITORING_API_KEY should be set for monitoring endpoints.")
    elif strict and len(mon_key) < 24:
        errors.append("MONITORING_API_KEY should be at least 24 chars in strict mode.")

    if truthy(env.get("MONITORING_CAPTURE_QUERY_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_QUERY_TEXT should be false in production.")
    if truthy(env.get("MONITORING_CAPTURE_RESUME_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_RESUME_TEXT should be false in production.")
    if truthy(env.get("MONITORING_CAPTURE_RESUME_BUILD_TEXT", "false")):
        errors.append("MONITORING_CAPTURE_RESUME_BUILD_TEXT should be false in production.")

    retention = env.get("MONITORING_RETENTION_SEC", "").strip()
    if retention:
        try:
            if int(retention) < 3600:
                errors.append("MONITORING_RETENTION_SEC should be >= 3600.")
        except ValueError:
            errors.append("MONITORING_RETENTION_SEC must be an integer.")

    if strict and truthy(env.get("TRUST_X_FORWARDED_FOR", "false")):
        errors.append("TRUST_X_FORWARDED_FOR should be false unless behind trusted proxy.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production-safe env policy.")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--strict", action="store_true", help="Enable stricter production checks")
    args = parser.parse_args()

    env = parse_env_file(Path(args.env_file))
    errors = validate(env, strict=args.strict)
    if errors:
        print("Production config validation failed:")
        for e in errors:
            print(f"- {e}")
        return 1
    print("Production config validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
