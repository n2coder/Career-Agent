import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from engine import RecruitmentEngine


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_skill_compare_json():
    eng = RecruitmentEngine()
    resume = """Naresh Chaudhary
Skills: Python, FastAPI, AWS, Docker, Linux, Git, SQL, PostgreSQL, Redis
Experience: Built APIs with FastAPI and deployed on AWS using Docker.
"""
    required = ["Python", "Kubernetes", "FastAPI", "Terraform", "PostgreSQL"]
    out = eng._build_skill_compare_json(resume, required)
    _assert("extracted_skills" in out and "missing_skills" in out and "recommendations" in out, "Bad keys")
    # Evidence-only: must not invent Kubernetes.
    _assert(not any(s.lower() == "kubernetes" for s in out["extracted_skills"]), "Invented skill")
    _assert(any(s.lower() == "kubernetes" for s in [x.lower() for x in out["missing_skills"]]), "Missing detection failed")
    # Recommendations must be URLs
    for k, urls in out["recommendations"].items():
        _assert(isinstance(urls, list) and urls and all(u.startswith("http") for u in urls), f"Bad URLs for {k}")


def test_salary_guard():
    eng = RecruitmentEngine()
    allowed = {"allowed": {"12-18 LPA", "9%"}, "salary_ranges": {"12-18 LPA"}, "percents": {"9%"}, "rents": set()}
    ans = "## Salary\n- Typical range: 12-18 LPA\n- Some claim: 30-40 LPA\n- Hikes: 9%\n"
    guarded = eng._apply_salary_guard(ans, allowed)
    _assert("30-40" not in guarded, "Ungrounded number not removed")
    _assert("12-18" in guarded and "9%" in guarded, "Grounded facts removed incorrectly")


if __name__ == "__main__":
    test_skill_compare_json()
    test_salary_guard()
    print(json.dumps({"ok": True}))
