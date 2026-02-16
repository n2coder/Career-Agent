import argparse
import datetime as dt
import re
from pathlib import Path

import requests

KNOWLEDGE_DIR = Path("knowledge_base")
REFRESH_LOG = KNOWLEDGE_DIR / "00_refresh_log.md"

SOURCE_URLS = [
    "https://www.nasscom.in/knowledge-center/publications/technology-sector-india-strategic-review-2025",
    "https://www.jll.com/en-in/insights/market-dynamics/india-office.html",
    "https://www.jll.com/en-in/newsroom/office-market-soars-gccs-and-domestic-demand-drive-q2-2025-growth",
    "https://www.jll.com/en-in/newsroom/gccs-drive-record-77-2-mn-sqft-office-leasing-in-india.html",
    "https://www.aon.com/apac/in-the-press/asia-newsroom/2025/salaries-in-india-projected-to-increase-by-nine-percent-in-2026-aon-study",
    "https://www.naukri.com/blog/",
    "https://www.numbeo.com/cost-of-living/in/Bangalore",
    "https://www.numbeo.com/cost-of-living/in/Hyderabad",
    "https://www.numbeo.com/cost-of-living/in/Pune",
    "https://www.numbeo.com/cost-of-living/in/Chennai",
    "https://www.numbeo.com/cost-of-living/in/Gurgaon",
    "https://www.numbeo.com/cost-of-living/in/Noida",
    "https://www.numbeo.com/cost-of-living/in/Mumbai",
]


def fetch_source_status(url: str, timeout: int = 15) -> dict:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        status = r.status_code
        html = r.text[:120000]
        title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = ""
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        return {
            "url": url,
            "ok": 200 <= status < 400,
            "status": status,
            "title": title or "(no title parsed)",
        }
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "status": "ERROR",
            "title": str(exc),
        }


def write_refresh_log(results: list[dict], notes: str) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Knowledge Base Refresh Log",
        "",
        f"Updated at: {now}",
        "",
        "## Source Check",
        "",
        "| Source | Status | Title/Info |",
        "|---|---:|---|",
    ]

    for item in results:
        status_text = f"OK ({item['status']})" if item["ok"] else f"FAIL ({item['status']})"
        title = item["title"].replace("|", "\\|")
        lines.append(f"| {item['url']} | {status_text} | {title} |")

    if notes:
        lines.extend(["", "## Notes", "", notes.strip(), ""])

    lines.extend(
        [
            "## Manual Follow-up Checklist",
            "",
            "- Revalidate salary ranges and hike assumptions in `knowledge_base/05_india_it_market_2026.md`.",
            "- Revalidate rent/cost signals in `knowledge_base/02_top_it_cities.md`.",
            "- Revalidate role demand bullets in `knowledge_base/03_skills_and_roadmaps.md`.",
            "- Update snapshot date in edited files.",
            "",
        ]
    )

    REFRESH_LOG.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and validate KB source links for India IT RAG data.")
    parser.add_argument("--notes", default="", help="Optional notes to include in refresh log.")
    parser.add_argument("--quick", action="store_true", help="Check only first 5 sources for quick validation.")
    args = parser.parse_args()

    urls = SOURCE_URLS[:5] if args.quick else SOURCE_URLS
    results = [fetch_source_status(url) for url in urls]
    write_refresh_log(results, args.notes)

    ok_count = sum(1 for r in results if r["ok"])
    print(f"Refresh log written to: {REFRESH_LOG}")
    print(f"Sources reachable: {ok_count}/{len(results)}")


if __name__ == "__main__":
    main()
