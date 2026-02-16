# Knowledge Base Guide

This folder is the active retrieval context for the app.

## Files
- `01_industry_overview.md`: India IT industry and hiring snapshot.
- `02_top_it_cities.md`: cities, job corridors, and cost signals.
- `03_skills_and_roadmaps.md`: role-based skill maps and project strategy.
- `04_career_planning.md`: level-wise career strategy.
- `05_india_it_market_2026.md`: consolidated latest-cycle market dossier.
- `00_refresh_log.md`: generated source health check log.

## Monthly Refresh Command
Run from project root:

```bash
.\\venv\\Scripts\\python.exe refresh_kb.py --notes "monthly refresh"
```

Quick check mode:

```bash
.\\venv\\Scripts\\python.exe refresh_kb.py --quick
```

After refresh:
- Update snapshot dates in edited markdown files.
- Restart app so `engine.py` reloads updated KB chunks.
