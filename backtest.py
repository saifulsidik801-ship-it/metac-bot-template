"""Backtest saved bot forecasts against actual Metaculus resolutions.

Scores only questions this bot actually forecasted AND saved (the bot saves
a JSON report per question when `folder_to_save_reports_to` is set in
main.py). No saved reports -> nothing to score; the script says so and
exits 0.

Scoring (vs ACTUAL resolutions — never vs the community prediction; that
path is deprecated, Metaculus removed the `aggregations` field):
- Binary: Brier score (p - outcome)^2, lower is better.
- Multiple choice: Brier sum over options, lower is better.
- Numeric/date: approximate log score from the saved CDF (density at the
  resolving value, interpolated between the bracketing percentiles).

Leakage screen: the saved explanation/research text is scanned for dates
(YYYY-MM-DD) later than the question's close date. A research briefing
that cites post-close information is leaked and the question is flagged —
its score is reported separately so it cannot flatter the backtest.

Usage:
    METACULUS_TOKEN=... python backtest.py --reports-dir ./reports
    python backtest.py --reports-dir ./reports --limit 20 --dry-run

Needs only the standard library. Never publishes anything.
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://www.metaculus.com/api"
DATE_RE = re.compile(r"\b(20\d\d)-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b")


def _get_nested(data, *paths):
    """Try several key paths, return the first hit or None."""
    for path in paths:
        current = data
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError, IndexError):
            continue
    return None


def fetch_question(question_id, token):
    url = f"{API_BASE}/questions/{question_id}/"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Token {token}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_close_date(question_json):
    for key in ("scheduled_close_time", "close_time", "actual_close_time"):
        raw = question_json.get(key)
        if raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def leakage_dates(explanation, close_date):
    """Dates in the research text that fall after the question closed."""
    if not close_date or not explanation:
        return []
    found = []
    for match in DATE_RE.finditer(explanation):
        try:
            date = datetime.fromisoformat(match.group(0)).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if date.date() > close_date.date():
            found.append(match.group(0))
    return sorted(set(found))


def score_binary(prediction, resolution):
    if resolution == "yes":
        outcome = 1.0
    elif resolution == "no":
        outcome = 0.0
    else:
        return None
    return (float(prediction) - outcome) ** 2


def score_multiple_choice(prediction, resolution):
    options = prediction.get("predicted_options", [])
    if not options or resolution is None:
        return None
    total = 0.0
    for option in options:
        outcome = 1.0 if option.get("option_name") == resolution else 0.0
        total += (float(option.get("probability", 0.0)) - outcome) ** 2
    return total


def score_numeric(prediction, resolution):
    """Approximate log score: -log(density at the resolving value)."""
    try:
        resolved = float(resolution)
    except (TypeError, ValueError):
        return None
    percentiles = prediction.get("declared_percentiles") or prediction.get(
        "percentiles"
    )
    if not percentiles:
        return None
    points = sorted(
        (float(p["percentile"]), float(p["value"])) for p in percentiles
    )
    if len(points) < 2:
        return None
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if v0 <= resolved <= v1 and v1 > v0:
            density = (p1 - p0) / (v1 - v0)
            return -math.log(max(density, 1e-12)) if density > 0 else None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.getenv("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN is not set.", file=sys.stderr)
        return 2
    if not os.path.isdir(args.reports_dir):
        print(f"Reports dir not found: {args.reports_dir}", file=sys.stderr)
        return 2

    files = sorted(
        f for f in os.listdir(args.reports_dir) if f.endswith(".json")
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(
            f"No saved reports in {args.reports_dir}. The bot saves one "
            "JSON per question when folder_to_save_reports_to is set in "
            "main.py — backtest after the bot has run."
        )
        return 0

    rows = []
    for filename in files:
        path = os.path.join(args.reports_dir, filename)
        try:
            report = json.load(open(path))
        except (OSError, ValueError) as e:
            print(f"SKIP {filename}: unreadable ({e})")
            continue
        question_id = _get_nested(
            report,
            ("question", "id_of_question"),
            ("question", "id"),
            ("question_id",),
            ("question", "id_of_post"),
        )
        prediction = report.get("prediction")
        explanation = report.get("explanation") or ""
        if question_id is None or prediction is None:
            print(
                f"SKIP {filename}: no question id / prediction in this "
                "report format"
            )
            continue
        rows.append((filename, question_id, prediction, explanation))

    if args.dry_run:
        print(f"{len(rows)} report(s) would be scored:")
        for filename, question_id, _, _ in rows:
            print(f"  {filename} -> question {question_id}")
        return 0

    binary_scores, mc_scores, numeric_scores = [], [], []
    leaked = 0
    for filename, question_id, prediction, explanation in rows:
        try:
            question_json = fetch_question(question_id, token)
        except Exception as e:
            print(f"SKIP {filename}: API fetch failed ({e})")
            continue
        resolution = question_json.get("resolution")
        if resolution in (None, "annulled"):
            print(f"SKIP {filename}: question {question_id} unresolved")
            continue
        close_date = parse_close_date(question_json)
        bad_dates = leakage_dates(explanation, close_date)
        score, kind = None, None
        if isinstance(prediction, (int, float)):
            score, kind = score_binary(prediction, resolution), "binary"
        elif isinstance(prediction, dict) and "predicted_options" in prediction:
            score, kind = (
                score_multiple_choice(prediction, resolution),
                "multiple_choice",
            )
        elif isinstance(prediction, dict):
            score, kind = score_numeric(prediction, resolution), "numeric"
        if score is None:
            print(f"SKIP {filename}: cannot score this prediction shape")
            continue
        if kind == "binary":
            binary_scores.append(score)
        elif kind == "multiple_choice":
            mc_scores.append(score)
        else:
            numeric_scores.append(score)
        flag = ""
        if bad_dates:
            leaked += 1
            flag = f"  LEAKED? post-close dates in research: {bad_dates}"
        print(
            f"{filename} q={question_id} {kind} score={score:.4f}{flag}"
        )

    def mean(values):
        return sum(values) / len(values) if values else None

    print("\n--- Backtest summary (vs actual resolutions) ---")
    print(f"questions scored: {len(binary_scores) + len(mc_scores) + len(numeric_scores)}")
    if binary_scores:
        print(f"binary Brier (lower better): {mean(binary_scores):.4f} over {len(binary_scores)}")
    if mc_scores:
        print(f"multiple-choice Brier (lower better): {mean(mc_scores):.4f} over {len(mc_scores)}")
    if numeric_scores:
        print(f"numeric approx log score (lower better): {mean(numeric_scores):.4f} over {len(numeric_scores)}")
    print(f"leakage flags: {leaked} (scores above include them — rerun after fixing the research if any)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
