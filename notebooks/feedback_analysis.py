# notebooks/feedback_analysis.py
"""
Feedback Analysis — SQL queries on your real user feedback data.

Run this after collecting 20+ feedback events to generate insights.
Add the output charts to your README — shows SQL + data analysis skills.

Usage:
    cd project_root
    python notebooks/feedback_analysis.py
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "ragdb.sqlite3"


def connect():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run the app and collect some feedback first.")
        return None
    return sqlite3.connect(str(DB_PATH))


def run_analysis():
    conn = connect()
    if not conn:
        return

    print("=" * 55)
    print("  DOCIQ FEEDBACK ANALYSIS")
    print("=" * 55)

    # ── Query 1: Overall satisfaction ─────────────────────────────────────
    q1 = """
    SELECT
        COUNT(*) AS total_feedback,
        SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS positive,
        SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS negative,
        ROUND(
            100.0 * SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) / COUNT(*),
            1
        ) AS satisfaction_pct
    FROM feedback;
    """
    print("\n[1] Overall Satisfaction")
    print("-" * 40)
    row = conn.execute(q1).fetchone()
    if row and row[0]:
        print(f"  Total feedback   : {row[0]}")
        print(f"  Positive (👍)    : {row[1]}")
        print(f"  Negative (👎)    : {row[2]}")
        print(f"  Satisfaction     : {row[3]}%")
    else:
        print("  No feedback data yet")

    # ── Query 2: Daily satisfaction trend ─────────────────────────────────
    q2 = """
    SELECT
        DATE(created_at) AS date,
        COUNT(*) AS total,
        ROUND(
            100.0 * SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) / COUNT(*),
            1
        ) AS satisfaction_pct
    FROM feedback
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 14;
    """
    print("\n[2] Daily Satisfaction Trend (last 14 days)")
    print("-" * 40)
    rows = conn.execute(q2).fetchall()
    if rows:
        for r in rows:
            bar = "█" * int((r[2] or 0) / 5)
            print(f"  {r[0]}  {r[2]:5.1f}%  {bar}")
    else:
        print("  No data yet")

    # ── Query 3: Questions that got negative feedback ──────────────────────
    q3 = """
    SELECT question, correction
    FROM feedback
    WHERE rating = -1
    ORDER BY created_at DESC
    LIMIT 10;
    """
    print("\n[3] Negative Feedback — Questions to Improve")
    print("-" * 40)
    rows = conn.execute(q3).fetchall()
    if rows:
        for i, r in enumerate(rows, 1):
            print(f"  {i}. Q: {r[0][:60]}...")
            if r[1]:
                print(f"     Correction: {r[1][:60]}...")
    else:
        print("  No negative feedback yet")

    # ── Query 4: DPO pair count ───────────────────────────────────────────
    q4 = """
    SELECT COUNT(*) AS dpo_pairs, source
    FROM dpo_pairs
    GROUP BY source;
    """
    print("\n[4] DPO Training Pairs Available")
    print("-" * 40)
    rows = conn.execute(q4).fetchall()
    if rows:
        for r in rows:
            print(f"  {r[1]}: {r[0]} pairs")
        total = sum(r[0] for r in rows)
        ready = "✓ Ready for DPO training!" if total >= 200 else f"✗ Need {200-total} more pairs"
        print(f"  Total: {total} | {ready}")
    else:
        print("  No DPO pairs yet (need negative feedback + corrections)")

    # ── Query 5: Most common question patterns ────────────────────────────
    q5 = """
    SELECT
        CASE
            WHEN LOWER(question) LIKE '%what%'    THEN 'What questions'
            WHEN LOWER(question) LIKE '%how%'     THEN 'How questions'
            WHEN LOWER(question) LIKE '%why%'     THEN 'Why questions'
            WHEN LOWER(question) LIKE '%when%'    THEN 'When questions'
            WHEN LOWER(question) LIKE '%compare%' THEN 'Comparison questions'
            ELSE 'Other'
        END AS question_type,
        COUNT(*) AS count,
        ROUND(
            100.0 * AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END),
            1
        ) AS avg_satisfaction
    FROM feedback
    GROUP BY question_type
    ORDER BY count DESC;
    """
    print("\n[5] Question Type Analysis")
    print("-" * 40)
    rows = conn.execute(q5).fetchall()
    if rows:
        print(f"  {'Type':<25} {'Count':>6} {'Satisfaction':>14}")
        print(f"  {'-'*25} {'-'*6} {'-'*14}")
        for r in rows:
            print(f"  {r[0]:<25} {r[1]:>6} {str(r[2])+'%':>14}")
    else:
        print("  No data yet")

    conn.close()
    print("\n" + "=" * 55)
    print("  Add these insights to your README for bonus impact!")
    print("=" * 55)


if __name__ == "__main__":
    run_analysis()
