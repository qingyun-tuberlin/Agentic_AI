#!/usr/bin/env python3
"""
Track total token usage of the latest session in the local SQLite database.
Usage:
    python scripts/track_latest_tokens.py [--db PATH_TO_DB]
"""

import os
import sqlite3
import json
import argparse

DEFAULT_DB = "machine_learning_engineering/.adk/session.db"

def main():
    parser = argparse.ArgumentParser(description="Track token usage of the latest session.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite session database")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database file not found at {args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Find the latest session that has events
    row = cur.execute("""
        SELECT s.id, s.update_time
        FROM sessions s
        JOIN events e ON e.session_id = s.id
        GROUP BY s.id
        ORDER BY s.update_time DESC
        LIMIT 1
    """).fetchone()

    # Fall back to the absolute latest session if no events exist
    if not row:
        row = cur.execute("SELECT id, update_time FROM sessions ORDER BY update_time DESC LIMIT 1").fetchone()

    if not row:
        print("Error: No sessions found in the database.")
        conn.close()
        return 1

    session_id, _ = row
    print(f"Session ID: {session_id}")

    # Query all event data for the session
    cur.execute("SELECT event_data FROM events WHERE session_id = ?", (session_id,))
    events = cur.fetchall()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    api_calls = 0

    for (event_str,) in events:
        try:
            data = json.loads(event_str)
            usage = data.get("usage_metadata")
            if usage:
                prompt_tokens += usage.get("prompt_token_count", 0)
                completion_tokens += usage.get("candidates_token_count", 0)
                total_tokens += usage.get("total_token_count", 0)
                api_calls += 1
        except Exception:
            pass

    print(f"Total API Calls:         {api_calls}")
    print(f"Total Prompt Tokens:     {prompt_tokens:,}")
    print(f"Total Completion Tokens: {completion_tokens:,}")
    print(f"Total Tokens:            {total_tokens:,}")

    conn.close()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
