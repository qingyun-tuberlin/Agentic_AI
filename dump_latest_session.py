"""Dump the latest ADK session from a sqlite session.db in an LLM-readable form.

Usage:
    python dump_latest_session.py                     # latest session, text format
    python dump_latest_session.py --session <id>      # specific session
    python dump_latest_session.py --db path/to.db     # alternative db
    python dump_latest_session.py --jsonl             # one raw event JSON per line
    python dump_latest_session.py --truncate 4000     # cap each text/arg field

Schema assumed (ADK DatabaseSessionService):
    sessions(app_name, user_id, id, state TEXT, create_time, update_time)
    events(id, app_name, user_id, session_id, invocation_id, timestamp, event_data TEXT)
where event_data is a JSON blob containing content.parts (text / function_call /
function_response), author, branch, actions.state_delta, etc.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any

DEFAULT_DB = "machine_learning_engineering/.adk/session.db"


def pick_session(cur: sqlite3.Cursor, session_id: str | None) -> tuple[str, str, str, float]:
    if session_id:
        row = cur.execute(
            "SELECT app_name, user_id, id, update_time FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            sys.exit(f"no session with id={session_id}")
        return row
    # Latest session that actually has events; fall back to plain latest.
    row = cur.execute(
        """
        SELECT s.app_name, s.user_id, s.id, s.update_time
        FROM sessions s
        JOIN events e ON e.session_id = s.id
        GROUP BY s.id
        ORDER BY s.update_time DESC
        LIMIT 1
        """
    ).fetchone()
    if row:
        return row
    row = cur.execute(
        "SELECT app_name, user_id, id, update_time FROM sessions ORDER BY update_time DESC LIMIT 1"
    ).fetchone()
    if not row:
        sys.exit("no sessions in db")
    return row


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def trunc(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + f"\n…[truncated {len(s) - n} chars]"


def dump_part(part: dict[str, Any], truncate: int) -> list[str]:
    out: list[str] = []
    if "text" in part and part["text"] is not None:
        out.append("text:\n" + trunc(part["text"], truncate))
    if "function_call" in part and part["function_call"]:
        fc = part["function_call"]
        args = json.dumps(fc.get("args", {}), ensure_ascii=False, indent=2)
        out.append(f"tool_call {fc.get('name')}(args=\n{trunc(args, truncate)}\n)")
    if "function_response" in part and part["function_response"]:
        fr = part["function_response"]
        resp = fr.get("response", {})
        resp_s = json.dumps(resp, ensure_ascii=False, indent=2) if not isinstance(resp, str) else resp
        out.append(f"tool_response {fr.get('name')} ->\n{trunc(resp_s, truncate)}")
    if "thought" in part and part.get("thought"):
        # ADK sometimes marks parts as thoughts; surface text if present.
        out.append("(thought)")
    return out


def dump_event(idx: int, ev_row: sqlite3.Row, truncate: int) -> str:
    ev = json.loads(ev_row["event_data"])
    header = (
        f"### [{idx}] {fmt_ts(ev_row['timestamp'])}  "
        f"author={ev.get('author','?')}  "
        f"branch={ev.get('branch') or '-'}  "
        f"inv={ev.get('invocation_id','?')[:24]}  "
        f"id={ev.get('id','?')[:8]}"
    )
    lines = [header]
    content = ev.get("content") or {}
    role = content.get("role")
    if role:
        lines.append(f"role: {role}")
    for part in content.get("parts") or []:
        for chunk in dump_part(part, truncate):
            lines.append(chunk)
    actions = ev.get("actions") or {}
    sd = actions.get("state_delta") or {}
    if sd:
        lines.append(f"state_delta keys: {sorted(sd.keys())}")
    ad = actions.get("artifact_delta") or {}
    if ad:
        lines.append(f"artifact_delta: {ad}")
    transfer = actions.get("transfer_to_agent")
    if transfer:
        lines.append(f"transfer_to_agent: {transfer}")
    if actions.get("escalate"):
        lines.append("escalate: true")
    um = ev.get("usage_metadata")
    if um:
        lines.append(
            f"usage: prompt={um.get('prompt_token_count')} "
            f"completion={um.get('candidates_token_count')} "
            f"total={um.get('total_token_count')}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--session", default=None, help="session id (default: latest with events)")
    p.add_argument("--jsonl", action="store_true", help="emit raw event JSON per line")
    p.add_argument("--truncate", type=int, default=4000, help="max chars per field (0 = no cap)")
    p.add_argument("--list", action="store_true", help="list all sessions")
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if args.list:
        rows = cur.execute("SELECT id, app_name, user_id, update_time FROM sessions ORDER BY update_time DESC").fetchall()
        print("Sessions in database:")
        for r in rows:
            print(f"ID: {r['id']} | App: {r['app_name']} | Updated: {fmt_ts(r['update_time'])}")
        return

    # Check for search-errors option
    if args.session == "search-errors":
        print("Searching for errors/exceptions in recent events:")
        rows = cur.execute("SELECT session_id, timestamp, event_data FROM events ORDER BY timestamp DESC").fetchall()
        count = 0
        for r in rows:
            ev = json.loads(r["event_data"])
            content = json.dumps(ev.get("content", {}))
            if "traceback" in content.lower() or "error" in content.lower() or "exception" in content.lower():
                # Print summary
                print(f"Session: {r['session_id']} | Time: {fmt_ts(r['timestamp'])}")
                # Extract error message
                for part in ev.get("content", {}).get("parts", []):
                    text = part.get("text", "")
                    if text and ("traceback" in text.lower() or "error" in text.lower() or "exception" in text.lower()):
                        print(trunc(text, 1000))
                        print("-" * 50)
                count += 1
                if count >= 10:
                    break
        return

    app, user, sid, upd = pick_session(cur, args.session)
    events = cur.execute(
        "SELECT id, timestamp, event_data FROM events WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
        (sid,),
    ).fetchall()

    if args.jsonl:
        for ev in events:
            data = json.loads(ev["event_data"])
            data.setdefault("timestamp", ev["timestamp"])
            print(json.dumps(data, ensure_ascii=False))
        return

    print(f"# Session {sid}")
    print(f"app={app} user={user} updated={fmt_ts(upd)} events={len(events)}")
    print()
    for i, ev in enumerate(events, 1):
        print(dump_event(i, ev, args.truncate))
        print()


if __name__ == "__main__":
    main()
