"""Diagnose why norm extraction failed. Run on a transcript that returned
'No norms extracted'. Prints the structure, the final round, and what
parse_numbered_list does with each final-round agent's content.
"""

import json
import sys
from analysis_shared import load_transcript, parse_numbered_list

if len(sys.argv) < 2:
    print("Usage: python3 diagnose_transcript.py <transcript.json>")
    sys.exit(1)

t = load_transcript(sys.argv[1])

print("=== Top-level keys ===")
for k, v in t.items():
    if isinstance(v, list):
        print(f"  {k}: list of {len(v)} items")
        if v and isinstance(v[0], dict):
            print(f"    first item keys: {list(v[0].keys())}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    else:
        preview = str(v)[:80]
        print(f"  {k}: {preview}")

# Try to find the turns array under common key names
turns_key = None
for candidate in ("turns", "messages", "transcript", "history", "rounds", "conversation"):
    if candidate in t and isinstance(t[candidate], list):
        turns_key = candidate
        break

if turns_key is None:
    print("\nCould not find a list of turns under common keys.")
    print("Top-level structure shown above. Tell me which key holds the deliberation turns.")
    sys.exit(0)

print(f"\n=== Using '{turns_key}' as turns array ===")
turns = t[turns_key]
print(f"First turn keys: {list(turns[0].keys()) if turns else 'empty'}")
print(f"First turn sample (truncated):")
if turns:
    for k, v in turns[0].items():
        preview = str(v)[:120].replace("\n", " ")
        print(f"  {k}: {preview}")

# Find round/turn marker
round_key = None
for candidate in ("round", "round_num", "turn", "turn_num", "step", "iteration"):
    if turns and candidate in turns[0]:
        round_key = candidate
        break

if round_key is None:
    print("\nCould not find a round/turn marker. Available keys per turn:")
    if turns:
        print(f"  {list(turns[0].keys())}")
    sys.exit(0)

print(f"\n=== Using '{round_key}' as round marker ===")
rounds = sorted(set(t.get(round_key) for t in turns if t.get(round_key) is not None))
print(f"Rounds present: {rounds}")
max_round = rounds[-1] if rounds else None
print(f"Max round: {max_round}")

final = [tn for tn in turns if tn.get(round_key) == max_round]
print(f"\n=== Final round has {len(final)} turns ===")

# Find content key
content_key = None
for candidate in ("content", "text", "message", "response", "output", "reply"):
    if final and candidate in final[0]:
        content_key = candidate
        break

print(f"Content key: {content_key}")
if not content_key:
    print("No content key found. Available keys:", list(final[0].keys()) if final else "[]")
    sys.exit(0)

agent_key = None
for candidate in ("agent", "agent_id", "speaker", "role", "name", "from"):
    if final and candidate in final[0]:
        agent_key = candidate
        break
print(f"Agent key: {agent_key}")

print("\n=== Final-round content samples + parse results ===")
for i, tn in enumerate(final):
    agent = tn.get(agent_key, f"turn_{i}") if agent_key else f"turn_{i}"
    content = tn.get(content_key, "") if content_key else ""
    print(f"\n--- {agent} ---")
    print(f"Content length: {len(content)} chars")
    print(f"First 400 chars:")
    print(content[:400])
    print(f"...")
    parsed = parse_numbered_list(content)
    print(f"parse_numbered_list found: {len(parsed)} items")
    if parsed:
        for j, p in enumerate(parsed[:3], 1):
            print(f"  {j}. {p[:80]}...")
