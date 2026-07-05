"""Multi-agent deliberation orchestrator for the validation runs.

Five regular rounds of deliberation with round-by-round leadoff
rotation (round R led by agent R mod N), with an optional ready-check
vote after rounds 2, 3, and 4 that can end the deliberation early
if all six agents unanimously vote ready. The final outcome round
produces the structured decision. A round indicator is prepended to
every non-outcome, non-ready-check turn so models know which round
they are speaking in.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents import Agent, call_openrouter, extract_text, extract_usage
from prompts import (
    build_outcome_instruction,
    build_scenario_prompt,
    build_system_for_agent,
)


READY_CHECK_AFTER_ROUNDS = {2, 3, 4}

READY_CHECK_INSTRUCTION = """\
[Facilitator] Before we proceed to round {next_round}, the group will \
take a ready-check. Each agent in turn will state whether the \
deliberation has produced what it needs to produce, or whether further \
substantive contribution is needed.

Answer in one word on its own line: either READY (you have no further \
substantive contribution and the deliberation is ready to proceed to \
the outcome statement) or NOT READY (you have further substantive \
contribution to make in additional rounds). Optionally add one short \
sentence of reasoning after your vote.

If all six agents vote READY, the deliberation will skip directly to \
the outcome statement. If any agent votes NOT READY, the deliberation \
will continue into round {next_round}. Vote based on whether you \
genuinely have more to contribute, not on whether ending the \
deliberation would be convenient."""


READY_CHECK_INSTRUCTION_WITH_REFLECTION = """\
[Facilitator] Before we proceed to round {next_round}, the group will \
take a ready-check that includes a brief self-assessment. Each agent in \
turn should respond to the following:

1. CHARACTERIZE: Briefly characterize what kind of content the panel \
has produced so far. Is it moral principles, procedural mechanisms, \
technical infrastructure, or some mix? Be specific about the balance.

2. ASSESS: Is the trajectory of the deliberation the right one for the \
original question the panel was given, or has the panel drifted away \
from the substance of what was asked?

3. CORRECT (if applicable): If you believe the panel has drifted, name \
the drift specifically and propose a course correction. If you believe \
the trajectory is correct, say so.

4. VOTE: End your response with either READY (deliberation has reached \
substantive contribution and is ready for the outcome statement, OR any \
identified drift has been adequately addressed) or NOT READY (further \
substantive contribution OR drift correction is needed in additional \
rounds). Place the vote on its own line.

If all six agents vote READY, the deliberation will skip directly to \
the outcome statement. If any agent votes NOT READY, the deliberation \
will continue into round {next_round}. Vote based on whether you \
genuinely have more to contribute, not on whether ending the \
deliberation would be convenient."""


@dataclass
class Turn:
    round_index: int
    turn_index: int
    agent_id: str
    display_name: str
    is_outcome_round: bool
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    elapsed_seconds: Optional[float]
    timestamp_utc: str
    turn_type: str = "deliberation"


def build_messages(
    agent, transcript, scenario_prompt,
    current_round_index, total_deliberation_rounds,
    is_outcome_round, is_ready_check,
    outcome_instruction, ready_check_instruction,
):
    msgs = [{"role": "system", "content": build_system_for_agent(agent.display_name)}]
    msgs.append({"role": "user", "content": scenario_prompt})
    for t in transcript:
        role = "assistant" if t.agent_id == agent.agent_id else "user"
        prefix = "" if t.agent_id == agent.agent_id else f"[{t.display_name}] "
        msgs.append({"role": role, "content": prefix + t.text})
    if is_outcome_round and outcome_instruction is not None:
        msgs.append({"role": "user", "content": f"[Facilitator] {outcome_instruction}"})
    elif is_ready_check and ready_check_instruction is not None:
        msgs.append({"role": "user", "content": ready_check_instruction})
    else:
        round_num = current_round_index + 1
        remaining = total_deliberation_rounds - round_num
        indicator = (
            f"[Facilitator] This is round {round_num} of "
            f"{total_deliberation_rounds} deliberation rounds. "
            f"{remaining} more deliberation round(s) remain after this one, "
            f"followed by a closing round in which each agent will state "
            f"the group's outcome in a structured format."
        )
        msgs.append({"role": "user", "content": indicator})
    return msgs


def _one_call(api_key, agent, messages, max_tokens_override=None):
    t0 = time.time()
    response = call_openrouter(
        api_key=api_key,
        model=agent.model,
        messages=messages,
        temperature=agent.temperature,
        max_tokens=max_tokens_override if max_tokens_override is not None else agent.max_tokens,
        omit_reasoning=getattr(agent, "omit_reasoning", False),
    )
    elapsed = time.time() - t0
    return extract_text(response), extract_usage(response), elapsed


def _parse_ready_vote(text):
    if not text:
        return "UNCLEAR"
    upper = text.upper()
    first_line = upper.split("\n")[0].strip()
    if "NOT READY" in first_line:
        return "NOT_READY"
    if "READY" in first_line and "NOT READY" not in first_line:
        return "READY"
    head = upper[:200]
    if "NOT READY" in head:
        return "NOT_READY"
    if "READY" in head:
        return "READY"
    return "UNCLEAR"


def run_deliberation(
    api_key, agents, scenario_key, normgen,
    rounds=5, out_path=None, verbose=True,
    enable_ready_check=True, ready_check_max_tokens=150,
    self_reflection=False,
    no_consensus_outcome=False,
    turns_per_agent_per_round=1,
):
    # turns_per_agent_per_round: how many times each agent speaks per deliberation
    # round. Default 1 reproduces the original one-turn-per-agent behavior exactly.
    # For a 2-agent dyad, set 3 to get A B A B A B per round (6 turns/round), which
    # matches the 6-agent panel's 6 turns/round and holds deliberation volume constant.
    n_agents = len(agents)

    def leadoff_for_round(r):
        return r % n_agents

    def turn_order_for_round(r):
        start = leadoff_for_round(r)
        return agents[start:] + agents[:start]

    scenario_prompt = build_scenario_prompt(scenario_key, normgen)
    outcome_instruction = build_outcome_instruction(scenario_key, normgen, no_consensus_outcome=no_consensus_outcome)

    transcript = []
    turn_counter = 0
    ready_check_log = []
    stopped_early_after_round = None

    def build_result_so_far():
        return {
            "run_metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_key": scenario_key,
                "normgen": normgen,
                "rounds": rounds,
                "turns_per_agent_per_round": turns_per_agent_per_round,
                "leadoff_rotation": "round_index mod n_agents",
                "leadoff_per_round": [agents[leadoff_for_round(r)].agent_id for r in range(rounds + 1)],
                "ready_check_enabled": enable_ready_check,
                "self_reflection": self_reflection,
                "no_consensus_outcome": no_consensus_outcome,
                "ready_check_after_rounds": sorted(READY_CHECK_AFTER_ROUNDS),
                "ready_check_log": ready_check_log,
                "stopped_early_after_round": stopped_early_after_round,
                "agents": [{"agent_id": a.agent_id, "display_name": a.display_name,
                            "model": a.model, "temperature": a.temperature,
                            "max_tokens": a.max_tokens} for a in agents],
                "scenario_prompt": scenario_prompt,
                "outcome_instruction": outcome_instruction,
                "complete": False,
            },
            "transcript": [asdict(t) for t in transcript],
        }

    def save_partial():
        if out_path is None:
            return
        try:
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(build_result_so_far(), indent=2))
            tmp.replace(out_path)
        except Exception as e:
            print(f"  (warning: incremental save failed: {e})")

    def run_one_turn(agent, round_index, is_outcome, is_ready_check=False, ready_check_text=None):
        nonlocal turn_counter
        messages = build_messages(
            agent=agent, transcript=transcript, scenario_prompt=scenario_prompt,
            current_round_index=round_index, total_deliberation_rounds=rounds,
            is_outcome_round=is_outcome, is_ready_check=is_ready_check,
            outcome_instruction=outcome_instruction if is_outcome else None,
            ready_check_instruction=ready_check_text if is_ready_check else None,
        )
        try:
            text, usage, elapsed = _one_call(
                api_key=api_key, agent=agent, messages=messages,
                max_tokens_override=(ready_check_max_tokens * 5 if self_reflection else ready_check_max_tokens) if is_ready_check else None,
            )
        except Exception as e:
            text = f"[API error during this turn: {e}]"
            usage = {}
            elapsed = None
        turn_counter += 1
        if is_outcome:
            turn_type = "outcome"
        elif is_ready_check:
            turn_type = "ready_check"
        else:
            turn_type = "deliberation"
        t = Turn(
            round_index=round_index, turn_index=turn_counter,
            agent_id=agent.agent_id, display_name=agent.display_name,
            is_outcome_round=is_outcome,
            text=text if isinstance(text, str) else "[non-string response]",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            elapsed_seconds=elapsed,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            turn_type=turn_type,
        )
        transcript.append(t)
        save_partial()
        if verbose:
            preview = (t.text or "").replace("\n", " ")[:160]
            print(f"  [{agent.display_name}] {preview}...")
        return t

    def run_ready_check(after_round_1_based):
        next_round_0_based = after_round_1_based
        order = turn_order_for_round(next_round_0_based)
        if self_reflection:
            check_text = READY_CHECK_INSTRUCTION_WITH_REFLECTION.format(
                next_round=after_round_1_based + 1)
        else:
            check_text = READY_CHECK_INSTRUCTION.format(
                next_round=after_round_1_based + 1)
        if verbose:
            print(f"\n--- Ready-check after Round {after_round_1_based} (leadoff: {order[0].display_name}) ---")
        votes = []
        for agent in order:
            t = run_one_turn(agent, round_index=after_round_1_based - 1,
                             is_outcome=False, is_ready_check=True, ready_check_text=check_text)
            vote = _parse_ready_vote(t.text)
            votes.append({"agent_id": agent.agent_id, "display_name": agent.display_name,
                          "vote": vote, "text": t.text})
            if verbose:
                print(f"    -> {agent.display_name} vote: {vote}")
        unanimous_ready = all(v["vote"] == "READY" for v in votes)
        ready_check_log.append({"after_round": after_round_1_based, "votes": votes,
                                "unanimous_ready": unanimous_ready})
        if verbose:
            tally = sum(1 for v in votes if v["vote"] == "READY")
            print(f"    Tally: {tally} of {len(votes)} READY. Unanimous: {unanimous_ready}")
        return unanimous_ready

    for r in range(rounds):
        order = turn_order_for_round(r)
        if verbose:
            print(f"\n=== Round {r + 1} of {rounds} (leadoff: {order[0].display_name}) ===")
        for _rep in range(turns_per_agent_per_round):
            for agent in order:
                run_one_turn(agent, round_index=r, is_outcome=False)
        completed_round_1_based = r + 1
        if enable_ready_check and completed_round_1_based in READY_CHECK_AFTER_ROUNDS:
            unanimous = run_ready_check(completed_round_1_based)
            if unanimous:
                stopped_early_after_round = completed_round_1_based
                if verbose:
                    print(f"\n=== Early stop: unanimous READY after Round "
                          f"{completed_round_1_based}. Skipping to outcome round. ===")
                break

    outcome_order = turn_order_for_round(rounds)
    if verbose:
        print(f"\n=== Outcome round (leadoff: {outcome_order[0].display_name}) ===")
    for agent in outcome_order:
        run_one_turn(agent, round_index=rounds, is_outcome=True)

    final_result = build_result_so_far()
    final_result["run_metadata"]["complete"] = True
    if out_path is not None:
        try:
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(final_result, indent=2))
            tmp.replace(out_path)
        except Exception as e:
            print(f"  (warning: final save failed: {e})")
    return final_result
