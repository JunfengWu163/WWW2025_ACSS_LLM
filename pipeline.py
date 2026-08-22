"""
ACSS Protocol Parsing Pipeline
================================
Calls the Claude API to run two-round LLM parsing of chemical synthesis
protocols into hierarchical DAGs.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    from pipeline import parse_protocol

The pipeline is intentionally stateless — call parse_protocol() once per
record; it returns the full hierarchical DAG as a plain Python dict.
"""

import os
import json
import time
import jsonschema
import anthropic
from dotenv import load_dotenv

from llm_prompts import (
    build_round1_prompt,
    build_round2_prompt,
    ROUND1_OUTPUT_SCHEMA,
    ROUND2_OUTPUT_SCHEMA,
)

# Loads ACSS-WWW2025/.env if present (see .env.template). A shell-exported
# ANTHROPIC_API_KEY still wins if both are set. Needed because a Jupyter
# kernel launched via VS Code / a Jupyter Desktop app doesn't always inherit
# ~/.zshenv or similar shell exports — this reads a file directly instead.
load_dotenv()

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Run:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Core LLM call — with retry and JSON extraction
# ---------------------------------------------------------------------------
MODEL   = "claude-sonnet-4-5"
MAX_TOKENS = 8192

def _call_llm(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    schema: dict,
    retries: int = 3,
    label: str = "",
) -> dict:
    """
    Call the Claude API, extract JSON from the response, and validate
    against the provided JSON schema. Retries on parse/validation failure.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if the model added them anyway
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            jsonschema.validate(parsed, schema)
            return parsed

        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            last_err = e
            print(f"  [attempt {attempt}/{retries}] {label} — parse/validation error: {e}")
            if attempt < retries:
                time.sleep(1)

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error during {label}: {e}") from e

    raise RuntimeError(
        f"Failed to get valid JSON from LLM after {retries} attempts "
        f"for {label}. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Round 1: record → Task DAG
# ---------------------------------------------------------------------------
def run_round1(record: dict, verbose: bool = True) -> dict:
    """
    Parse a dataset record into a Task DAG.
    Returns the validated Round 1 dict.
    """
    client = _get_client()
    system, user = build_round1_prompt(record)

    if verbose:
        print(f"[Round 1] Parsing protocol → Task DAG  "
              f"(target: {record.get('targets_string')}, "
              f"type: {record.get('type')}, "
              f"steps: {len(record.get('operations', []))})")

    result = _call_llm(client, system, user, ROUND1_OUTPUT_SCHEMA, label="Round 1")

    if verbose:
        print(f"  ✓ {len(result['nodes'])} task nodes, "
              f"{len(result['edges'])} dependency edges")
        # Show parallel branches (nodes with no incoming edge)
        targets = {e["to"] for e in result["edges"]}
        roots = [n for n in result["nodes"] if n["id"] not in targets]
        if len(roots) > 1:
            print(f"  ✓ Detected {len(roots)} parallel root tasks: "
                  f"{[r['id'] + ':' + r['label'][:30] for r in roots]}")

    return result


# ---------------------------------------------------------------------------
# Round 2: each Task node → atomic-op sub-DAG
# ---------------------------------------------------------------------------
def run_round2(
    round1_result: dict,
    verbose: bool = True,
) -> dict:
    """
    For every task node in the Round 1 DAG, call the LLM to decompose it
    into atomic operations.

    Returns an enriched dict with each node's 'atomic_ops' sub-DAG attached.
    """
    client  = _get_client()
    nodes   = round1_result["nodes"]
    target  = round1_result["target_material"]
    syntype = round1_result["synthesis_type"]

    enriched_nodes = []
    total_atomic   = 0

    for i, node in enumerate(nodes, 1):
        if verbose:
            print(f"[Round 2] Node {i}/{len(nodes)}: {node['id']} — {node['label']}")

        system, user = build_round2_prompt(
            task_node      = node,
            target         = target,
            synthesis_type = syntype,
            all_nodes      = nodes,
        )
        r2 = _call_llm(
            client, system, user, ROUND2_OUTPUT_SCHEMA,
            label=f"Round 2 / {node['id']}"
        )

        enriched = dict(node)
        enriched["atomic_ops"] = r2["atomic_ops"]
        enriched["round2_reasoning"] = r2["reasoning"]
        enriched_nodes.append(enriched)
        total_atomic += len(r2["atomic_ops"])

        if verbose:
            print(f"  ✓ {len(r2['atomic_ops'])} atomic ops")

    result = dict(round1_result)
    result["nodes"] = enriched_nodes

    if verbose:
        print(f"\n[Done] Hierarchical DAG: "
              f"{len(enriched_nodes)} tasks × avg "
              f"{total_atomic / len(enriched_nodes):.1f} atomic ops each "
              f"= {total_atomic} total atomic ops")

    return result


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------
def parse_protocol(
    record: dict,
    verbose: bool = True,
) -> dict:
    """
    Full two-round parse of a synthesis record.

    Parameters
    ----------
    record  : one item from solutionsynthesis_dataset_202185.json
    verbose : print progress to stdout

    Returns
    -------
    hierarchical_dag : dict with keys:
        protocol_id, target_material, synthesis_type, reasoning,
        nodes (each with atomic_ops), edges
    """
    if verbose:
        print("=" * 60)
        print(f"Parsing: {record.get('targets_string')}  [{record.get('type')}]")
        print("=" * 60)

    r1 = run_round1(record, verbose=verbose)
    r2 = run_round2(r1,     verbose=verbose)

    return r2


# ---------------------------------------------------------------------------
# Equipment colour map (used by visualisation)
# ---------------------------------------------------------------------------
EQUIPMENT_COLORS = {
    "MAGNETIC_STIRRER_HOTPLATE":  "#4C9BE8",   # blue
    "OVERHEAD_STIRRER":           "#5BA4CF",   # light blue
    "ULTRASONIC_BATH_PROBE":      "#7EC8E3",   # sky
    "AUTOCLAVE":                  "#E07B39",   # orange
    "MUFFLE_FURNACE":             "#D94F3D",   # red
    "TUBE_FURNACE":               "#C0392B",   # dark red
    "COOLING_BATH":               "#82C4A0",   # green
    "CENTRIFUGE":                 "#9B59B6",   # purple
    "VACUUM_FILTRATION":          "#8E44AD",   # dark purple
    "DRYING_OVEN":                "#F1C40F",   # yellow
    "ROTARY_EVAPORATOR":          "#F39C12",   # amber
    "SPRAY_DRYER":                "#E67E22",   # dark amber
    "BALL_MILL":                  "#95A5A6",   # grey
    "LIQUID_HANDLING_ROBOT":      "#2ECC71",   # bright green
    "ANALYTICAL_BALANCE":         "#1ABC9C",   # teal
    "PH_METER_CONTROLLER":        "#16A085",   # dark teal
    "REFLUX_CONDENSER_SETUP":     "#3498DB",   # medium blue
    "FREEZE_DRYER":               "#BDC3C7",   # light grey
    "GAS_SUPPLY_MANIFOLD":        "#E74C3C",   # bright red
    None:                         "#ECF0F1",   # near-white (no equipment)
}

CATEGORY_COLORS = {
    "MixingOperation":           "#4C9BE8",
    "HeatingOperation":          "#D94F3D",
    "CoolingOperation":          "#82C4A0",
    "PurificationOperation":     "#9B59B6",
    "DryingOperation":           "#F1C40F",
    "ShapingOperation":          "#95A5A6",
    "CharacterizationOperation": "#1ABC9C",
}


# ---------------------------------------------------------------------------
# Quick smoke test (python pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib, sys

    data_path = pathlib.Path(__file__).parent / "solutionsynthesis_dataset_202185.json"
    demo_path = pathlib.Path(__file__).parent / "demo_protocols.json"

    with open(demo_path) as f:
        demos = json.load(f)

    # Parse just DEMO_2 (CoFe2O4, simplest) as a smoke test
    demo = next(d for d in demos if d["demo_id"] == "DEMO_2")
    record = demo["record"]

    result = parse_protocol(record, verbose=True)

    out = pathlib.Path(__file__).parent / "dag_CoFe2O4.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
