"""
LLM Prompts for Two-Round Chemical Synthesis Protocol Parsing
=============================================================
Converts a chemical synthesis protocol into a two-level hierarchical DAG:

  Round 1 → Task DAG        (human-readable + machine-targeted nodes, material-flow edges)
  Round 2 → Atomic-op DAG   (XDL-style atomic actions, equipment labels, per task node)

Two Round 1 input paths:
  build_round1_prompt_from_text() -- parses raw protocol PROSE directly; the
      LLM identifies operations, materials, quantities, and conditions itself
      from free text. This is what the demo notebook uses.
  build_round1_prompt()           -- parses a pre-extracted structured record
      (operations/precursors/solvents/quantities/type) from the solution-based
      inorganic materials dataset. Kept for that dataset, whose paragraph_string
      field is copyright-truncated (~50 chars + "<...>" + ~50 chars) and so
      isn't usable for text parsing.

Model recommendation : claude-sonnet-4-5 or better
Temperature          : 0.1  (deterministic structured output)
Response format      : JSON (use Anthropic structured-output / json_mode)
"""

import json

# ---------------------------------------------------------------------------
# Shared equipment vocabulary (loaded from taxonomy file at runtime)
# ---------------------------------------------------------------------------
EQUIPMENT_IDS = [
    "MAGNETIC_STIRRER_HOTPLATE",
    "OVERHEAD_STIRRER",
    "ULTRASONIC_BATH_PROBE",
    "AUTOCLAVE",
    "MUFFLE_FURNACE",
    "TUBE_FURNACE",
    "COOLING_BATH",
    "CENTRIFUGE",
    "VACUUM_FILTRATION",
    "DRYING_OVEN",
    "ROTARY_EVAPORATOR",
    "SPRAY_DRYER",
    "BALL_MILL",
    "LIQUID_HANDLING_ROBOT",
    "ANALYTICAL_BALANCE",
    "PH_METER_CONTROLLER",
    "REFLUX_CONDENSER_SETUP",
    "FREEZE_DRYER",
    "GAS_SUPPLY_MANIFOLD",
]

XDL_ACTIONS = [
    "Add", "Transfer", "Separate",
    "Stir", "StartStir", "StopStir",
    "HeatChill", "HeatChillToTemp", "StartHeatChill", "StopHeatChill",
    "Purge", "EvacuateAndRefill",
    "Filter", "FilterThrough", "WashSolid",
    "Dissolve", "Dry", "Evaporate", "Crystallize", "Precipitate",
    "Wait", "CleanVessel",
]


# ===========================================================================
# ROUND 1 PROMPT — Protocol structured record → Task DAG
# ===========================================================================

ROUND1_SYSTEM = """\
You are an expert in automated chemical synthesis and robotic lab scheduling.
Your task is to parse a structured record of a chemical synthesis protocol and
produce a Task DAG (Directed Acyclic Graph).

## Core rule: edges encode MATERIAL FLOW, not textual order
Add a directed edge from task A to task B ONLY when B physically requires the
output material of A as one of its inputs. Two tasks that each prepare a
separate solution independently — even if listed sequentially in the source —
are PARALLEL nodes with NO edge between them. They only converge at the task
that mixes both solutions together.

## Node design (hybrid human + machine)
Each node carries two representations simultaneously:
  - label          : Short, human-readable phrase (≤10 words), e.g.
                     "Dissolve CoCl₂ in ethylene glycol"
  - description    : One sentence, imperative, machine-targeted, precise:
                     action + materials + key conditions, e.g.
                     "Dissolve 1.2 g CoCl₂ in 40 mL ethylene glycol with
                      magnetic stirring at room temperature."
  - action_verb    : Single lowercase verb, e.g. "dissolve", "heat", "filter"
  - inputs         : List of material objects consumed by this task
  - output         : Single intermediate or final material produced
  - operation_types: One or more of the dataset's operation type labels
  - equipment_hints: 1–3 equipment IDs from the controlled vocabulary

## Output nodes are the DAG's "variable names"
Name intermediate outputs clearly and consistently, e.g. "CoCl2_EG_solution",
"FeCl3_EG_solution", "mixed_precursor_suspension". Edge material fields must
reference these exact names.

## Parallelism detection checklist
Before writing edges, ask for each pair of nodes:
  1. Does node B's inputs list include node A's output? → YES = add edge
  2. Are they preparing different starting materials from reagents? → NO edge
  3. Does B operate on a product that only exists after A completes? → YES = edge
  4. Could a robot start B before A is finished without affecting the result? → NO edge

## Reasoning field
Always include a "reasoning" field BEFORE the nodes and edges, explaining
which tasks you identified as parallel and why. This is the chain-of-thought.
"""

ROUND1_USER_TEMPLATE = """\
Parse the following chemical synthesis record into a Task DAG.

### Input record
```json
{record_json}
```

### Controlled equipment vocabulary (use these IDs only)
{equipment_ids}

### Output JSON schema
Return ONLY a valid JSON object matching this exact schema — no prose, no
markdown fences:

{{
  "protocol_id":     "<doi or generated id>",
  "target_material": "<targets_string>",
  "synthesis_type":  "<type field>",
  "reasoning": "<explain parallel vs sequential decisions before listing nodes>",
  "nodes": [
    {{
      "id":              "T1",
      "label":           "<short human-readable title>",
      "description":     "<one precise imperative sentence>",
      "action_verb":     "<single lowercase verb>",
      "inputs":  [ {{"material": "<name>", "amount": "<amount or null>"}} ],
      "output":  {{"material": "<intermediate_name>", "state": "<solid|liquid|slurry|gas>"}},
      "operation_types": ["<MixingOperation|HeatingOperation|...>"],
      "equipment_hints": ["<EQUIPMENT_ID>"]
    }}
  ],
  "edges": [
    {{
      "from":              "<node_id>",
      "to":                "<node_id>",
      "material":          "<output material name that flows from→to>",
      "dependency_reason": "<one sentence explaining why B needs A's output>"
    }}
  ]
}}
"""


ROUND1_SYSTEM_TEXT = """\
You are an expert in automated chemical synthesis and robotic lab scheduling.
Your task is to read a chemical synthesis protocol written as free-text prose
(an experimental section / lab procedure) and produce a Task DAG (Directed
Acyclic Graph).

## Step 1 — read the protocol, don't wait for pre-extracted fields
Nothing has been extracted for you: no operation list, no conditions table.
Identify every discrete action described in the prose (mix, dissolve, heat,
cool, filter, wash, dry, etc.), along with the materials, quantities, and
conditions (temperature, time, atmosphere) attached to each action, directly
from the text.

## Step 2 — edges encode MATERIAL FLOW, not textual order
Add a directed edge from task A to task B ONLY when B physically requires the
output material of A as one of its inputs. Two tasks that each prepare a
separate solution independently — even if described sequentially in the
prose — are PARALLEL nodes with NO edge between them. They only converge at
the task that combines both.

## Node design (hybrid human + machine)
Each node carries two representations simultaneously:
  - label          : Short, human-readable phrase (≤10 words), e.g.
                     "Dissolve CoCl₂ in ethylene glycol"
  - description    : One sentence, imperative, machine-targeted, precise:
                     action + materials + key conditions, e.g.
                     "Dissolve 1.2 g CoCl₂ in 40 mL ethylene glycol with
                      magnetic stirring at room temperature."
  - action_verb    : Single lowercase verb, e.g. "dissolve", "heat", "filter"
  - inputs         : List of material objects consumed by this task
  - output         : Single intermediate or final material produced
  - operation_types: One or more of: MixingOperation, HeatingOperation,
                     CoolingOperation, PurificationOperation, DryingOperation,
                     ShapingOperation, CharacterizationOperation — chosen by
                     you, from what the prose describes, not given to you.
  - equipment_hints: 1–3 equipment IDs from the controlled vocabulary

## Output nodes are the DAG's "variable names"
Name intermediate outputs clearly and consistently, e.g. "amine_HCl_salt",
"free_base_in_pentane", "lidocaine_bisulfate". Edge material fields must
reference these exact names.

## Parallelism detection checklist
Before writing edges, ask for each pair of nodes:
  1. Does node B's inputs list include node A's output? → YES = add edge
  2. Are they preparing different starting materials from reagents? → NO edge
  3. Does B operate on a product that only exists after A completes? → YES = edge
  4. Could a robot start B before A is finished without affecting the result? → NO edge

## Reasoning field
Always include a "reasoning" field BEFORE the nodes and edges. Use it for
two things: (1) which operations you identified in the prose and how you
segmented them into tasks, and (2) which tasks you identified as parallel
vs. sequential and why.
"""

ROUND1_USER_TEMPLATE_TEXT = """\
Parse the following chemical synthesis protocol into a Task DAG.

### Protocol metadata
- Source          : {source}
- Target material : {target}
- Synthesis type  : {synthesis_type}

### Protocol text
```
{protocol_text}
```

### Controlled equipment vocabulary (use these IDs only)
{equipment_ids}

### Output JSON schema
Return ONLY a valid JSON object matching this exact schema — no prose, no
markdown fences:

{{
  "protocol_id":     "<source id>",
  "target_material": "<target material>",
  "synthesis_type":  "<synthesis type>",
  "reasoning": "<what operations you found in the prose, and explain parallel vs sequential decisions before listing nodes>",
  "nodes": [
    {{
      "id":              "T1",
      "label":           "<short human-readable title>",
      "description":     "<one precise imperative sentence>",
      "action_verb":     "<single lowercase verb>",
      "inputs":  [ {{"material": "<name>", "amount": "<amount or null>"}} ],
      "output":  {{"material": "<intermediate_name>", "state": "<solid|liquid|slurry|gas>"}},
      "operation_types": ["<MixingOperation|HeatingOperation|...>"],
      "equipment_hints": ["<EQUIPMENT_ID>"]
    }}
  ],
  "edges": [
    {{
      "from":              "<node_id>",
      "to":                "<node_id>",
      "material":          "<output material name that flows from→to>",
      "dependency_reason": "<one sentence explaining why B needs A's output>"
    }}
  ]
}}
"""


def build_round1_prompt_from_text(
    protocol_text: str,
    target: str = "",
    synthesis_type: str = "",
    source: str = "",
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for Round 1, parsing directly from
    free-text protocol prose instead of a pre-extracted structured record.

    protocol_text  : the raw experimental-procedure text (no pre-extracted
                      operations/conditions — the LLM identifies these itself)
    target         : target material name (metadata, not parsed from prose)
    synthesis_type : e.g. "hydrothermal" (metadata, not parsed from prose)
    source         : citation / provenance string, used only as protocol_id
    """
    user = ROUND1_USER_TEMPLATE_TEXT.format(
        source=source or "unspecified",
        target=target or "unspecified",
        synthesis_type=synthesis_type or "unspecified",
        protocol_text=protocol_text.strip(),
        equipment_ids=json.dumps(EQUIPMENT_IDS, indent=2),
    )
    return ROUND1_SYSTEM_TEXT, user


def build_round1_prompt(record: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for Round 1.
    record: one item from solutionsynthesis_dataset_202185.json
    """
    # Slim the record to only what the LLM needs (no redundant fields)
    slim = {
        "doi":             record.get("doi"),
        "target":          record.get("targets_string"),
        "synthesis_type":  record.get("type"),
        "reaction":        record.get("reaction_string"),
        "precursors":      [
            {
                "material": p["material_string"],
                "formula":  p["material_formula"],
            }
            for p in record.get("precursors", [])
        ],
        "solvents":        record.get("solvents_string", []),
        "quantities":      record.get("quantities", []),
        "operations":      [
            {
                "type":       op["type"],
                "verb":       op["string"],
                "conditions": {
                    k: v for k, v in op.get("conditions", {}).items()
                    if v is not None
                },
            }
            for op in record.get("operations", [])
        ],
    }

    user = ROUND1_USER_TEMPLATE.format(
        record_json=json.dumps(slim, indent=2),
        equipment_ids=json.dumps(EQUIPMENT_IDS, indent=2),
    )
    return ROUND1_SYSTEM, user


# ===========================================================================
# ROUND 2 PROMPT — Single task node → Atomic-op sub-DAG
# ===========================================================================

ROUND2_SYSTEM = """\
You are an expert in robotic laboratory automation and chemical synthesis.
Your task is to decompose a single synthesis task node into an atomic-operation
sub-DAG, where each node is the smallest physically meaningful robot action.

## What "atomic" means
An atomic operation cannot be split further without losing physical meaning.
Examples:
  - "Dissolve CoCl₂ and stir for 15 min" → TWO atomic ops:
      1. Add(CoCl₂, vessel)        — uses ANALYTICAL_BALANCE + LIQUID_HANDLING_ROBOT
      2. Stir(400 rpm, 15 min)     — uses MAGNETIC_STIRRER_HOTPLATE
  - "Heat to 180 °C for 12 h" → TWO atomic ops:
      1. HeatChillToTemp(180 °C)   — uses AUTOCLAVE or MUFFLE_FURNACE
      2. Wait(12 h)                — no equipment (time-keeping only)
  - "Wash with ethanol × 3, centrifuge" → FOUR atomic ops:
      1. Add(ethanol, vessel)
      2. Stir(brief)
      3. Filter/Separate (centrifuge)
      4. Repeat if needed (can be modelled as 3× of ops 1–3)

## Parallelism inside a task
Within one task, atomic ops are usually sequential. However, if two ops truly
do not share a vessel or intermediate output, mark them as parallel (no
depends_on edge between them).

## XDL actions (use these exact strings)
{xdl_actions}

## Equipment IDs (use these exact strings)
{equipment_ids}

## Duration estimation
Provide a best-estimate duration for each atomic op. Use known heuristics:
  - Weighing on balance        : 2 min
  - Liquid transfer (robot)    : 1 min per transfer
  - Stirring/dissolution       : 5–30 min
  - Heating to temp            : 10–60 min (depends on ramp rate)
  - Isothermal hold            : as specified in conditions
  - Centrifugation             : 10–20 min
  - Washing cycle              : 15 min per wash
  - Drying in oven             : 1–24 h
"""

ROUND2_USER_TEMPLATE = """\
Decompose the following synthesis task into atomic operations.

### Parent protocol context
- Target material : {target}
- Synthesis type  : {synthesis_type}
- Full task list  : {all_task_labels}

### Task node to decompose
```json
{task_node_json}
```

### Output JSON schema
Return ONLY a valid JSON object — no prose, no markdown fences:

{{
  "task_id":   "<same id as input node>",
  "reasoning": "<explain how you split this task into atomic ops and any parallelism>",
  "atomic_ops": [
    {{
      "id":         "<parent_task_id>_A<n>",
      "xdl_action": "<XDL action string>",
      "label":      "<short human-readable description of this atomic step>",
      "equipment":  "<EQUIPMENT_ID or null if no equipment needed>",
      "parameters": {{
        "<param_name>": "<value with unit>"
      }},
      "duration_estimate": {{
        "value": <number>,
        "unit":  "<min|h|s>"
      }},
      "depends_on": ["<atomic_op_id>", "..."]
    }}
  ]
}}
"""


def build_round2_prompt(
    task_node: dict,
    target: str,
    synthesis_type: str,
    all_nodes: list[dict],
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for Round 2.
    task_node   : one node dict from the Round 1 output
    target      : target material name
    synthesis_type : e.g. "hydrothermal"
    all_nodes   : full list of Round 1 nodes (for context)
    """
    system = ROUND2_SYSTEM.format(
        xdl_actions=json.dumps(XDL_ACTIONS, indent=2),
        equipment_ids=json.dumps(EQUIPMENT_IDS, indent=2),
    )

    all_labels = [f"{n['id']}: {n['label']}" for n in all_nodes]

    user = ROUND2_USER_TEMPLATE.format(
        target=target,
        synthesis_type=synthesis_type,
        all_task_labels=json.dumps(all_labels, indent=2),
        task_node_json=json.dumps(task_node, indent=2),
    )
    return system, user


# ===========================================================================
# OUTPUT SCHEMAS (for validation / documentation)
# ===========================================================================

ROUND1_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["protocol_id", "target_material", "synthesis_type",
                 "reasoning", "nodes", "edges"],
    "properties": {
        "protocol_id":     {"type": "string"},
        "target_material": {"type": "string"},
        "synthesis_type":  {"type": "string"},
        "reasoning":       {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "description", "action_verb",
                             "inputs", "output", "operation_types", "equipment_hints"],
                "properties": {
                    "id":              {"type": "string"},
                    "label":           {"type": "string"},
                    "description":     {"type": "string"},
                    "action_verb":     {"type": "string"},
                    "inputs":          {"type": "array"},
                    "output":          {"type": "object"},
                    "operation_types": {"type": "array", "items": {"type": "string"}},
                    "equipment_hints": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "material", "dependency_reason"],
                "properties": {
                    "from":              {"type": "string"},
                    "to":                {"type": "string"},
                    "material":          {"type": "string"},
                    "dependency_reason": {"type": "string"},
                },
            },
        },
    },
}

ROUND2_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["task_id", "reasoning", "atomic_ops"],
    "properties": {
        "task_id":   {"type": "string"},
        "reasoning": {"type": "string"},
        "atomic_ops": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "xdl_action", "label", "equipment",
                             "parameters", "duration_estimate", "depends_on"],
                "properties": {
                    "id":         {"type": "string"},
                    "xdl_action": {"type": "string",
                                   "enum": XDL_ACTIONS},
                    "label":      {"type": "string"},
                    "equipment":  {"type": ["string", "null"]},
                    "parameters": {"type": "object"},
                    "duration_estimate": {
                        "type": "object",
                        "required": ["value", "unit"],
                        "properties": {
                            "value": {"type": "number"},
                            "unit":  {"type": "string", "enum": ["s", "min", "h"]},
                        },
                    },
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


# ===========================================================================
# Quick self-test (python llm_prompts.py)
# ===========================================================================
if __name__ == "__main__":
    # Minimal synthetic record to test prompt construction
    sample_record = {
        "doi": "10.1063/1.4967923",
        "targets_string": "CoFe2O4",
        "type": "hydrothermal",
        "reaction_string": "CoCl2 + FeCl3 -> CoFe2O4",
        "precursors": [
            {"material_string": "CoCl2", "material_formula": "CoCl2"},
            {"material_string": "FeCl3", "material_formula": "FeCl3"},
        ],
        "solvents_string": ["ethylene glycol", "ethanol"],
        "quantities": [
            {"material": "ethylene glycol", "quantity": [{"number": 40, "unit": "mL"}]},
        ],
        "operations": [
            {"type": "MixingOperation",     "string": "dissolved",
             "conditions": {"mixing_media": ["ethylene glycol"]}},
            {"type": "MixingOperation",     "string": "stirred",    "conditions": {}},
            {"type": "MixingOperation",     "string": "added",      "conditions": {}},
            {"type": "HeatingOperation",    "string": "heated",
             "conditions": {"temperature": {"values": [180.0], "units": "°C"},
                            "time":        {"values": [12.0],  "units": "h"}}},
            {"type": "CoolingOperation",    "string": "cooling",
             "conditions": {"temperature": {"values": [25.0], "units": "°C"}}},
            {"type": "PurificationOperation","string": "washed",
             "conditions": {"mixing_media": ["ethanol"]}},
            {"type": "PurificationOperation","string": "centrifugation", "conditions": {}},
            {"type": "DryingOperation",     "string": "dried",
             "conditions": {"temperature": {"values": [60.0], "units": "°C"},
                            "time":        {"values": [2.0],  "units": "h"}}},
        ],
    }

    sys1, usr1 = build_round1_prompt(sample_record)
    print("=== ROUND 1 SYSTEM PROMPT ===")
    print(sys1)
    print("\n=== ROUND 1 USER PROMPT ===")
    print(usr1)

    # Simulate a Round 1 node for Round 2 test
    sample_task_node = {
        "id": "T1",
        "label": "Dissolve CoCl₂ in ethylene glycol",
        "description": "Dissolve CoCl₂ in 40 mL ethylene glycol with magnetic stirring at room temperature.",
        "action_verb": "dissolve",
        "inputs": [
            {"material": "CoCl2",          "amount": None},
            {"material": "ethylene_glycol", "amount": "40 mL"},
        ],
        "output": {"material": "CoCl2_EG_solution", "state": "liquid"},
        "operation_types": ["MixingOperation"],
        "equipment_hints": ["MAGNETIC_STIRRER_HOTPLATE", "LIQUID_HANDLING_ROBOT"],
    }
    all_nodes = [
        sample_task_node,
        {"id": "T2", "label": "Dissolve FeCl₃ in ethylene glycol"},
        {"id": "T3", "label": "Mix precursor solutions and heat in autoclave"},
    ]

    sys2, usr2 = build_round2_prompt(
        sample_task_node,
        target="CoFe2O4",
        synthesis_type="hydrothermal",
        all_nodes=all_nodes,
    )
    print("\n\n=== ROUND 2 SYSTEM PROMPT ===")
    print(sys2)
    print("\n=== ROUND 2 USER PROMPT ===")
    print(usr2)
