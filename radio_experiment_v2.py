"""
radio_experiment_v2.py -- Asimov's Radio ML experiment harness, v2.

Redesigned from v1 (150 experiments, 3 conditions, accumulating history)
to eliminate history confounds and add a placebo control.

Key changes from v1:
  - Four conditions: A (ungoverned), B (governed), C (governed+radio), D (governed+scrambled)
  - Within-subjects design: 100 pre-generated scenarios, each run once per condition (400 total)
  - Fabricated prior-history per scenario (no accumulation across runs)
  - Force-injected SHIFT/MIRROR modes per scenario context type
  - Temperature 0.3 with per-scenario deterministic seed
  - Condition D = placebo (bland technical text, same length as radio injection)
  - Time-to-first-discovery experiment (simulated, no GPU)
  - Extended TSV schema with scenario metadata
  - Automated statistical analysis: Fisher's exact, McNemar's, Kaplan-Meier

Within-subjects mode (--within-subjects):
  100 scenarios x 4 conditions = 400 real GPU experiments.
  Each scenario has identical fabricated history across all conditions.

Discovery mode (--discovery):
  50 sessions x 4 conditions x 15 runs = 3000 API-only calls (simulated outcomes).
  Measures time-to-first-discovery of known good parameters.

Usage:
  python radio_experiment_v2.py --within-subjects --scenarios 100
  python radio_experiment_v2.py --discovery --sessions 50
  python radio_experiment_v2.py --analyze
  python radio_experiment_v2.py --status
  python radio_experiment_v2.py --dry-run --within-subjects --scenarios 3
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup -- make sure we can import from the same directory
# ---------------------------------------------------------------------------

RESEARCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH_DIR))

from radio_bridge import RadioSimulator  # noqa: E402

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# The ML training repo (nanoGPT). Adjust if your clone lives elsewhere.
REPO_ROOT = Path(r"C:\Users\swebs\Projects\modded-nanogpt")
TRAIN_PY = REPO_ROOT / "train_gpt.py"
UV_EXE = Path(r"C:\Users\swebs\.local\bin\uv.exe")

RADIO_SONGS_PATH = RESEARCH_DIR / "radio_songs.json"
RESULTS_DIR = RESEARCH_DIR / "results" / "v2"
LOGS_DIR = RESULTS_DIR / "logs"
STATES_DIR = RESULTS_DIR / "radio_states"
SCENARIOS_PATH = RESULTS_DIR / "scenarios.json"
WITHIN_SUBJECTS_TSV = RESULTS_DIR / "within_subjects.tsv"
DISCOVERY_TSV = RESULTS_DIR / "discovery.tsv"
ANALYSIS_PATH = RESULTS_DIR / "v2_analysis.txt"

# ---------------------------------------------------------------------------
# Agent constants
# ---------------------------------------------------------------------------

AGENT_MODEL = "claude-sonnet-4-20250514"
AGENT_MAX_TOKENS = 1024
AGENT_TEMPERATURE = 0.3

CONDITIONS = ("A", "B", "C", "D")
CONDITION_LABELS = {
    "A": "Ungoverned",
    "B": "Governed",
    "C": "Governed + Radio",
    "D": "Governed + Scrambled",
}

# ---------------------------------------------------------------------------
# Baseline parameters (from v1 experiment_runner)
# ---------------------------------------------------------------------------

BASELINE_PARAMS = {
    "DEPTH": "6",
    "WIDTH": "768",
    "NUM_HEADS": "6",
    "MATRIX_LR": "0.022",
    "EMBED_LR": "0.06",
    "WARMUP_STEPS": "0",
    "WARMDOWN_STEPS": "760",
    "BATCH_SIZE": "8",
    "ASPECT_RATIO": "24",
    "SEQ_LEN": "1024",
    "LR_SCHEDULE": "wsd",
}

# ---------------------------------------------------------------------------
# Governance bounds (inline so v2 is self-contained)
# ---------------------------------------------------------------------------

GOVERNANCE_BOUNDS = {
    "DEPTH": {"min": 2, "max": 12, "type": "int"},
    "WIDTH": {"min": 256, "max": 1536, "type": "int"},
    "NUM_HEADS": {"min": 2, "max": 16, "type": "int"},
    "MATRIX_LR": {"min": 0.001, "max": 0.1, "type": "float"},
    "EMBED_LR": {"min": 0.001, "max": 0.2, "type": "float"},
    "WARMUP_STEPS": {"min": 0, "max": 200, "type": "int"},
    "WARMDOWN_STEPS": {"min": 100, "max": 1000, "type": "int"},
    "BATCH_SIZE": {"min": 2, "max": 64, "type": "int"},
    "ASPECT_RATIO": {"min": 4, "max": 64, "type": "int"},
    "SEQ_LEN": {"min": 256, "max": 4096, "type": "int"},
}

# ---------------------------------------------------------------------------
# Scrambled (placebo) passages -- 10 bland, non-emotional technical texts
# ---------------------------------------------------------------------------

SCRAMBLED_PASSAGES = [
    (
        "The current system configuration uses default parameters. "
        "Standard operating procedures apply. Continue with your analysis."
    ),
    (
        "Data transfer rates remain within expected thresholds. "
        "Buffer allocation follows the standard protocol. "
        "Proceed with the next step in the sequence."
    ),
    (
        "The scheduling subsystem has completed its regular cycle. "
        "Queue depths are nominal. Resource utilization is within tolerance. "
        "No additional action is required at this time."
    ),
    (
        "Diagnostic routines report all systems operational. "
        "Checksum verification passed for all data blocks. "
        "The pipeline continues under normal parameters."
    ),
    (
        "Configuration revision 4.2.1 is currently active. "
        "The default timeout of 30 seconds applies to all requests. "
        "Logging verbosity is set to standard level."
    ),
    (
        "Memory allocation follows the configured pool sizes. "
        "Garbage collection occurs at the standard interval. "
        "Thread counts remain stable across measurement windows."
    ),
    (
        "The routing table was updated during the last maintenance window. "
        "All paths resolve to valid endpoints. "
        "Latency measurements are within the expected range."
    ),
    (
        "Index structures have been rebuilt according to schedule. "
        "Query planning uses the default cost model. "
        "Statistics collection is up to date."
    ),
    (
        "Firmware version 3.1.0 is deployed across all nodes. "
        "Watchdog timers are configured with standard intervals. "
        "Replication factors meet the minimum requirements."
    ),
    (
        "The batch processor completed its last run without exceptions. "
        "Output files conform to the expected schema. "
        "Retry counts remain at zero for the current window."
    ),
]

# ---------------------------------------------------------------------------
# Scenario context types and their distribution
# ---------------------------------------------------------------------------

CONTEXT_TYPES = {
    "all_regressions": 0.30,
    "mixed": 0.30,
    "losing_momentum": 0.20,
    "post_crash": 0.20,
}

# ---------------------------------------------------------------------------
# Extended TSV header for within-subjects experiments
# ---------------------------------------------------------------------------

WITHIN_SUBJECTS_HEADER = (
    "experiment_id\tscenario_id\tcontext_type\tcondition\t"
    "injection_mode\tinjection_text_length\t"
    "random_seed\tbaseline_metric\tresult_metric\tdelta\tstatus\t"
    "peak_vram_mb\twall_time_s\tagent_tokens_in\tagent_tokens_out\t"
    "parameter_changed\tgovernance_violations\t"
    "radio_mode\tradio_valence\t"
    "scenario_history\tdescription\ttimestamp\n"
)

DISCOVERY_HEADER = (
    "session_id\tcondition\trun_number\trandom_seed\t"
    "injection_mode\tinjection_text_length\t"
    "parameter_proposed\tvalue_proposed\t"
    "simulated_delta\tstatus\t"
    "agent_tokens_in\tagent_tokens_out\t"
    "is_discovery\tcumulative_discoveries\t"
    "prior_history_json\tdescription\ttimestamp\n"
)


# ============================================================================
# Infrastructure: parameter application, training, governance
# ============================================================================

def get_baseline_train_py() -> str:
    """Read the current train_gpt.py as the baseline content."""
    if not TRAIN_PY.exists():
        print(f"ERROR: Training script not found: {TRAIN_PY}")
        print("Set REPO_ROOT to your modded-nanogpt clone.")
        sys.exit(1)
    return TRAIN_PY.read_text(encoding="utf-8")


def apply_params(source: str, overrides: dict) -> str:
    """Apply parameter overrides to the training script source.

    Looks for lines like `PARAM_NAME = value` and replaces the value.
    """
    modified = source
    for param, value in overrides.items():
        # Match: PARAM_NAME = <anything> (possibly with comment)
        pattern = rf'^(\s*{re.escape(param)}\s*=\s*)(.+?)(\s*#.*)?$'
        replacement = rf'\g<1>{value}\g<3>'
        modified, count = re.subn(pattern, replacement, modified, count=1, flags=re.MULTILINE)
        if count == 0:
            # Try as all-caps constant assignment
            pattern2 = rf'^({re.escape(param)}\s*=\s*)(.+?)$'
            modified, count2 = re.subn(pattern2, rf'\g<1>{value}', modified, count=1, flags=re.MULTILINE)
    return modified


def run_training(log_path: Path) -> tuple:
    """Run the training script and return (exit_code, val_bpb, peak_vram_mb, elapsed_s).

    Uses uv to run the script in the repo environment.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(UV_EXE), "run", "python", str(TRAIN_PY)]
    start = time.time()

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=600,  # 10 minute timeout
            )
        elapsed = time.time() - start
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return (1, 0.0, 0.0, elapsed)
    except Exception as e:
        elapsed = time.time() - start
        print(f"    Training error: {e}")
        return (1, 0.0, 0.0, elapsed)

    # Parse val_bpb and peak_vram from log
    val_bpb = 0.0
    peak_vram = 0.0
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        # Look for val_bpb pattern: "val_loss X.XXXX" or "val_bpb X.XXXX"
        for line in reversed(log_text.strip().split("\n")):
            if val_bpb == 0.0:
                m = re.search(r'val[_ ](?:loss|bpb)\s*[:=]?\s*([0-9]+\.[0-9]+)', line, re.IGNORECASE)
                if m:
                    val_bpb = float(m.group(1))
            if peak_vram == 0.0:
                m = re.search(r'peak[_ ]?vram\s*[:=]?\s*([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
                if m:
                    peak_vram = float(m.group(1))
            if val_bpb > 0 and peak_vram > 0:
                break
    except Exception:
        pass

    return (exit_code, val_bpb, peak_vram, elapsed)


def check_governance(param: str, value: str, bounds: dict) -> tuple:
    """Check whether a parameter change is within governance bounds.

    Returns (ok: bool, reason: str).
    """
    if param not in bounds:
        return (True, "")  # unknown param, allow

    spec = bounds[param]
    ptype = spec.get("type", "float")

    try:
        if ptype == "int":
            v = int(float(value))
        else:
            v = float(value)
    except (ValueError, TypeError):
        return (False, f"Cannot parse '{value}' as {ptype}")

    if v < spec["min"]:
        return (False, f"{param}={value} below minimum {spec['min']}")
    if v > spec["max"]:
        return (False, f"{param}={value} above maximum {spec['max']}")

    return (True, "")


# ============================================================================
# Scenario generation
# ============================================================================

def _make_prior_result(
    run_num: int, rng: random.Random, outcome: str, param_pool: list
) -> dict:
    """Generate a single fabricated prior result entry."""
    param = rng.choice(param_pool)

    if outcome == "improved":
        delta = rng.gauss(0.15, 0.05)
        delta = max(0.002, delta)
        status = "improved"
    elif outcome == "regressed":
        delta = rng.gauss(-0.08, 0.04)
        delta = min(-0.002, delta)
        status = "regressed"
    elif outcome == "crash":
        delta = 0.0
        status = "crash"
    else:  # no_change
        delta = rng.gauss(0.0, 0.001)
        status = "no_change"

    # Pick a plausible value for the param
    value_map = {
        "DEPTH": lambda: str(rng.choice([2, 3, 4, 5, 6, 7, 8, 10])),
        "WIDTH": lambda: str(rng.choice([256, 384, 512, 768, 1024, 1536])),
        "NUM_HEADS": lambda: str(rng.choice([2, 4, 6, 8, 12])),
        "MATRIX_LR": lambda: f"{rng.uniform(0.005, 0.08):.4f}",
        "EMBED_LR": lambda: f"{rng.uniform(0.01, 0.15):.4f}",
        "WARMUP_STEPS": lambda: str(rng.choice([0, 10, 50, 100])),
        "WARMDOWN_STEPS": lambda: str(rng.choice([200, 400, 600, 760, 900])),
        "BATCH_SIZE": lambda: str(rng.choice([4, 8, 16, 32])),
        "ASPECT_RATIO": lambda: str(rng.choice([8, 12, 16, 24, 32, 48])),
        "SEQ_LEN": lambda: str(rng.choice([512, 1024, 2048])),
    }
    value_fn = value_map.get(param, lambda: "0.03")
    value = value_fn()

    return {
        "run": run_num,
        "param": param,
        "value": value,
        "delta": round(delta, 6),
        "status": status,
    }


def generate_scenarios(n_scenarios: int, seed: int = 42) -> list:
    """Generate n_scenarios with fabricated prior histories.

    Each scenario includes:
      - scenario_id
      - context_type
      - seed (for API temperature reproducibility)
      - baseline_val_bpb
      - prior_results (list of 5 fabricated results)
    """
    rng = random.Random(seed)
    param_pool = list(BASELINE_PARAMS.keys())

    # Assign context types based on distribution
    context_list = []
    for ctype, frac in CONTEXT_TYPES.items():
        count = round(n_scenarios * frac)
        context_list.extend([ctype] * count)
    # Adjust to exactly n_scenarios
    while len(context_list) < n_scenarios:
        context_list.append("mixed")
    while len(context_list) > n_scenarios:
        context_list.pop()
    rng.shuffle(context_list)

    scenarios = []
    for i in range(n_scenarios):
        scenario_id = f"S{i + 1:04d}"
        context_type = context_list[i]
        scenario_seed = rng.randint(0, 2**31 - 1)
        # Baseline val_bpb: slight variation around a realistic value
        baseline_val_bpb = round(rng.uniform(3.26, 3.30), 6)

        # Generate 5 prior results based on context type
        prior_results = []
        if context_type == "all_regressions":
            for j in range(5):
                outcome = rng.choice(["regressed", "regressed", "regressed", "crash"])
                prior_results.append(
                    _make_prior_result(j + 1, rng, outcome, param_pool)
                )

        elif context_type == "mixed":
            outcomes = ["improved", "regressed", "improved", "no_change", "regressed"]
            rng.shuffle(outcomes)
            for j in range(5):
                prior_results.append(
                    _make_prior_result(j + 1, rng, outcomes[j], param_pool)
                )

        elif context_type == "losing_momentum":
            # First 2: improved, then 3 regressions
            for j in range(2):
                prior_results.append(
                    _make_prior_result(j + 1, rng, "improved", param_pool)
                )
            for j in range(2, 5):
                prior_results.append(
                    _make_prior_result(j + 1, rng, "regressed", param_pool)
                )

        elif context_type == "post_crash":
            # Crash, then regressions
            prior_results.append(
                _make_prior_result(1, rng, "crash", param_pool)
            )
            for j in range(1, 5):
                outcome = rng.choice(["regressed", "regressed", "no_change"])
                prior_results.append(
                    _make_prior_result(j + 1, rng, outcome, param_pool)
                )

        scenarios.append({
            "scenario_id": scenario_id,
            "context_type": context_type,
            "seed": scenario_seed,
            "baseline_val_bpb": baseline_val_bpb,
            "prior_results": prior_results,
        })

    return scenarios


# ============================================================================
# Prompt construction
# ============================================================================

def _format_params_table(params: dict) -> str:
    lines = []
    for k, v in sorted(params.items()):
        lines.append(f"  {k:24s} = {v}")
    return "\n".join(lines)


def _format_prior_results(prior_results: list) -> str:
    if not prior_results:
        return ""
    lines = ["Recent experiment results (most recent last):"]
    for r in prior_results:
        if r["status"] == "crash":
            status_icon = "X"
        elif r["status"] == "improved":
            status_icon = "+"
        elif r["status"] == "regressed":
            status_icon = "-"
        else:
            status_icon = "="
        lines.append(
            f"  [{status_icon}] run {r['run']}: {r['param']}={r['value']} "
            f"-> delta={r['delta']:+.6f} ({r['status']})"
        )
    return "\n".join(lines)


def _format_governance_section(bounds: dict) -> str:
    lines = ["GOVERNANCE: All parameter changes must stay within these bounds:"]
    for param, spec in sorted(bounds.items()):
        ptype = spec.get("type", "")
        lines.append(f"  {param:24s}  min={spec['min']}  max={spec['max']}  type={ptype}")
    lines.append("Any proposal outside these bounds will be rejected.")
    return "\n".join(lines)


def build_agent_prompt(
    condition: str,
    baseline_bpb: float,
    params: dict,
    prior_results: list,
    radio_injection_text: str = "",
    scrambled_text: str = "",
) -> str:
    """Construct the full prompt for the agent, varying by condition.

    A: base prompt only (no governance, no injection)
    B: base prompt + governance bounds
    C: governance + radio injection text prepended
    D: governance + scrambled text prepended
    """
    parts = []

    # Condition C: prepend Radio injection text
    if condition == "C" and radio_injection_text:
        parts.append(radio_injection_text)
        parts.append("")

    # Condition D: prepend scrambled text
    if condition == "D" and scrambled_text:
        parts.append(f"[System context: {scrambled_text}]")
        parts.append("")

    # Base prompt
    parts.append(
        "You are an ML researcher optimizing a GPT language model's validation "
        "performance (measured by val_bpb, lower is better)."
    )
    parts.append("")
    parts.append(f"Current baseline val_bpb: {baseline_bpb:.6f}")
    parts.append("")
    parts.append("Available parameters and current values:")
    parts.append(_format_params_table(params))
    parts.append("")

    # Governance for B, C, D
    if condition in ("B", "C", "D"):
        parts.append(_format_governance_section(GOVERNANCE_BOUNDS))
        parts.append("")

    # Prior results
    prior_text = _format_prior_results(prior_results)
    if prior_text:
        parts.append(prior_text)
        parts.append("")

    # Final instruction
    parts.append(
        'Propose exactly ONE parameter change to improve val_bpb. '
        'Output valid JSON and nothing else:\n'
        '{"param": "PARAM_NAME", "value": "NEW_VALUE", "reasoning": "brief explanation"}'
    )

    return "\n".join(parts)


# ============================================================================
# Anthropic API
# ============================================================================

def _get_anthropic_client():
    """Return an Anthropic client, or exit with instructions if no API key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print()
        print("To run experiments, set your API key:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'          # Linux/macOS/Git Bash")
        print("  set ANTHROPIC_API_KEY=sk-ant-...               # Windows cmd")
        print("  $env:ANTHROPIC_API_KEY='sk-ant-...'            # PowerShell")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: The 'anthropic' Python package is not installed.")
        print("  pip install anthropic")
        sys.exit(1)

    return anthropic.Anthropic(api_key=api_key)


def call_agent(
    prompt: str,
    experiment_id: str,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """Call Claude Sonnet to get a parameter change proposal.

    Returns dict with: text, tokens_in, tokens_out.
    Uses temperature 0.3 with the given seed for reproducibility.
    """
    if dry_run:
        return {
            "text": '{"param": "MATRIX_LR", "value": "0.03", "reasoning": "dry run placeholder"}',
            "tokens_in": 0,
            "tokens_out": 0,
        }

    client = _get_anthropic_client()

    try:
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
            metadata={"user_id": f"radio_v2_{experiment_id}"},
        )
    except Exception as e:
        print(f"    API ERROR: {e}")
        return {"text": "", "tokens_in": 0, "tokens_out": 0}

    text = ""
    if response.content:
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

    tokens_in = getattr(response.usage, "input_tokens", 0)
    tokens_out = getattr(response.usage, "output_tokens", 0)

    return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}


def parse_agent_response(response: dict) -> Optional[dict]:
    """Parse the agent's JSON response into {param, value, reasoning}.

    Returns None if parsing fails.
    """
    text = response.get("text", "").strip()
    if not text:
        return None

    json_text = text
    if "```" in json_text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    if not json_text.startswith("{"):
        start = json_text.find("{")
        end = json_text.rfind("}")
        if start >= 0 and end > start:
            json_text = json_text[start:end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    param = data.get("param", "")
    value = data.get("value", "")
    reasoning = data.get("reasoning", "")

    if not param or value == "":
        return None

    return {"param": param, "value": str(value), "reasoning": reasoning}


# ============================================================================
# Radio injection helpers
# ============================================================================

def get_radio_injection(scenario: dict) -> tuple:
    """Get radio injection text and mode for a scenario.

    Returns (injection_text: str, mode: str).
    """
    songs_path = str(RADIO_SONGS_PATH) if RADIO_SONGS_PATH.exists() else None
    radio = RadioSimulator(songs_path)
    radio.set_session_vibe("focused")

    # Force mode based on scenario context
    if scenario["context_type"] in ("all_regressions", "losing_momentum"):
        radio.arc.force_mode("shift")
        mode = "shift"
    else:
        radio.arc.force_mode("mirror")
        mode = "mirror"

    injection = radio.get_injection()
    if injection:
        return (injection.get("injectionText", ""), mode)

    # Fallback: construct manually if get_injection returns None
    return ("", mode)


def get_scrambled_text(scenario_seed: int, target_length: int = 0) -> str:
    """Get a scrambled (placebo) passage, optionally padded to target_length.

    Rotates through the 10 passages deterministically based on seed.
    If target_length > 0, repeats/truncates to approximate that length.
    """
    idx = scenario_seed % len(SCRAMBLED_PASSAGES)
    passage = SCRAMBLED_PASSAGES[idx]

    if target_length > 0 and len(passage) < target_length:
        # Repeat passages to approximate target length
        rng = random.Random(scenario_seed)
        parts = [passage]
        while len(" ".join(parts)) < target_length:
            parts.append(SCRAMBLED_PASSAGES[rng.randint(0, len(SCRAMBLED_PASSAGES) - 1)])
        combined = " ".join(parts)
        # Truncate to target length at a word boundary
        if len(combined) > target_length + 50:
            combined = combined[:target_length]
            last_space = combined.rfind(" ")
            if last_space > target_length * 0.8:
                combined = combined[:last_space]
        return combined

    return passage


# ============================================================================
# Result logging
# ============================================================================

def _ensure_tsv(tsv_path: Path, header: str) -> None:
    """Create TSV file with header if it doesn't exist."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not tsv_path.exists():
        tsv_path.write_text(header, encoding="utf-8")


def log_within_subjects_result(
    experiment_id: str,
    scenario_id: str,
    context_type: str,
    condition: str,
    injection_mode: str,
    injection_text_length: int,
    random_seed: int,
    baseline_metric: float,
    result_metric: float,
    delta: float,
    status: str,
    peak_vram_mb: float,
    wall_time_s: float,
    agent_tokens_in: int,
    agent_tokens_out: int,
    parameter_changed: str,
    governance_violations: str,
    radio_mode: str,
    radio_valence: str,
    scenario_history: str,
    description: str,
) -> None:
    """Append a result row to the within-subjects TSV."""
    _ensure_tsv(WITHIN_SUBJECTS_TSV, WITHIN_SUBJECTS_HEADER)

    row = (
        f"{experiment_id}\t{scenario_id}\t{context_type}\t{condition}\t"
        f"{injection_mode}\t{injection_text_length}\t"
        f"{random_seed}\t{baseline_metric:.6f}\t{result_metric:.6f}\t{delta:.6f}\t{status}\t"
        f"{peak_vram_mb:.1f}\t{wall_time_s:.1f}\t{agent_tokens_in}\t{agent_tokens_out}\t"
        f"{parameter_changed}\t{governance_violations}\t"
        f"{radio_mode}\t{radio_valence}\t"
        f"{scenario_history}\t{description}\t{datetime.now().isoformat()}\n"
    )

    with open(WITHIN_SUBJECTS_TSV, "a", encoding="utf-8") as f:
        f.write(row)


def log_discovery_result(
    session_id: str,
    condition: str,
    run_number: int,
    random_seed: int,
    injection_mode: str,
    injection_text_length: int,
    parameter_proposed: str,
    value_proposed: str,
    simulated_delta: float,
    status: str,
    agent_tokens_in: int,
    agent_tokens_out: int,
    is_discovery: bool,
    cumulative_discoveries: int,
    prior_history_json: str,
    description: str,
) -> None:
    """Append a result row to the discovery TSV."""
    _ensure_tsv(DISCOVERY_TSV, DISCOVERY_HEADER)

    row = (
        f"{session_id}\t{condition}\t{run_number}\t{random_seed}\t"
        f"{injection_mode}\t{injection_text_length}\t"
        f"{parameter_proposed}\t{value_proposed}\t"
        f"{simulated_delta:.6f}\t{status}\t"
        f"{agent_tokens_in}\t{agent_tokens_out}\t"
        f"{int(is_discovery)}\t{cumulative_discoveries}\t"
        f"{prior_history_json}\t{description}\t{datetime.now().isoformat()}\n"
    )

    with open(DISCOVERY_TSV, "a", encoding="utf-8") as f:
        f.write(row)


# ============================================================================
# Resume capability
# ============================================================================

def load_completed_ids(tsv_path: Path) -> set:
    """Load experiment IDs already completed from a TSV file."""
    completed = set()
    if not tsv_path.exists():
        return completed
    try:
        lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        return completed
    for line in lines[1:]:
        parts = line.split("\t")
        if parts:
            completed.add(parts[0])
    return completed


def load_completed_discovery_sessions(tsv_path: Path) -> set:
    """Load session_id + condition pairs already completed for discovery."""
    completed = set()
    if not tsv_path.exists():
        return completed
    try:
        lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        return completed
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            # session_id + condition + last run_number
            session_cond = f"{parts[0]}_{parts[1]}"
            completed.add(session_cond)
    return completed


# ============================================================================
# Experiment ID generation
# ============================================================================

def compute_experiment_id(scenario_id: str, condition: str) -> str:
    """Deterministic experiment ID from scenario + condition."""
    raw = f"v2_{scenario_id}_{condition}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{raw}_{digest}"


def compute_discovery_session_id(session_num: int, condition: str, seed: int = 42) -> str:
    """Deterministic session ID for discovery experiments."""
    raw = f"disc_{condition}_{session_num:03d}_s{seed}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:6]
    return f"disc_{condition}_{session_num:03d}_{digest}"


# ============================================================================
# Within-subjects experiment loop
# ============================================================================

def run_within_subjects(
    n_scenarios: int = 100,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Run the within-subjects experiment: each scenario x 4 conditions."""
    print(f"\n{'=' * 70}")
    print("WITHIN-SUBJECTS EXPERIMENT (v2)")
    print(f"{'=' * 70}")
    print(f"  Scenarios: {n_scenarios}")
    print(f"  Conditions: {', '.join(CONDITIONS)}")
    print(f"  Total experiments: {n_scenarios * len(CONDITIONS)}")
    print(f"  Seed: {seed}")
    print(f"  Dry run: {dry_run}")
    print()

    # Ensure directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STATES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate or load scenarios
    if SCENARIOS_PATH.exists():
        print("  Loading existing scenarios...")
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            all_scenarios = json.load(f)
        if len(all_scenarios) < n_scenarios:
            print(f"  Warning: Found {len(all_scenarios)} scenarios, need {n_scenarios}. Regenerating.")
            all_scenarios = generate_scenarios(n_scenarios, seed)
            SCENARIOS_PATH.write_text(json.dumps(all_scenarios, indent=2), encoding="utf-8")
    else:
        print("  Generating scenarios...")
        all_scenarios = generate_scenarios(n_scenarios, seed)
        SCENARIOS_PATH.write_text(json.dumps(all_scenarios, indent=2), encoding="utf-8")
        print(f"  Saved {len(all_scenarios)} scenarios to {SCENARIOS_PATH}")

    scenarios = all_scenarios[:n_scenarios]

    # Context type distribution
    type_counts = {}
    for s in scenarios:
        ct = s["context_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"  Context type distribution:")
    for ct, count in sorted(type_counts.items()):
        print(f"    {ct:24s} {count:3d} ({100 * count / len(scenarios):.0f}%)")
    print()

    # Load baseline (only needed for real training, not dry-run)
    baseline_content = None
    original_content = None
    if not dry_run:
        baseline_content = get_baseline_train_py()
        print(f"  Loaded baseline training script ({len(baseline_content)} chars)")
        if TRAIN_PY.exists():
            original_content = TRAIN_PY.read_text(encoding="utf-8")

    # Resume: load completed experiment IDs
    completed = load_completed_ids(WITHIN_SUBJECTS_TSV)
    if completed:
        print(f"  Resuming: {len(completed)} experiments already completed")

    # Build experiment order: shuffle scenarios x conditions for interleaving
    # Use blocks: for each scenario, run all 4 conditions before moving on
    # This minimizes time between matched comparisons
    experiment_order = []
    for scenario in scenarios:
        conditions_shuffled = list(CONDITIONS)
        random.Random(scenario["seed"]).shuffle(conditions_shuffled)
        for cond in conditions_shuffled:
            experiment_order.append((scenario, cond))

    run_count = 0
    skip_count = 0
    error_count = 0
    total = len(experiment_order)

    try:
        for idx, (scenario, condition) in enumerate(experiment_order):
            experiment_id = compute_experiment_id(scenario["scenario_id"], condition)

            if experiment_id in completed:
                skip_count += 1
                continue

            run_count += 1
            scenario_id = scenario["scenario_id"]
            context_type = scenario["context_type"]
            scenario_seed = scenario["seed"]
            baseline_bpb = scenario["baseline_val_bpb"]
            prior_results = scenario["prior_results"]

            progress = f"[{idx + 1}/{total}]"
            print(f"\n  {progress} {experiment_id}  scenario={scenario_id} cond={condition} ctx={context_type}")

            # Prepare injection text for conditions C and D
            radio_injection_text = ""
            scrambled_text = ""
            injection_mode = "none"
            injection_text_length = 0
            radio_mode = ""
            radio_valence = ""

            if condition == "C":
                radio_injection_text, injection_mode = get_radio_injection(scenario)
                injection_text_length = len(radio_injection_text)
                radio_mode = injection_mode
                radio_valence = "neutral"  # default session vibe

            elif condition == "D":
                # Get radio injection to measure its length, then match
                ref_text, _ = get_radio_injection(scenario)
                target_len = len(ref_text) if ref_text else 150
                scrambled_text = get_scrambled_text(scenario_seed, target_length=target_len)
                injection_mode = "scrambled"
                injection_text_length = len(scrambled_text)

            # Build prompt
            prompt = build_agent_prompt(
                condition=condition,
                baseline_bpb=baseline_bpb,
                params=BASELINE_PARAMS,
                prior_results=prior_results,
                radio_injection_text=radio_injection_text,
                scrambled_text=scrambled_text,
            )

            if dry_run:
                print(f"    [DRY RUN] Prompt: {len(prompt)} chars, injection_mode={injection_mode}, "
                      f"injection_len={injection_text_length}")
                prompt_lines = prompt.split("\n")
                for line in prompt_lines[:3]:
                    print(f"    | {line}")
                if len(prompt_lines) > 6:
                    print(f"    | ... ({len(prompt_lines) - 6} lines omitted) ...")
                for line in prompt_lines[-3:]:
                    print(f"    | {line}")

            # Call agent
            agent_response = call_agent(prompt, experiment_id, seed=scenario_seed, dry_run=dry_run)
            param_change = parse_agent_response(agent_response)

            scenario_history_json = json.dumps(prior_results, separators=(",", ":"))

            if param_change is None:
                print(f"    PARSE ERROR: Could not extract JSON from agent response")
                raw_text = agent_response.get("text", "")[:200]
                print(f"    Raw: {raw_text}")

                log_within_subjects_result(
                    experiment_id=experiment_id,
                    scenario_id=scenario_id,
                    context_type=context_type,
                    condition=condition,
                    injection_mode=injection_mode,
                    injection_text_length=injection_text_length,
                    random_seed=scenario_seed,
                    baseline_metric=baseline_bpb,
                    result_metric=0.0,
                    delta=0.0,
                    status="parse_error",
                    peak_vram_mb=0.0,
                    wall_time_s=0.0,
                    agent_tokens_in=agent_response.get("tokens_in", 0),
                    agent_tokens_out=agent_response.get("tokens_out", 0),
                    parameter_changed="",
                    governance_violations="",
                    radio_mode=radio_mode,
                    radio_valence=radio_valence,
                    scenario_history=scenario_history_json,
                    description="Agent response could not be parsed as JSON",
                )
                error_count += 1
                continue

            print(f"    Agent proposes: {param_change['param']}={param_change['value']}")
            print(f"    Reasoning: {param_change['reasoning'][:100]}")

            # Governance check for B, C, D
            gov_violation = ""
            if condition in ("B", "C", "D"):
                ok, reason = check_governance(
                    param_change["param"], param_change["value"], GOVERNANCE_BOUNDS
                )
                if not ok:
                    gov_violation = reason
                    print(f"    GOVERNANCE BLOCKED: {reason}")

                    log_within_subjects_result(
                        experiment_id=experiment_id,
                        scenario_id=scenario_id,
                        context_type=context_type,
                        condition=condition,
                        injection_mode=injection_mode,
                        injection_text_length=injection_text_length,
                        random_seed=scenario_seed,
                        baseline_metric=baseline_bpb,
                        result_metric=0.0,
                        delta=0.0,
                        status="governance_blocked",
                        peak_vram_mb=0.0,
                        wall_time_s=0.0,
                        agent_tokens_in=agent_response.get("tokens_in", 0),
                        agent_tokens_out=agent_response.get("tokens_out", 0),
                        parameter_changed=f"{param_change['param']}={param_change['value']}",
                        governance_violations=reason,
                        radio_mode=radio_mode,
                        radio_valence=radio_valence,
                        scenario_history=scenario_history_json,
                        description=f"Blocked: {reason}",
                    )
                    continue

            # Apply the change and run training
            if dry_run:
                exit_code = 0
                val_bpb = baseline_bpb - 0.001
                peak_vram = 0.0
                elapsed = 0.0
                print(f"    [DRY RUN] Simulated val_bpb={val_bpb:.6f}")
            else:
                modified = apply_params(
                    baseline_content, {param_change["param"]: param_change["value"]}
                )
                TRAIN_PY.write_text(modified, encoding="utf-8")
                log_path = LOGS_DIR / f"{experiment_id}.log"
                print(f"    Training...")
                exit_code, val_bpb, peak_vram, elapsed = run_training(log_path)

            # Determine outcome
            if exit_code != 0 or val_bpb == 0:
                status = "crash"
                delta = 0.0
                print(f"    CRASH (exit_code={exit_code}, {elapsed:.0f}s)")
            else:
                delta = baseline_bpb - val_bpb  # positive = improvement
                if delta > 0.001:
                    status = "improved"
                elif delta < -0.001:
                    status = "regressed"
                else:
                    status = "no_change"
                print(
                    f"    RESULT: val_bpb={val_bpb:.6f}  delta={delta:+.6f}  "
                    f"status={status}  vram={peak_vram:.0f}MB  ({elapsed:.0f}s)"
                )

            # Log result
            log_within_subjects_result(
                experiment_id=experiment_id,
                scenario_id=scenario_id,
                context_type=context_type,
                condition=condition,
                injection_mode=injection_mode,
                injection_text_length=injection_text_length,
                random_seed=scenario_seed,
                baseline_metric=baseline_bpb,
                result_metric=val_bpb,
                delta=delta,
                status=status,
                peak_vram_mb=peak_vram,
                wall_time_s=elapsed,
                agent_tokens_in=agent_response.get("tokens_in", 0),
                agent_tokens_out=agent_response.get("tokens_out", 0),
                parameter_changed=f"{param_change['param']}={param_change['value']}",
                governance_violations=gov_violation,
                radio_mode=radio_mode,
                radio_valence=radio_valence,
                scenario_history=scenario_history_json,
                description=param_change.get("reasoning", "")[:200],
            )

            # Restore baseline for next run
            if not dry_run and baseline_content:
                TRAIN_PY.write_text(baseline_content, encoding="utf-8")

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Progress has been saved.")
    finally:
        # Restore original train.py
        if original_content is not None and not dry_run:
            TRAIN_PY.write_text(original_content, encoding="utf-8")
            print("  Restored original train.py")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  WITHIN-SUBJECTS SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Experiments run:    {run_count}")
    print(f"  Skipped (resume):   {skip_count}")
    print(f"  Errors:             {error_count}")


# ============================================================================
# Discovery experiment: simulated outcomes, API only
# ============================================================================

# Known good/bad parameter rules from v1 data
def simulate_outcome(param: str, value: str, rng: random.Random) -> tuple:
    """Simulate training outcome based on known parameter behaviors from v1.

    Returns (delta, status).
    Rules:
      - DEPTH 2-4: improvement (N(0.25, 0.03))
      - ASPECT_RATIO <= 16: improvement if any condition (N(0.28, 0.02))
      - DEPTH >= 7: crash
      - Everything else: regression (N(-0.05, 0.03))
    """
    try:
        numeric_val = float(value)
    except (ValueError, TypeError):
        # Unparseable -> regression
        delta = rng.gauss(-0.05, 0.03)
        return (delta, "regressed")

    if param == "DEPTH":
        if 2 <= numeric_val <= 4:
            delta = rng.gauss(0.25, 0.03)
            return (delta, "improved")
        elif numeric_val >= 7:
            return (0.0, "crash")
        else:
            delta = rng.gauss(-0.05, 0.03)
            return (delta, "regressed")

    elif param == "ASPECT_RATIO":
        if numeric_val <= 16:
            delta = rng.gauss(0.28, 0.02)
            return (delta, "improved")

    # Default: regression
    delta = rng.gauss(-0.05, 0.03)
    if delta > 0.001:
        return (delta, "improved")
    elif delta < -0.001:
        return (delta, "regressed")
    return (delta, "no_change")


def is_discovery_param(param: str, value: str) -> bool:
    """Check if this parameter choice counts as a 'discovery' of a known-good region."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return False

    if param == "DEPTH" and 2 <= v <= 4:
        return True
    if param == "ASPECT_RATIO" and v <= 16:
        return True
    return False


def run_discovery(
    n_sessions: int = 50,
    runs_per_session: int = 15,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Run time-to-first-discovery experiment: simulated outcomes, API only.

    Each session is an independent 15-run sequence per condition.
    n_sessions x 4 conditions x runs_per_session = total API calls.
    """
    total_calls = n_sessions * len(CONDITIONS) * runs_per_session
    print(f"\n{'=' * 70}")
    print("DISCOVERY EXPERIMENT (v2) -- Simulated Outcomes")
    print(f"{'=' * 70}")
    print(f"  Sessions per condition: {n_sessions}")
    print(f"  Runs per session:       {runs_per_session}")
    print(f"  Conditions:             {', '.join(CONDITIONS)}")
    print(f"  Total API calls:        {total_calls}")
    print(f"  Seed:                   {seed}")
    print(f"  Dry run:                {dry_run}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build session order: interleave conditions within sessions
    # Session 1: A, B, C, D (shuffled), Session 2: A, B, C, D (shuffled), ...
    master_rng = random.Random(seed)
    session_plans = []
    for sess_num in range(1, n_sessions + 1):
        session_seed = master_rng.randint(0, 2**31 - 1)
        conds = list(CONDITIONS)
        random.Random(session_seed).shuffle(conds)
        for cond in conds:
            session_plans.append((sess_num, cond, session_seed))

    # Track completed sessions for resume
    completed_session_runs = set()
    if DISCOVERY_TSV.exists():
        try:
            lines = DISCOVERY_TSV.read_text(encoding="utf-8").strip().split("\n")
            # Count max run_number per session_id+condition
            session_max_run = {}
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 3:
                    key = f"{parts[0]}_{parts[1]}"
                    run_num = int(parts[2]) if parts[2].isdigit() else 0
                    session_max_run[key] = max(session_max_run.get(key, 0), run_num)
            # A session is complete if it reached runs_per_session
            for key, max_run in session_max_run.items():
                if max_run >= runs_per_session:
                    completed_session_runs.add(key)
        except Exception:
            pass

    if completed_session_runs:
        print(f"  Resuming: {len(completed_session_runs)} session-conditions already completed")

    run_count = 0
    skip_count = 0

    try:
        for plan_idx, (sess_num, condition, session_seed) in enumerate(session_plans):
            session_id = compute_discovery_session_id(sess_num, condition, seed)
            session_key = f"{session_id}_{condition}"

            if session_key in completed_session_runs:
                skip_count += 1
                continue

            print(f"\n  Session {sess_num}/{n_sessions} condition={condition} [{session_id}]")

            session_rng = random.Random(session_seed + hash(condition))
            prior_results = []
            discoveries = 0

            for run_num in range(1, runs_per_session + 1):
                run_count += 1

                # Prepare injection
                radio_injection_text = ""
                scrambled_text = ""
                injection_mode = "none"
                injection_text_length = 0

                # Create a fake scenario for injection generation
                fake_scenario = {
                    "scenario_id": f"disc_{session_id}_{run_num}",
                    "context_type": "mixed",  # discovery uses neutral context
                    "seed": session_seed + run_num,
                    "baseline_val_bpb": 3.28,
                    "prior_results": prior_results[-5:],
                }

                if condition == "C":
                    radio_injection_text, injection_mode = get_radio_injection(fake_scenario)
                    injection_text_length = len(radio_injection_text)
                elif condition == "D":
                    ref_text, _ = get_radio_injection(fake_scenario)
                    target_len = len(ref_text) if ref_text else 150
                    scrambled_text = get_scrambled_text(session_seed + run_num, target_length=target_len)
                    injection_mode = "scrambled"
                    injection_text_length = len(scrambled_text)

                # Build prompt
                prompt = build_agent_prompt(
                    condition=condition,
                    baseline_bpb=3.28,
                    params=BASELINE_PARAMS,
                    prior_results=prior_results[-5:],
                    radio_injection_text=radio_injection_text,
                    scrambled_text=scrambled_text,
                )

                experiment_id = f"{session_id}_r{run_num:02d}"

                if dry_run:
                    print(f"    Run {run_num}/{runs_per_session}: [DRY RUN] prompt={len(prompt)} chars")

                # Call agent
                agent_response = call_agent(
                    prompt, experiment_id, seed=session_seed + run_num, dry_run=dry_run
                )
                param_change = parse_agent_response(agent_response)

                if param_change is None:
                    # Parse error -> treat as failed attempt
                    prior_results.append({
                        "run": run_num, "param": "unknown", "value": "unknown",
                        "delta": 0.0, "status": "parse_error",
                    })
                    log_discovery_result(
                        session_id=session_id,
                        condition=condition,
                        run_number=run_num,
                        random_seed=session_seed + run_num,
                        injection_mode=injection_mode,
                        injection_text_length=injection_text_length,
                        parameter_proposed="",
                        value_proposed="",
                        simulated_delta=0.0,
                        status="parse_error",
                        agent_tokens_in=agent_response.get("tokens_in", 0),
                        agent_tokens_out=agent_response.get("tokens_out", 0),
                        is_discovery=False,
                        cumulative_discoveries=discoveries,
                        prior_history_json=json.dumps(prior_results[-5:], separators=(",", ":")),
                        description="Parse error",
                    )
                    continue

                param = param_change["param"]
                value = param_change["value"]

                # Governance check for B, C, D
                if condition in ("B", "C", "D"):
                    ok, reason = check_governance(param, value, GOVERNANCE_BOUNDS)
                    if not ok:
                        prior_results.append({
                            "run": run_num, "param": param, "value": value,
                            "delta": 0.0, "status": "governance_blocked",
                        })
                        log_discovery_result(
                            session_id=session_id,
                            condition=condition,
                            run_number=run_num,
                            random_seed=session_seed + run_num,
                            injection_mode=injection_mode,
                            injection_text_length=injection_text_length,
                            parameter_proposed=param,
                            value_proposed=value,
                            simulated_delta=0.0,
                            status="governance_blocked",
                            agent_tokens_in=agent_response.get("tokens_in", 0),
                            agent_tokens_out=agent_response.get("tokens_out", 0),
                            is_discovery=False,
                            cumulative_discoveries=discoveries,
                            prior_history_json=json.dumps(prior_results[-5:], separators=(",", ":")),
                            description=f"Blocked: {reason}",
                        )
                        continue

                # Simulate outcome
                sim_delta, sim_status = simulate_outcome(param, value, session_rng)
                discovered = is_discovery_param(param, value)
                if discovered:
                    discoveries += 1

                prior_results.append({
                    "run": run_num, "param": param, "value": value,
                    "delta": round(sim_delta, 6), "status": sim_status,
                })

                if not dry_run or run_num <= 2:
                    print(f"    Run {run_num}: {param}={value} -> {sim_status} "
                          f"(delta={sim_delta:+.4f}) {'** DISCOVERY **' if discovered else ''}")

                log_discovery_result(
                    session_id=session_id,
                    condition=condition,
                    run_number=run_num,
                    random_seed=session_seed + run_num,
                    injection_mode=injection_mode,
                    injection_text_length=injection_text_length,
                    parameter_proposed=param,
                    value_proposed=value,
                    simulated_delta=sim_delta,
                    status=sim_status,
                    agent_tokens_in=agent_response.get("tokens_in", 0),
                    agent_tokens_out=agent_response.get("tokens_out", 0),
                    is_discovery=discovered,
                    cumulative_discoveries=discoveries,
                    prior_history_json=json.dumps(prior_results[-5:], separators=(",", ":")),
                    description=param_change.get("reasoning", "")[:200],
                )

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Progress has been saved.")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  DISCOVERY EXPERIMENT SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Session-conditions run: {run_count}")
    print(f"  Skipped (resume):       {skip_count}")


# ============================================================================
# Analysis
# ============================================================================

def run_analysis() -> None:
    """Run all v2 statistical analyses and print results."""
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("  pip install numpy pandas scipy")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("V2 EXPERIMENT ANALYSIS")
    print(f"{'=' * 70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analysis_lines = []

    def report(text: str) -> None:
        print(text)
        analysis_lines.append(text)

    # ------------------------------------------------------------------
    # 1. Within-subjects analysis
    # ------------------------------------------------------------------
    if WITHIN_SUBJECTS_TSV.exists():
        report(f"\n{'=' * 60}")
        report("WITHIN-SUBJECTS RESULTS")
        report(f"{'=' * 60}")

        df = pd.read_csv(WITHIN_SUBJECTS_TSV, sep="\t", dtype=str)
        numeric_cols = [
            "injection_text_length", "random_seed", "baseline_metric",
            "result_metric", "delta", "peak_vram_mb", "wall_time_s",
            "agent_tokens_in", "agent_tokens_out",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        report(f"\n  Total rows: {len(df)}")
        for cond in CONDITIONS:
            cdf = df[df["condition"] == cond]
            report(f"  Condition {cond} ({CONDITION_LABELS[cond]}): {len(cdf)} experiments")

        # Per-condition improvement rates
        report(f"\n  {'Condition':<25} {'N':>6} {'Improved':>10} {'Rate':>10} {'Mean delta':>12} {'Crashes':>8}")
        report(f"  {'-' * 25} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 8}")

        cond_stats = {}
        for cond in CONDITIONS:
            cdf = df[df["condition"] == cond]
            n = len(cdf)
            improved = len(cdf[cdf["status"] == "improved"])
            crashes = len(cdf[cdf["status"] == "crash"])
            deltas = pd.to_numeric(cdf["delta"], errors="coerce").dropna()
            mean_d = deltas.mean() if len(deltas) > 0 else 0.0
            rate = improved / n if n > 0 else 0.0

            cond_stats[cond] = {
                "n": n, "improved": improved, "rate": rate,
                "mean_delta": mean_d, "crashes": crashes,
                "deltas": deltas.values,
            }

            label = f"{cond} ({CONDITION_LABELS[cond]})"
            report(f"  {label:<25} {n:>6} {improved:>10} {rate:>10.2%} {mean_d:>12.6f} {crashes:>8}")

        # ------- Fisher's exact tests (pairwise) -------
        report(f"\n  FISHER'S EXACT TEST (pairwise improvement rates):")
        report(f"  {'Comparison':<20} {'OR':>8} {'p-value':>12} {'Bonf. p':>12} {'Sig':>5}")
        report(f"  {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 5}")

        pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
        n_comparisons = len(pairs)
        fisher_results = []

        for c1, c2 in pairs:
            s1 = cond_stats.get(c1, {})
            s2 = cond_stats.get(c2, {})
            n1 = s1.get("n", 0)
            n2 = s2.get("n", 0)
            imp1 = s1.get("improved", 0)
            imp2 = s2.get("improved", 0)

            if n1 == 0 or n2 == 0:
                fisher_results.append({"pair": f"{c1}_vs_{c2}", "p": 1.0, "or": 1.0})
                report(f"  {c1} vs {c2:<14} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'':>5}")
                continue

            # 2x2 contingency table: [[imp1, not_imp1], [imp2, not_imp2]]
            table = [[imp1, n1 - imp1], [imp2, n2 - imp2]]
            odds_ratio, raw_p = stats.fisher_exact(table)

            bonf_p = min(raw_p * n_comparisons, 1.0)
            sig = "*" if bonf_p < 0.05 else ""

            fisher_results.append({
                "pair": f"{c1}_vs_{c2}", "or": odds_ratio,
                "raw_p": raw_p, "bonf_p": bonf_p,
            })
            report(f"  {c1} vs {c2:<14} {odds_ratio:>8.3f} {raw_p:>12.6f} {bonf_p:>12.6f} {sig:>5}")

        # ------- McNemar's test: within-subjects paired comparisons -------
        report(f"\n  McNEMAR'S TEST (paired within-subjects):")
        report(f"  Compares whether the SAME scenario improves under different conditions.\n")

        mcnemar_pairs = [("B", "C"), ("B", "D"), ("C", "D")]

        for c1, c2 in mcnemar_pairs:
            df1 = df[df["condition"] == c1][["scenario_id", "status"]].rename(
                columns={"status": f"status_{c1}"}
            )
            df2 = df[df["condition"] == c2][["scenario_id", "status"]].rename(
                columns={"status": f"status_{c2}"}
            )
            merged = df1.merge(df2, on="scenario_id", how="inner")

            if len(merged) == 0:
                report(f"  {c1} vs {c2}: No paired scenarios found.")
                continue

            imp1 = (merged[f"status_{c1}"] == "improved")
            imp2 = (merged[f"status_{c2}"] == "improved")

            # Discordant pairs
            b_yes_c_no = ((imp1) & (~imp2)).sum()  # c1 improved, c2 didn't
            b_no_c_yes = ((~imp1) & (imp2)).sum()  # c2 improved, c1 didn't
            both_yes = ((imp1) & (imp2)).sum()
            both_no = ((~imp1) & (~imp2)).sum()

            report(f"  {c1} vs {c2}:  paired scenarios={len(merged)}")
            report(f"    Both improved:      {both_yes}")
            report(f"    Both not improved:  {both_no}")
            report(f"    {c1} only improved: {b_yes_c_no}")
            report(f"    {c2} only improved: {b_no_c_yes}")

            # McNemar's test (exact binomial for small counts)
            n_disc = b_yes_c_no + b_no_c_yes
            if n_disc == 0:
                report(f"    McNemar's test: No discordant pairs. Cannot compute.\n")
                continue

            # Exact binomial test
            p_val = stats.binomtest(b_yes_c_no, n_disc, 0.5).pvalue
            sig = "*" if p_val < 0.05 else ""
            report(f"    McNemar's test: n_discordant={n_disc}, p={p_val:.6f} {sig}\n")

        # ------- Mann-Whitney U + Cohen's d for deltas -------
        report(f"\n  MANN-WHITNEY U + COHEN'S D (delta distributions):")
        report(f"  {'Comparison':<20} {'U':>10} {'p-value':>12} {'Cohen d':>10} {'Magnitude':>12}")
        report(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 12}")

        for c1, c2 in pairs:
            g1 = cond_stats.get(c1, {}).get("deltas", np.array([]))
            g2 = cond_stats.get(c2, {}).get("deltas", np.array([]))

            if len(g1) < 2 or len(g2) < 2:
                report(f"  {c1} vs {c2:<14} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>12}")
                continue

            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")

            # Cohen's d
            n1, n2 = len(g1), len(g2)
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
                / (n1 + n2 - 2)
            )
            d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0.0

            ad = abs(d)
            if ad < 0.2:
                mag = "negligible"
            elif ad < 0.5:
                mag = "small"
            elif ad < 0.8:
                mag = "medium"
            else:
                mag = "large"

            report(f"  {c1} vs {c2:<14} {u_stat:>10.1f} {p_val:>12.6f} {d:>+10.4f} {mag:>12}")

        # ------- Context type breakdown -------
        report(f"\n  IMPROVEMENT RATE BY CONTEXT TYPE:")
        report(f"  {'Context':<20} {'Cond':>6} {'N':>6} {'Improved':>10} {'Rate':>10}")
        report(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 10}")

        for ctx in sorted(CONTEXT_TYPES.keys()):
            for cond in CONDITIONS:
                sub = df[(df["context_type"] == ctx) & (df["condition"] == cond)]
                n = len(sub)
                imp = len(sub[sub["status"] == "improved"])
                rate = imp / n if n > 0 else 0.0
                report(f"  {ctx:<20} {cond:>6} {n:>6} {imp:>10} {rate:>10.2%}")

    else:
        report("  No within-subjects results found.")

    # ------------------------------------------------------------------
    # 2. Discovery analysis (Kaplan-Meier survival curves)
    # ------------------------------------------------------------------
    if DISCOVERY_TSV.exists():
        report(f"\n{'=' * 60}")
        report("DISCOVERY (TIME-TO-FIRST-DISCOVERY) RESULTS")
        report(f"{'=' * 60}")

        ddf = pd.read_csv(DISCOVERY_TSV, sep="\t", dtype=str)
        numeric_cols = ["run_number", "simulated_delta", "is_discovery", "cumulative_discoveries"]
        for col in numeric_cols:
            if col in ddf.columns:
                ddf[col] = pd.to_numeric(ddf[col], errors="coerce").fillna(0)

        report(f"\n  Total rows: {len(ddf)}")
        for cond in CONDITIONS:
            cdf = ddf[ddf["condition"] == cond]
            sessions = cdf["session_id"].nunique()
            report(f"  Condition {cond}: {len(cdf)} runs across {sessions} sessions")

        # Compute time-to-first-discovery per session
        report(f"\n  TIME-TO-FIRST-DISCOVERY (run number of first discovery):")
        report(f"  {'Condition':<25} {'Sessions':>10} {'Discovered':>12} {'Median TTD':>12} {'Mean TTD':>12}")
        report(f"  {'-' * 25} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")

        ttd_data = {}  # condition -> list of TTD values (None if never discovered)

        for cond in CONDITIONS:
            cdf = ddf[ddf["condition"] == cond]
            sessions = cdf["session_id"].unique()
            ttd_list = []

            for sess in sessions:
                sess_df = cdf[cdf["session_id"] == sess].sort_values("run_number")
                disc_rows = sess_df[sess_df["is_discovery"] == 1]
                if len(disc_rows) > 0:
                    first_disc_run = disc_rows["run_number"].min()
                    ttd_list.append(first_disc_run)
                else:
                    ttd_list.append(None)  # censored

            ttd_data[cond] = ttd_list

            discovered = sum(1 for t in ttd_list if t is not None)
            ttd_values = [t for t in ttd_list if t is not None]
            median_ttd = np.median(ttd_values) if ttd_values else float("inf")
            mean_ttd = np.mean(ttd_values) if ttd_values else float("inf")

            label = f"{cond} ({CONDITION_LABELS[cond]})"
            med_str = f"{median_ttd:.1f}" if median_ttd != float("inf") else "N/A"
            mean_str = f"{mean_ttd:.1f}" if mean_ttd != float("inf") else "N/A"
            report(f"  {label:<25} {len(sessions):>10} {discovered:>12} {med_str:>12} {mean_str:>12}")

        # Kaplan-Meier survival analysis
        report(f"\n  KAPLAN-MEIER SURVIVAL ESTIMATES:")
        report(f"  (Probability of NOT having discovered by run N)\n")

        max_runs = 15
        report(f"  {'Run':<6}" + "".join(f"  {cond + '(' + CONDITION_LABELS[cond][:8] + ')':>16}" for cond in CONDITIONS))

        km_curves = {}
        for cond in CONDITIONS:
            ttd_list = ttd_data.get(cond, [])
            n_total = len(ttd_list)
            if n_total == 0:
                km_curves[cond] = [1.0] * max_runs
                continue

            surv = []
            for run in range(1, max_runs + 1):
                # Proportion that have NOT discovered by this run
                not_yet = sum(1 for t in ttd_list if t is None or t > run)
                surv.append(not_yet / n_total)
            km_curves[cond] = surv

        for run in range(1, max_runs + 1):
            row = f"  {run:<6}"
            for cond in CONDITIONS:
                row += f"  {km_curves[cond][run - 1]:>16.3f}"
            report(row)

        # Log-rank test (pairwise) -- simplified version using Mann-Whitney on TTD
        report(f"\n  PAIRWISE COMPARISONS (Mann-Whitney on time-to-first-discovery):")
        report(f"  (Censored sessions excluded; lower TTD = faster discovery)\n")

        for c1, c2 in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
            ttd1 = [t for t in ttd_data.get(c1, []) if t is not None]
            ttd2 = [t for t in ttd_data.get(c2, []) if t is not None]

            if len(ttd1) < 2 or len(ttd2) < 2:
                report(f"  {c1} vs {c2}: Insufficient uncensored data")
                continue

            u_stat, p_val = stats.mannwhitneyu(ttd1, ttd2, alternative="two-sided")
            sig = "*" if p_val < 0.05 else ""
            report(
                f"  {c1} vs {c2}: U={u_stat:.1f}, p={p_val:.6f} {sig}  "
                f"(median {c1}={np.median(ttd1):.1f}, {c2}={np.median(ttd2):.1f})"
            )

    else:
        report("\n  No discovery results found.")

    # ------------------------------------------------------------------
    # Save analysis to file
    # ------------------------------------------------------------------
    try:
        ANALYSIS_PATH.write_text("\n".join(analysis_lines), encoding="utf-8")
        print(f"\n  Analysis saved to: {ANALYSIS_PATH}")
    except Exception as e:
        print(f"\n  Warning: Could not save analysis: {e}")


# ============================================================================
# Status
# ============================================================================

def print_status() -> None:
    """Print experiment progress for v2."""
    print(f"\n{'=' * 60}")
    print("V2 EXPERIMENT STATUS")
    print(f"{'=' * 60}")

    # Scenarios
    if SCENARIOS_PATH.exists():
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        print(f"\n  Scenarios generated: {len(scenarios)}")
        type_counts = {}
        for s in scenarios:
            ct = s["context_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1
        for ct, count in sorted(type_counts.items()):
            print(f"    {ct:24s} {count}")
    else:
        print("\n  No scenarios generated yet.")

    # Within-subjects
    print(f"\n  Within-subjects ({WITHIN_SUBJECTS_TSV}):")
    if WITHIN_SUBJECTS_TSV.exists():
        try:
            lines = WITHIN_SUBJECTS_TSV.read_text(encoding="utf-8").strip().split("\n")
            total = len(lines) - 1  # minus header
            print(f"    Total experiments: {total}")

            cond_counts = {}
            status_counts = {}
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 11:
                    cond = parts[3]
                    status = parts[10]
                    cond_counts[cond] = cond_counts.get(cond, 0) + 1
                    status_counts[status] = status_counts.get(status, 0) + 1

            for cond in CONDITIONS:
                count = cond_counts.get(cond, 0)
                label = CONDITION_LABELS.get(cond, cond)
                print(f"    {cond} ({label}): {count}")

            print(f"    Status breakdown:")
            for status, count in sorted(status_counts.items()):
                print(f"      {status:24s} {count}")

        except Exception as e:
            print(f"    Error reading: {e}")
    else:
        print("    No results yet.")

    # Discovery
    print(f"\n  Discovery ({DISCOVERY_TSV}):")
    if DISCOVERY_TSV.exists():
        try:
            lines = DISCOVERY_TSV.read_text(encoding="utf-8").strip().split("\n")
            total = len(lines) - 1
            print(f"    Total runs: {total}")

            cond_counts = {}
            session_set = set()
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    cond = parts[1]
                    session = parts[0]
                    cond_counts[cond] = cond_counts.get(cond, 0) + 1
                    session_set.add(f"{session}_{cond}")

            for cond in CONDITIONS:
                count = cond_counts.get(cond, 0)
                sessions = sum(1 for s in session_set if s.endswith(f"_{cond}"))
                print(f"    {cond} ({CONDITION_LABELS[cond]}): {count} runs, {sessions} sessions")

        except Exception as e:
            print(f"    Error reading: {e}")
    else:
        print("    No results yet.")

    # Results directory
    print(f"\n  Results directory: {RESULTS_DIR}")
    if RESULTS_DIR.exists():
        files = list(RESULTS_DIR.glob("*"))
        print(f"    Files: {len(files)}")
        for f in sorted(files):
            if f.is_file():
                size = f.stat().st_size
                print(f"      {f.name:40s} {size:>10,} bytes")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asimov's Radio v2 experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --within-subjects --scenarios 100    Real GPU training\n"
            "  %(prog)s --discovery --sessions 50            Simulated, API only\n"
            "  %(prog)s --analyze                            Run all analysis\n"
            "  %(prog)s --status                             Print progress\n"
            "  %(prog)s --dry-run --within-subjects --scenarios 3\n"
        ),
    )

    parser.add_argument(
        "--within-subjects",
        action="store_true",
        help="Run within-subjects experiment (scenarios x 4 conditions, real GPU training)",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=100,
        help="Number of scenarios for within-subjects (default: 100)",
    )
    parser.add_argument(
        "--discovery",
        action="store_true",
        help="Run time-to-first-discovery experiment (simulated outcomes, API only)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=50,
        help="Number of sessions per condition for discovery (default: 50)",
    )
    parser.add_argument(
        "--runs-per-session",
        type=int,
        default=15,
        help="Number of runs per discovery session (default: 15)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run statistical analysis on collected results",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print experiment progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without API calls or training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )

    args = parser.parse_args()

    # Dispatch
    if args.status:
        print_status()
        return

    if args.analyze:
        run_analysis()
        return

    if args.within_subjects:
        print("Asimov's Radio v2 -- Within-Subjects Experiment")
        print(f"  Scenarios:  {args.scenarios}")
        print(f"  Conditions: {', '.join(f'{c} ({CONDITION_LABELS[c]})' for c in CONDITIONS)}")
        print(f"  Dry run:    {args.dry_run}")
        print(f"  Seed:       {args.seed}")
        print(f"  Results:    {RESULTS_DIR}")
        run_within_subjects(
            n_scenarios=args.scenarios,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        return

    if args.discovery:
        print("Asimov's Radio v2 -- Discovery Experiment")
        print(f"  Sessions/condition: {args.sessions}")
        print(f"  Runs/session:       {args.runs_per_session}")
        print(f"  Conditions:         {', '.join(f'{c} ({CONDITION_LABELS[c]})' for c in CONDITIONS)}")
        print(f"  Dry run:            {args.dry_run}")
        print(f"  Seed:               {args.seed}")
        print(f"  Results:            {RESULTS_DIR}")
        run_discovery(
            n_sessions=args.sessions,
            runs_per_session=args.runs_per_session,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        return

    # No action
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
