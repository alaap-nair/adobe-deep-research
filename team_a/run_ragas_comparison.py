#!/usr/bin/env python3
"""
Prepare RAGAS-shaped datasets, run ragas_eval.py on before/after, and write comparison artifacts.

Usage:
  python run_ragas_comparison.py

Requires OPENAI_API_KEY (or OPENROUTER_API_KEY + OPENAI_BASE_URL for OpenRouter) for ragas_eval.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
# Main project venv may be a pyenv build without `_lzma` (required by `datasets` / RAGAS).
RAGAS_VENV_PYTHON = REPO_ROOT / ".venv_ragas" / "bin" / "python"
ROOT_BEFORE = REPO_ROOT / "before_data.json"
DATA_BEFORE = REPO_ROOT / "data" / "before_data.json"
DATA_AFTER = REPO_ROOT / "data" / "after.json"
BEFORE_TXT = REPO_ROOT / "before_results.txt"
AFTER_TXT = REPO_ROOT / "after_results.txt"
COMPARISON_JSON = REPO_ROOT / "data" / "ragas_comparison.json"
COMPARISON_MD = REPO_ROOT / "data" / "ragas_comparison.md"
METRICS_BEFORE = REPO_ROOT / "data" / ".ragas_metrics_before.json"
METRICS_AFTER = REPO_ROOT / "data" / ".ragas_metrics_after.json"


def _ensure_openai_env() -> None:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        alt = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if alt:
            os.environ["OPENAI_API_KEY"] = alt
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def assignment_rows_to_ragas(rows: list[dict]) -> list[dict]:
    """Convert assignment baseline JSON (question, ground_truth, actual_terminal_result) to RAGAS rows."""
    out: list[dict] = []
    for index, item in enumerate(rows, start=1):
        q = (item.get("question") or item.get("user_input") or "").strip()
        ref = item.get("ground_truth") or item.get("reference") or ""
        ref = str(ref).strip()
        atr = item.get("actual_terminal_result") or {}
        answer = atr.get("answer")
        if answer is None:
            answer = item.get("response", "")
        response = str(answer).strip()
        citations = atr.get("citations") or []
        contexts = [str(c) for c in citations if c is not None]
        if not contexts:
            excerpt = atr.get("reasoning_excerpt")
            if excerpt:
                contexts = [str(excerpt).strip()]
        if not contexts:
            contexts = ["(no retrieved context recorded for this baseline row)"]
        if not q:
            raise ValueError(f"Row {index}: missing question / user_input")
        if not ref:
            raise ValueError(f"Row {index}: missing ground_truth / reference (needed for recall & answer correctness)")
        out.append(
            {
                "user_input": q,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": ref,
            }
        )
    return out


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array or object")
    return data


def build_or_load_before_ragas(force_from_assignment: bool = False) -> list[dict]:
    DATA_BEFORE.parent.mkdir(parents=True, exist_ok=True)
    if not force_from_assignment and DATA_BEFORE.is_file():
        rows = load_json(DATA_BEFORE)
        if rows and "user_input" in rows[0] and "response" in rows[0]:
            return rows
    if not ROOT_BEFORE.is_file():
        raise FileNotFoundError(f"Need baseline source at {ROOT_BEFORE} to build {DATA_BEFORE}")
    raw = load_json(ROOT_BEFORE)
    rows = assignment_rows_to_ragas(raw)
    DATA_BEFORE.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def validate_ragas_rows(rows: list[dict], label: str) -> None:
    for i, row in enumerate(rows, start=1):
        for key in ("user_input", "response", "retrieved_contexts", "reference"):
            if key not in row:
                raise ValueError(f"{label} row {i}: missing {key!r}")
        ctx = row["retrieved_contexts"]
        if not isinstance(ctx, list) or not ctx:
            raise ValueError(f"{label} row {i}: retrieved_contexts must be a non-empty list of strings")
        for j, c in enumerate(ctx):
            if not isinstance(c, str) or not str(c).strip():
                raise ValueError(f"{label} row {i}: retrieved_contexts[{j}] must be a non-empty string")
        if not str(row.get("reference", "")).strip():
            raise ValueError(f"{label} row {i}: reference must be non-empty for comparable runs")
        if not str(row.get("response", "")).strip():
            raise ValueError(f"{label} row {i}: response is empty")


def after_needs_regeneration(rows: list[dict]) -> bool:
    for row in rows:
        resp = str(row.get("response", ""))
        if "Pipeline error" in resp or "OPENAI_API_KEY" in resp:
            return True
        ctx = row.get("retrieved_contexts")
        if not isinstance(ctx, list) or len(ctx) == 0:
            return True
    return False


def maybe_regenerate_after() -> None:
    if not DATA_AFTER.is_file():
        print("data/after.json missing; run: python generate_after_dataset.py", file=sys.stderr)
        return
    rows = load_json(DATA_AFTER)
    if not after_needs_regeneration(rows):
        print("data/after.json looks valid; skipping regeneration.")
        return
    print("Regenerating data/after.json (extractive answers, Neo4j contexts; no answer LLM required) …")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "generate_after_dataset.py"),
            "--answer-mode",
            "extractive",
            "--input",
            str(DATA_BEFORE),
            "--output",
            str(DATA_AFTER),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


def ragas_python_executable() -> str:
    if os.getenv("RAGAS_PYTHON"):
        return os.environ["RAGAS_PYTHON"]
    if RAGAS_VENV_PYTHON.is_file():
        return str(RAGAS_VENV_PYTHON)
    return sys.executable


def run_ragas(path: Path, metrics_out: Path, log_out: Path) -> None:
    py = ragas_python_executable()
    cmd = [
        py,
        str(REPO_ROOT / "ragas_eval.py"),
        str(path),
        "--write-metrics",
        str(metrics_out),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    log_out.write_text(proc.stdout + (f"\n--- stderr ---\n{proc.stderr}" if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)


def metric_display_name(key: str) -> str:
    mapping = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision_without_reference": "Context Precision (without reference)",
        "llm_context_precision_without_reference": "Context Precision (without reference)",
        "context_recall": "Context Recall",
        "answer_correctness": "Answer Correctness",
    }
    return mapping.get(key, key.replace("_", " ").title())


def build_comparison(before: dict[str, float], after: dict[str, float]) -> dict:
    keys = sorted(set(before) | set(after))
    per_metric = []
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        delta = None
        if b is not None and a is not None:
            delta = round(a - b, 6)
        per_metric.append(
            {
                "metric_key": key,
                "metric": metric_display_name(key),
                "before": b,
                "after": a,
                "delta_after_minus_before": delta,
            }
        )
    return {"metrics": per_metric, "before_means": before, "after_means": after}


def render_report_lines(comp: dict) -> list[str]:
    lines: list[str] = []
    for m in comp["metrics"]:
        b, a, d = m["before"], m["after"], m["delta_after_minus_before"]
        name = m["metric"]
        lines.append(f"{name}:")
        lines.append(f"  Before: {b:.4f}" if isinstance(b, (int, float)) else "  Before: —")
        lines.append(f"  After: {a:.4f}" if isinstance(a, (int, float)) else "  After: —")
        if isinstance(d, (int, float)):
            sign = "+" if d >= 0 else ""
            lines.append(f"  Improvement: {sign}{d:.4f}")
        else:
            lines.append("  Improvement: —")
        lines.append("")
    return lines


def render_markdown(comp: dict) -> str:
    lines = [
        "# RAGAS: baseline vs improved",
        "",
        "| Metric | Before | After | Δ (After − Before) |",
        "|--------|--------|-------|----------------------|",
    ]
    for m in comp["metrics"]:
        b = m["before"]
        a = m["after"]
        d = m["delta_after_minus_before"]
        lines.append(
            f"| {m['metric']} | {b if b is not None else '—'} | {a if a is not None else '—'} | "
            f"{d if d is not None else '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_terminal_table(comp: dict) -> None:
    print()
    print("=== RAGAS COMPARISON RESULTS ===")
    print()
    print(f"{'Metric':<45} {'Before':>10} {'After':>10} {'Δ':>10}")
    print("-" * 77)
    for m in comp["metrics"]:
        b = m["before"]
        a = m["after"]
        d = m["delta_after_minus_before"]
        bs = f"{b:.4f}" if isinstance(b, (int, float)) else "—"
        a_s = f"{a:.4f}" if isinstance(a, (int, float)) else "—"
        d_s = f"{d:+.4f}" if isinstance(d, (int, float)) else "—"
        name = m["metric"]
        if len(name) > 44:
            name = name[:41] + "..."
        print(f"{name:<45} {bs:>10} {a_s:>10} {d_s:>10}")
    print()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    _ensure_openai_env()

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit("Set OPENAI_API_KEY (or OPENROUTER_API_KEY) for RAGAS evaluation.")

    before_rows = build_or_load_before_ragas(force_from_assignment=True)
    validate_ragas_rows(before_rows, "before")

    maybe_regenerate_after()
    after_rows = load_json(DATA_AFTER)
    validate_ragas_rows(after_rows, "after")

    for path, metrics_path, log_path in (
        (DATA_BEFORE, METRICS_BEFORE, BEFORE_TXT),
        (DATA_AFTER, METRICS_AFTER, AFTER_TXT),
    ):
        print(f"Running ragas_eval.py on {path.name} …")
        run_ragas(path, metrics_path, log_path)

    before_m = json.loads(METRICS_BEFORE.read_text(encoding="utf-8"))
    after_m = json.loads(METRICS_AFTER.read_text(encoding="utf-8"))
    comp = build_comparison(before_m, after_m)
    md = render_markdown(comp)
    report_lines = "\n".join(render_report_lines(comp))

    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Mean RAGAS scores; before = baseline assignment outputs, after = Workstream 3 Neo4j pipeline.",
        "before_results_file": str(BEFORE_TXT.relative_to(REPO_ROOT)),
        "after_results_file": str(AFTER_TXT.relative_to(REPO_ROOT)),
        "comparison": comp,
        "markdown_table": md,
        "report_text": report_lines,
    }
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    COMPARISON_MD.write_text(report_lines + "\n" + md, encoding="utf-8")

    print_terminal_table(comp)
    print(f"Wrote {COMPARISON_JSON}")
    print(f"Wrote {COMPARISON_MD}")
    print(f"Full logs: {BEFORE_TXT.name}, {AFTER_TXT.name}")


if __name__ == "__main__":
    main()
