"""
Hybrid ARC solver.
Strategy order:
  1. Rule-based  (free, instant)
  2. Code synthesis via LLM  (verified, reliable)
  3. Direct LLM output  (fallback for tasks code synthesis can't crack)
"""
import os
from typing import Optional

from .strategies import rule_based
from .data_loader import grids_equal

Grid = list


def solve(
    task,
    use_llm: bool = True,
    llm_attempts: int = 10,
    llm_model: str = 'claude-opus-4-8',
    verbose: bool = False,
) -> Optional[Grid]:
    """
    Solve a task. Returns predicted output for the first test pair.
    """
    # ── Stage 1: Rule-based ──────────────────────────────────────────────────
    result = rule_based.solve(task)
    if result is not None:
        if verbose:
            print('  [rule-based] ✓')
        return result

    if not use_llm:
        return None

    # ── Stage 2: Code synthesis (LLM writes + verifies Python code) ──────────
    from .strategies import code_synthesis
    if verbose:
        print(f'  [code-synth] trying {llm_attempts} attempts…')
    result = code_synthesis.solve(task, n_attempts=llm_attempts, model=llm_model)
    if result is not None:
        if verbose:
            print('  [code-synth] ✓')
        return result

    # ── Stage 3: Direct LLM output (multi-attempt + voting) ─────────────────
    from .strategies import llm_solver
    if verbose:
        print(f'  [llm-direct] trying {llm_attempts} attempts…')
    result = llm_solver.solve(task, n_attempts=llm_attempts, model=llm_model)
    if result is not None:
        if verbose:
            print('  [llm-direct] ✓')
    return result


def evaluate_task(task, prediction: Optional[Grid]) -> bool:
    if prediction is None:
        return False
    expected = task['test'][0]['output']
    return grids_equal(prediction, expected)
