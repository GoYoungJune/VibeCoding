"""
Analyze which rule-based patterns cover the training set,
and identify the hardest tasks for LLM focus.

Usage: python -m arc_solver.analyze
"""
import os
import sys
from collections import Counter

from .data_loader import load_split
from .strategies.rule_based import SINGLE_PRIMITIVES, _TWO_STEP_FNS, _matches_all, _build_two_step_fns, _apply_safe
from .data_loader import grids_equal


def analyze_coverage(split: str = 'training', limit: int = None):
    tasks = load_split(split)
    if limit:
        tasks = tasks[:limit]

    rule_solved = []
    unsolved = []

    for task_id, task in tasks:
        solved = False
        for name, fn in SINGLE_PRIMITIVES:
            if _matches_all(fn, task['train']):
                rule_solved.append((task_id, name))
                solved = True
                break

        if not solved:
            two_step = _build_two_step_fns()
            for name, fn in two_step:
                if _matches_all(fn, task['train']):
                    rule_solved.append((task_id, name))
                    solved = True
                    break

        if not solved:
            unsolved.append(task_id)

    total = len(tasks)
    print(f"\nRule-based coverage on '{split}': {len(rule_solved)}/{total} = {len(rule_solved)/total*100:.1f}%")
    print(f"Unsolved by rules: {len(unsolved)}")

    # Which rules fire most?
    rule_counter = Counter(name for _, name in rule_solved)
    print("\nTop rules:")
    for rule, count in rule_counter.most_common(15):
        print(f"  {rule:40s} {count:4d}")

    return rule_solved, unsolved


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'training'
    analyze_coverage(split)
