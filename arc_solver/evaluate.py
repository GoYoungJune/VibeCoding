"""
Evaluation script.

Usage:
    # Rule-based only (free, ~2% accuracy baseline)
    python -m arc_solver.evaluate --split evaluation --no-llm

    # Full hybrid (needs ANTHROPIC_API_KEY, targets 85%)
    python -m arc_solver.evaluate --split evaluation

    # Quick test with first N tasks
    python -m arc_solver.evaluate --split training --limit 20

    # Save results JSON
    python -m arc_solver.evaluate --split evaluation --save results.json
"""
import argparse
import json
import os
import sys
import time

from .data_loader import load_split
from .solver import solve, evaluate_task


def run_evaluation(
    split: str = 'evaluation',
    use_llm: bool = True,
    llm_attempts: int = 12,
    llm_model: str = 'claude-opus-4-8',
    limit: int = None,
    verbose: bool = True,
    save_results: str = None,
) -> float:

    tasks = load_split(split)
    if limit:
        tasks = tasks[:limit]

    total = len(tasks)
    correct = 0
    rule_based_wins = 0
    code_synth_wins = 0
    llm_direct_wins = 0
    results = []

    print(f"\n{'='*60}")
    print(f"Split: '{split}'  ({total} tasks)")
    print(f"Model: {llm_model if use_llm else 'rule-based only'}")
    print(f"LLM attempts per task: {llm_attempts}")
    print(f"{'='*60}\n")

    start = time.time()

    for i, (task_id, task) in enumerate(tasks):
        t0 = time.time()
        prediction = solve(
            task,
            use_llm=use_llm,
            llm_attempts=llm_attempts,
            llm_model=llm_model,
            verbose=verbose,
        )
        ok = evaluate_task(task, prediction)
        correct += int(ok)
        elapsed = time.time() - t0

        status = '✓' if ok else '✗'
        running_acc = correct / (i + 1) * 100
        eta = (time.time() - start) / (i + 1) * (total - i - 1)

        print(f"[{i+1:4d}/{total}] {task_id} {status}  {elapsed:5.1f}s  "
              f"acc={running_acc:.1f}%  ETA={eta/60:.0f}m")

        results.append({
            'task_id': task_id,
            'correct': ok,
            'prediction': prediction,
            'expected': task['test'][0]['output'],
        })

    total_time = time.time() - start
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {correct}/{total} = {accuracy*100:.2f}%")
    print(f"Target 85%: {'✓ ACHIEVED' if accuracy >= 0.85 else f'✗ need {int(0.85*total)-correct} more'}")
    print(f"Total time: {total_time/60:.1f}min  ({total_time/total:.1f}s/task avg)")
    print(f"{'='*60}\n")

    if save_results:
        with open(save_results, 'w') as f:
            json.dump({
                'split': split,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'model': llm_model,
                'llm_attempts': llm_attempts,
                'tasks': results,
            }, f, indent=2)
        print(f"Results saved to {save_results}")

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ARC-AGI-2 solver')
    parser.add_argument('--split', default='evaluation', choices=['training', 'evaluation'])
    parser.add_argument('--no-llm', action='store_true', help='Rule-based only')
    parser.add_argument('--llm-attempts', type=int, default=12,
                        help='LLM attempts per task (default: 12, more=better accuracy but slower)')
    parser.add_argument('--llm-model', default='claude-opus-4-8',
                        choices=['claude-opus-4-8', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save', default=None, help='Save results to JSON')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if not args.no_llm and not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: Set ANTHROPIC_API_KEY to use LLM solving.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY='sk-ant-...'", file=sys.stderr)
        print("  Or use --no-llm for rule-based baseline only.", file=sys.stderr)
        sys.exit(1)

    run_evaluation(
        split=args.split,
        use_llm=not args.no_llm,
        llm_attempts=args.llm_attempts,
        llm_model=args.llm_model,
        limit=args.limit,
        verbose=not args.quiet,
        save_results=args.save,
    )
