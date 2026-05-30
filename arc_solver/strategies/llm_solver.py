"""
LLM-based ARC solver using Claude API.
Uses chain-of-thought reasoning + multiple attempts with majority voting.
"""
import json
import os
import time
from collections import Counter
from typing import List, Optional, Tuple

import anthropic

Grid = List[List[int]]

COLOR_NAMES = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'grey', 6: 'magenta', 7: 'orange', 8: 'azure', 9: 'maroon'
}

COLOR_CHARS = {
    0: '.', 1: 'B', 2: 'R', 3: 'G', 4: 'Y',
    5: 'S', 6: 'M', 7: 'O', 8: 'A', 9: 'N'
}


def grid_to_str(grid: Grid) -> str:
    lines = []
    for row in grid:
        lines.append(' '.join(COLOR_CHARS[v] for v in row))
    return '\n'.join(lines)


def grid_to_json_str(grid: Grid) -> str:
    return json.dumps(grid)


def format_task_prompt(task) -> str:
    train_pairs = task['train']
    test_input = task['test'][0]['input']

    legend = "Color legend: " + ", ".join(f"{v}={COLOR_CHARS[v]}({n})" for v, n in COLOR_NAMES.items())

    parts = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.",
        "Each puzzle has a hidden transformation rule. Study the training examples and apply the rule to the test input.",
        "",
        legend,
        "",
        "=== TRAINING EXAMPLES ===",
    ]

    for i, pair in enumerate(train_pairs):
        inp = pair['input']
        out = pair['output']
        parts.append(f"\nExample {i+1}:")
        parts.append(f"Input ({len(inp)}x{len(inp[0])}):")
        parts.append(grid_to_str(inp))
        parts.append(f"Input (JSON): {grid_to_json_str(inp)}")
        parts.append(f"Output ({len(out)}x{len(out[0])}):")
        parts.append(grid_to_str(out))
        parts.append(f"Output (JSON): {grid_to_json_str(out)}")

    parts.append("\n=== TEST INPUT ===")
    parts.append(f"Input ({len(test_input)}x{len(test_input[0])}):")
    parts.append(grid_to_str(test_input))
    parts.append(f"Input (JSON): {grid_to_json_str(test_input)}")

    parts.append("""
=== YOUR TASK ===
1. Analyze the training examples carefully. Look for patterns in:
   - Size changes (input vs output dimensions)
   - Color transformations
   - Geometric operations (rotation, reflection, tiling, scaling)
   - Structural patterns (symmetry, repetition, object manipulation)
   - Relationships between objects in the grid

2. State the transformation rule clearly and concisely.

3. Apply the rule to the test input and produce the output grid.

4. Output the result as a valid JSON 2D array (list of lists of integers 0-9).
   Format your final answer exactly like this:
   ANSWER: [[row1], [row2], ...]

Think step by step. Be precise about the output dimensions and values.""")

    return '\n'.join(parts)


def parse_answer(text: str) -> Optional[Grid]:
    """Extract and parse the grid from LLM response."""
    # Look for ANSWER: [...] pattern
    import re

    # Try ANSWER: pattern first
    match = re.search(r'ANSWER:\s*(\[\[.*?\]\])', text, re.DOTALL)
    if match:
        try:
            grid = json.loads(match.group(1))
            if _valid_grid(grid):
                return grid
        except json.JSONDecodeError:
            pass

    # Try to find any 2D JSON array in the response (last one wins)
    pattern = r'(\[\s*\[[\d,\s\[\]]+\]\s*\])'
    matches = re.findall(pattern, text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            grid = json.loads(candidate)
            if _valid_grid(grid):
                return grid
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _valid_grid(grid) -> bool:
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if not isinstance(grid[0], list) or len(grid[0]) == 0:
        return False
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True


def _grid_to_tuple(grid: Grid) -> tuple:
    return tuple(tuple(row) for row in grid)


def solve(task, n_attempts: int = 8, model: str = 'claude-opus-4-8') -> Optional[Grid]:
    """
    Solve using Claude API with multiple attempts + majority voting.
    Returns the most common valid answer, or None if all fail.
    """
    client = anthropic.Anthropic()
    prompt = format_task_prompt(task)

    results: List[Grid] = []

    for attempt in range(n_attempts):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.7 if attempt > 0 else 0.2,  # lower temp for first attempt
                system=(
                    "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. "
                    "You excel at identifying abstract transformation rules from examples and applying them precisely. "
                    "Always output your final answer as a JSON array."
                ),
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            grid = parse_answer(text)
            if grid is not None:
                results.append(grid)
        except anthropic.RateLimitError:
            time.sleep(30)
        except anthropic.APIError as e:
            time.sleep(5)
            continue

        # Small delay to avoid rate limits
        if attempt < n_attempts - 1:
            time.sleep(0.5)

    if not results:
        return None

    # Majority voting
    counts = Counter(_grid_to_tuple(g) for g in results)
    best_tuple, best_count = counts.most_common(1)[0]

    # Return best result; if tied, return first occurrence
    return [list(row) for row in best_tuple]


def solve_with_confidence(task, n_attempts: int = 8, model: str = 'claude-opus-4-8') -> Tuple[Optional[Grid], float]:
    """Returns (grid, confidence) where confidence = vote_fraction."""
    client = anthropic.Anthropic()
    prompt = format_task_prompt(task)

    results: List[Grid] = []

    for attempt in range(n_attempts):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.7 if attempt > 0 else 0.2,
                system=(
                    "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. "
                    "You excel at identifying abstract transformation rules from examples and applying them precisely. "
                    "Always output your final answer as a JSON array."
                ),
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            grid = parse_answer(text)
            if grid is not None:
                results.append(grid)
        except anthropic.RateLimitError:
            time.sleep(30)
        except anthropic.APIError:
            time.sleep(5)
            continue

        if attempt < n_attempts - 1:
            time.sleep(0.5)

    if not results:
        return None, 0.0

    counts = Counter(_grid_to_tuple(g) for g in results)
    best_tuple, best_count = counts.most_common(1)[0]
    confidence = best_count / n_attempts
    return [list(row) for row in best_tuple], confidence
