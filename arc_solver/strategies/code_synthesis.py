"""
Code synthesis ARC solver:
  1. Ask Claude to analyze the pattern (chain-of-thought)
  2. Ask Claude to write a verified Python transform function
  3. Mechanically verify against all training examples
  4. Retry with targeted error feedback if needed

Uses prompt caching to reduce cost on retries.
"""
import re
import time
import copy
import traceback
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

SYSTEM_PROMPT = """\
You are an expert programmer and abstract pattern recognition specialist solving ARC \
(Abstraction and Reasoning Corpus) puzzles. Your strength is identifying precise \
transformation rules and implementing them as correct Python code. \
You always think carefully before coding and write simple, readable solutions."""

TASK_CONTEXT = """\
ARC puzzles: Each puzzle shows 2-10 input→output grid pairs. \
Grids are 2D arrays of integers 0-9 (representing colors). \
Your goal is to find the exact transformation rule and implement it as \
`def transform(grid: list[list[int]]) -> list[list[int]]:` \
using only Python built-ins (no imports needed).

Common transformation types (not exhaustive):
- Geometric: rotation, reflection, tiling, scaling, cropping, transposing
- Color: remapping, filling, pattern-based recoloring
- Structural: symmetry completion, pattern repetition, object manipulation
- Logical: masking, overlay, region detection, connected components"""


def _grid_visual(grid: Grid, label: str = '') -> str:
    lines = []
    if label:
        lines.append(label)
    for row in grid:
        lines.append(' '.join(COLOR_CHARS[v] for v in row))
    return '\n'.join(lines)


def _make_task_block(task) -> str:
    parts = [
        f"Colors: {', '.join(f'{v}={COLOR_CHARS[v]}({n})' for v, n in COLOR_NAMES.items())}",
        "",
        "=== TRAINING EXAMPLES ===",
    ]
    for i, pair in enumerate(task['train']):
        inp, out = pair['input'], pair['output']
        parts.append(f"\nExample {i+1}:")
        parts.append(_grid_visual(inp, f"Input ({len(inp)}×{len(inp[0])})"))
        parts.append(_grid_visual(out, f"Output ({len(out)}×{len(out[0])})"))
    return '\n'.join(parts)


def _analysis_prompt(task_block: str) -> str:
    return f"""{task_block}

=== ANALYSIS ===
Before writing code, analyze the transformation:
1. How do the output dimensions relate to input dimensions?
2. What happens to the colors?
3. What is the precise rule in plain English?
4. Are there any edge cases to handle?

Be concise and precise. This analysis will guide your code."""


def _coding_prompt(task_block: str, analysis: str, error_hint: str = None) -> str:
    parts = [task_block, ""]

    if analysis:
        parts.append(f"=== PATTERN ANALYSIS ===\n{analysis}\n")

    if error_hint:
        parts.append(f"=== PREVIOUS ERROR ===\n{error_hint}\n")
        parts.append("Fix the error and write a corrected function.\n")

    parts.append("""=== IMPLEMENT ===
Write `def transform(grid: list[list[int]]) -> list[list[int]]:` that passes all training examples.
Rules:
- No imports (built-ins only: range, len, zip, enumerate, list, etc.)
- Return a new 2D list, do not mutate input
- All values must be integers 0-9

Output ONLY the function in a ```python ... ``` block.""")

    return '\n'.join(parts)


def _extract_code(text: str) -> Optional[str]:
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'(def transform\s*\(.*?)(?=\n(?:def |\Z))', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _execute(code: str, grid: Grid) -> Tuple[Optional[Grid], Optional[str]]:
    ns = {}
    try:
        exec(compile(code, '<transform>', 'exec'), ns)
    except SyntaxError as e:
        return None, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return None, f"Compile error: {e}"

    fn = ns.get('transform')
    if fn is None:
        return None, "No 'transform' function defined"

    try:
        result = fn(copy.deepcopy(grid))
        if not isinstance(result, list) or not result or not isinstance(result[0], list):
            return None, "Must return a 2D list"
        for r, row in enumerate(result):
            for c, v in enumerate(row):
                if not isinstance(v, int) or not (0 <= v <= 9):
                    return None, f"Cell [{r}][{c}]={v!r} is not int 0-9"
        return result, None
    except Exception:
        return None, f"RuntimeError:\n{traceback.format_exc(limit=5)}"


def _grids_equal(a: Grid, b: Grid) -> bool:
    return len(a) == len(b) and all(list(ra) == list(rb) for ra, rb in zip(a, b))


def _verify(code: str, train_pairs) -> Tuple[bool, Optional[str]]:
    for i, pair in enumerate(train_pairs):
        result, err = _execute(code, pair['input'])
        if err:
            return False, err
        exp = pair['output']
        if not _grids_equal(result, exp):
            got_dim = f"{len(result)}×{len(result[0])}"
            exp_dim = f"{len(exp)}×{len(exp[0])}"
            if got_dim != exp_dim:
                hint = f"Example {i+1}: wrong size — got {got_dim}, expected {exp_dim}"
            else:
                # Find first differing cell
                diff_cells = [(r, c, result[r][c], exp[r][c])
                              for r in range(len(exp)) for c in range(len(exp[r]))
                              if result[r][c] != exp[r][c]]
                cell_info = ', '.join(f"[{r},{c}]:{g}≠{e}" for r, c, g, e in diff_cells[:5])
                hint = f"Example {i+1}: wrong values — {cell_info}"
            return False, hint
    return True, None


def solve(task, n_attempts: int = 12, model: str = 'claude-opus-4-8') -> Optional[Grid]:
    """
    Try to synthesize a verified transform function.
    Uses a two-step approach: analyze pattern, then code it.
    """
    client = anthropic.Anthropic()
    train_pairs = task['train']
    test_input = task['test'][0]['input']

    task_block = _make_task_block(task)

    # Step 1: Get analysis (once, cached)
    analysis = None
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TASK_CONTEXT,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": _analysis_prompt(task_block),
                        }
                    ]
                }
            ]
        )
        analysis = resp.content[0].text.strip()
    except Exception:
        pass  # Proceed without analysis

    # Step 2: Generate and verify code
    last_error = None
    for attempt in range(n_attempts):
        temp = 0.1 if attempt == 0 else min(0.2 + attempt * 0.1, 1.0)
        prompt = _coding_prompt(task_block, analysis, last_error if attempt > 0 else None)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TASK_CONTEXT,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temp,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            code = _extract_code(resp.content[0].text)
            if code is None:
                last_error = "No ```python``` code block found. Output only the function."
                continue

            ok, err = _verify(code, train_pairs)
            if ok:
                result, exec_err = _execute(code, test_input)
                if result is not None:
                    return result
                last_error = f"Execution on test input failed: {exec_err}"
            else:
                last_error = err

        except anthropic.RateLimitError:
            time.sleep(30)
        except anthropic.APIError:
            time.sleep(5)

        if attempt < n_attempts - 1:
            time.sleep(0.3)

    return None
