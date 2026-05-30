"""
Rule-based ARC solver: exhaustive search over primitive transformation compositions.
Covers the most common ARC-AGI patterns.
"""
import itertools
from typing import List, Callable, Optional, Tuple

Grid = List[List[int]]


# ── Primitive transformations ──────────────────────────────────────────────────

def rot90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]

def rot180(g: Grid) -> Grid:
    return rot90(rot90(g))

def rot270(g: Grid) -> Grid:
    return rot90(rot90(rot90(g)))

def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    return g[::-1]

def transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]

def anti_transpose(g: Grid) -> Grid:
    return flip_h(transpose(flip_h(flip_v(g))))

def identity(g: Grid) -> Grid:
    return [list(row) for row in g]

def scale2(g: Grid) -> Grid:
    result = []
    for row in g:
        new_row = []
        for v in row:
            new_row.extend([v, v])
        result.append(new_row)
        result.append(list(new_row))
    return result

def scale3(g: Grid) -> Grid:
    result = []
    for row in g:
        new_row = []
        for v in row:
            new_row.extend([v, v, v])
        for _ in range(3):
            result.append(list(new_row))
    return result

def tile_h2(g: Grid) -> Grid:
    return [row + row for row in g]

def tile_h3(g: Grid) -> Grid:
    return [row + row + row for row in g]

def tile_v2(g: Grid) -> Grid:
    return list(g) + list(g)

def tile_v3(g: Grid) -> Grid:
    return list(g) + list(g) + list(g)

def tile_2x2(g: Grid) -> Grid:
    row_double = [row + row for row in g]
    return row_double + row_double

def tile_3x3(g: Grid) -> Grid:
    row_triple = [row + row + row for row in g]
    return row_triple + row_triple + row_triple

def tile_mirror_h(g: Grid) -> Grid:
    """Tile horizontally with horizontal mirror: [A | flip(A)]"""
    return [row + row[::-1] for row in g]

def tile_mirror_v(g: Grid) -> Grid:
    """Tile vertically with vertical mirror: [A / flip(A)]"""
    return list(g) + g[::-1]

def tile_mirror_2x2(g: Grid) -> Grid:
    """2x2 tiling with mirrors: A | fh(A) / fv(A) | rot180(A)"""
    top = [row + row[::-1] for row in g]
    bottom = [row + row[::-1] for row in g[::-1]]
    return top + bottom

def tile_mirror_h3(g: Grid) -> Grid:
    """3-wide mirror tile: A | flip(A) | A"""
    return [row + row[::-1] + row for row in g]

def tile_mirror_v3(g: Grid) -> Grid:
    """3-tall mirror tile: A / flip(A) / A"""
    return list(g) + g[::-1] + list(g)

def tile_h2_flip(g: Grid) -> Grid:
    """A | rot180(A)"""
    r = rot180(g)
    return [gr + rr for gr, rr in zip(g, r)]

def tile_v2_flip(g: Grid) -> Grid:
    """A / rot180(A)"""
    return list(g) + rot180(g)

def color_invert(g: Grid) -> Grid:
    """Invert all non-zero colors (treat 0 as background, leave unchanged)."""
    all_vals = sorted({v for row in g for v in row if v != 0})
    if not all_vals:
        return g
    mapping = {}
    for i, v in enumerate(all_vals):
        mapping[v] = all_vals[-(i + 1)]
    return [[mapping.get(v, v) for v in row] for row in g]


# ── Composable primitive set ───────────────────────────────────────────────────

SINGLE_PRIMITIVES: List[Tuple[str, Callable]] = [
    ('identity', identity),
    ('rot90', rot90),
    ('rot180', rot180),
    ('rot270', rot270),
    ('flip_h', flip_h),
    ('flip_v', flip_v),
    ('transpose', transpose),
    ('anti_transpose', anti_transpose),
    ('scale2', scale2),
    ('scale3', scale3),
    ('tile_h2', tile_h2),
    ('tile_h3', tile_h3),
    ('tile_v2', tile_v2),
    ('tile_v3', tile_v3),
    ('tile_2x2', tile_2x2),
    ('tile_3x3', tile_3x3),
    ('tile_mirror_h', tile_mirror_h),
    ('tile_mirror_v', tile_mirror_v),
    ('tile_mirror_2x2', tile_mirror_2x2),
    ('tile_mirror_h3', tile_mirror_h3),
    ('tile_mirror_v3', tile_mirror_v3),
    ('tile_h2_flip', tile_h2_flip),
    ('tile_v2_flip', tile_v2_flip),
    ('color_invert', color_invert),
]

# Geometric-only prims for composition (avoid chaining two tiles)
GEO_PRIMS: List[Tuple[str, Callable]] = [
    ('rot90', rot90),
    ('rot180', rot180),
    ('rot270', rot270),
    ('flip_h', flip_h),
    ('flip_v', flip_v),
    ('transpose', transpose),
    ('anti_transpose', anti_transpose),
    ('color_invert', color_invert),
]

TILE_PRIMS: List[Tuple[str, Callable]] = [
    ('tile_h2', tile_h2),
    ('tile_h3', tile_h3),
    ('tile_v2', tile_v2),
    ('tile_v3', tile_v3),
    ('tile_2x2', tile_2x2),
    ('tile_3x3', tile_3x3),
    ('tile_mirror_h', tile_mirror_h),
    ('tile_mirror_v', tile_mirror_v),
    ('tile_mirror_2x2', tile_mirror_2x2),
    ('tile_mirror_h3', tile_mirror_h3),
    ('tile_mirror_v3', tile_mirror_v3),
    ('tile_h2_flip', tile_h2_flip),
    ('tile_v2_flip', tile_v2_flip),
    ('scale2', scale2),
    ('scale3', scale3),
]


def _apply_safe(fn: Callable, g: Grid) -> Optional[Grid]:
    try:
        result = fn(g)
        if not result or not result[0]:
            return None
        return result
    except Exception:
        return None


def _grids_match(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, rb if False else b):
        if list(ra) != list(rb):
            return False
    return True


def _matches_all(fn: Callable, train_pairs) -> bool:
    for pair in train_pairs:
        result = _apply_safe(fn, pair['input'])
        if result is None:
            return False
        if not _grids_match(result, pair['output']):
            return False
    return True


def _build_two_step_fns():
    """geo then tile, or geo then geo"""
    fns = []
    for gname, gfn in GEO_PRIMS:
        for tname, tfn in TILE_PRIMS:
            def make_fn(g, t):
                def fn(grid):
                    return t(g(grid))
                return fn
            fns.append((f'{gname}+{tname}', make_fn(gfn, tfn)))
        for gname2, gfn2 in GEO_PRIMS:
            if gname == gname2:
                continue
            def make_fn2(g1, g2):
                def fn(grid):
                    return g2(g1(grid))
                return fn
            fns.append((f'{gname}+{gname2}', make_fn2(gfn, gfn2)))
    return fns


_TWO_STEP_FNS = None


def solve(task) -> Optional[Grid]:
    """
    Try to solve an ARC task using rule-based transformations.
    Returns the predicted output grid for the first test input, or None.
    """
    global _TWO_STEP_FNS
    train_pairs = task['train']
    test_input = task['test'][0]['input']

    # Stage 1: single primitives
    for name, fn in SINGLE_PRIMITIVES:
        if _matches_all(fn, train_pairs):
            return _apply_safe(fn, test_input)

    # Stage 2: two-step compositions
    if _TWO_STEP_FNS is None:
        _TWO_STEP_FNS = _build_two_step_fns()

    for name, fn in _TWO_STEP_FNS:
        if _matches_all(fn, train_pairs):
            return _apply_safe(fn, test_input)

    return None
