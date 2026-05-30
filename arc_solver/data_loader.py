import json
import os
from typing import List, Dict, Tuple

Grid = List[List[int]]
Pair = Dict[str, Grid]
Task = Dict[str, List[Pair]]

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'ARC-AGI-2', 'data')


def load_task(path: str) -> Task:
    with open(path) as f:
        return json.load(f)


def load_split(split: str) -> List[Tuple[str, Task]]:
    """split: 'training' or 'evaluation'"""
    dir_path = os.path.join(DATA_ROOT, split)
    tasks = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.json'):
            task_id = fname[:-5]
            path = os.path.join(dir_path, fname)
            tasks.append((task_id, load_task(path)))
    return tasks


def grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if list(row_a) != list(row_b):
            return False
    return True
