"""Spawned worker processes (none inherits the parent's threads or state), a bounded ordered map, the memory budget."""

import collections
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, wait
from pathlib import Path

SPAWN = multiprocessing.get_context("spawn")


def memory() -> int:
    """This process's memory: the machine's, or its container's cgroup v2 limit when lower ("max": no limit)."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    return physical if text == "max" else min(physical, int(text))


# Half: DuckDB's guidance, as some operations bypass its memory limit (DuckDB documentation, Guides, Troubleshooting,
# "Out of Memory Errors"); the rest is left to other processes
MEMORY = memory() // 2


def ordered[T, R](pool: Executor, fn: Callable[[T], R], tasks: Iterable[T], window: int) -> Iterator[R]:
    """fn of each task, in task order, with at most `window` results outstanding. Closed early, it cancels the tasks
    not started and waits for the running ones."""
    pending: collections.deque[Future[R]] = collections.deque()
    try:
        for task in tasks:
            pending.append(pool.submit(fn, task))
            if len(pending) == window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        wait(pending)
