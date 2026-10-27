"""Spawned worker processes (none inherits the parent's threads or state), a bounded ordered map, the memory budget."""

import collections
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, wait
from pathlib import Path

SPAWN = multiprocessing.get_context("spawn")


def memory() -> int:
    """This process's memory: the machine's, or its cgroup v2 memory.max when lower."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    return physical if text == "max" else min(physical, int(text))


# Half: DuckDB documentation, Guides, Performance, "Out-of-Memory Issues" (some operations bypass its limit)
MEMORY = memory() // 2


def ordered[T, R](pool: Executor, fn: Callable[[T], R], tasks: Iterable[T], window: int) -> Iterator[R]:
    """fn of each task, in task order, at most `window` outstanding; closed early, it cancels unstarted tasks and waits
    for running ones."""
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
