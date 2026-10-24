"""
The corpus's parallel execution. Worker processes start by spawn, so none inherits a parent's threads (DuckDB's,
tiktoken's, the data crate's) or state. ordered() keeps a bounded window of tasks in flight and yields their results
in task order, as Python 3.14's Executor.map(buffersize=window) does (the project runs on 3.12). MEMORY bounds what a
job holds at once.
"""

import collections
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from pathlib import Path

SPAWN = multiprocessing.get_context("spawn")


def memory() -> int:
    """The memory this process may use: the machine's, or its container's limit (cgroup v2 memory.max) when lower."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = Path("/sys/fs/cgroup/memory.max")
    text = limit.read_text().strip() if limit.exists() else "max"
    return physical if text == "max" else min(physical, int(text))


# Half of it: the lower end of DuckDB's guidance, as some operations bypass the buffer manager's limit
# (duckdb.org/docs/lts/guides/troubleshooting/oom_errors), and other jobs share the unified memory
MEMORY = memory() // 2


def ordered[T, R](pool: Executor, fn: Callable[[T], R], tasks: Iterable[T], window: int) -> Iterator[R]:
    """fn of each task in task order, at most `window` submitted and not yet yielded, so no more results wait in
    memory than the window holds and tasks are submitted only as results are taken."""
    pending: collections.deque[Future[R]] = collections.deque()
    for task in tasks:
        pending.append(pool.submit(fn, task))
        if len(pending) == window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()
