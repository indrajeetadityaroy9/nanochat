"""
The corpus's parallel execution. Worker processes start by spawn, so none inherits a parent's threads (DuckDB's,
tiktoken's, the data crate's) or state. ordered() keeps a bounded window of tasks in flight and yields their results
in task order, as Python 3.14's Executor.map(buffersize=window) does (the project runs on 3.12).
"""

import collections
import multiprocessing
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future

SPAWN = multiprocessing.get_context("spawn")


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
