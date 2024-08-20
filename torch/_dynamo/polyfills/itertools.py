"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator, TypeVar

from ..decorators import substitute_in_graph


_T = TypeVar("_T")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain
@substitute_in_graph(itertools.chain.__new__)  # type: ignore[arg-type]
def chain___new__(
    cls: type[itertools.chain[_T]],
    *iterables: Iterable[_T],
) -> Iterator[_T]:
    assert cls is itertools.chain

    for iterable in iterables:
        yield from iterable


@substitute_in_graph(itertools.chain.from_iterable)  # type: ignore[arg-type]
def chain_from_iterable(iterable: Iterable[Iterable[_T]], /) -> Iterator[_T]:
    return itertools.chain(*iterable)


# Reference: https://docs.python.org/3/library/itertools.html#itertools.count
@substitute_in_graph(itertools.count.__new__)  # type: ignore[arg-type]
def count___new__(
    cls: type[itertools.count[_T]],  # type: ignore[type-var]
    start: _T = 0,  # type: ignore[assignment]
    step: _T = 1,  # type: ignore[assignment]
) -> Iterator[_T]:
    assert cls is itertools.count

    n = start
    while True:
        yield n
        n += step  # type: ignore[operator]


# Reference: https://docs.python.org/3/library/itertools.html#itertools.islice
@substitute_in_graph(itertools.islice.__new__)  # type: ignore[arg-type]
def islice___new__(
    cls: type[itertools.islice[_T]],
    iterable: Iterable[_T],
    *args: int | None,
) -> Iterator[_T]:
    assert cls is itertools.islice

    s = slice(*args)
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step
    if start < 0 or (stop is not None and stop < 0) or step <= 0:
        raise ValueError(
            "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.",
        )

    indices = itertools.count() if stop is None else range(max(start, stop))
    next_i = start
    for i, element in zip(indices, iterable):
        if i == next_i:
            yield element
            next_i += step


# Reference: https://docs.python.org/3/library/itertools.html#itertools.tee
@substitute_in_graph(itertools.tee)
def tee(iterable: Iterable[_T], n: int = 2, /) -> tuple[Iterator[_T], ...]:
    iterator = iter(iterable)
    shared_link = [None, None]

    def _tee(link) -> Iterator[_T]:  # type: ignore[no-untyped-def]
        try:
            while True:
                if link[1] is None:
                    link[0] = next(iterator)
                    link[1] = [None, None]
                value, link = link
                yield value
        except StopIteration:
            return

    return tuple(_tee(shared_link) for _ in range(n))
