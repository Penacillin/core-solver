from typing import Iterable, List, Set, Tuple

from sortedcontainers import SortedSet

Vertex = int
Edge = Tuple[Vertex, Vertex, float]


def top_sort(weights: List[int], es: List[Edge]):
    s = SortedSet()
    for v, w in enumerate(weights):
        s.add((w, v))

    l: List[Vertex] = []
    ls = set()

    for u, v, c in es:
        s.remove((weights[u], u))

    while len(s) > 0:
        w, v = s.pop()
        l.append(v)
        ls.add(v)

        et = []
        for u, v, c in es:
            if v not in ls:
                et.append((u, v, c))
            else:
                s.add((weights[u], u))
        es = et

    assert len(l) == len(weights), (len(l), len(weights))
    return l
