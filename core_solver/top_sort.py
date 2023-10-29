from typing import List, Tuple

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
        pot_wu = (weights[u], u)
        if pot_wu in s:
            s.remove(pot_wu)

    while len(s) > 0:
        w, v = s.pop()
        l.append(v)
        ls.add(v)

        et = []
        for u, v, c in es:
            pot_wu = (weights[u], u)
            if v not in ls:
                et.append((u, v, c))
                if pot_wu in s:
                    s.remove(pot_wu)
            else:
                s.add(pot_wu)
        es = et

    assert len(l) == len(weights), (len(l), len(weights))
    return l
