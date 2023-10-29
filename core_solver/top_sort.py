from typing import Iterable, List, Set, Tuple

Vertex = int
Edge = Tuple[Vertex, Vertex, float]


def top_sort(vs: Iterable[int], es: List[Edge]):
    s = set(vs)
    l: List[Vertex] = []
    ls = set()

    for u, v, c in es:
        s.remove(u)

    while len(s) > 0:
        v = s.pop()
        l.append(v)
        ls.add(v)

        et = []
        for u, v, c in es:
            if v not in ls:
                et.append((u, v, c))
                if u in s:
                    s.remove(u)
            else:
                s.add(u)
        es = et

    assert len(l) == len(vs)
    return l
