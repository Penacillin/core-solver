import copy
import dataclasses
import heapq
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import numpy as np
from ortools.sat.python import cp_model

from .top_sort import top_sort

Vertex = int
Edge = Tuple[Vertex, Vertex, float]
EdgeCon = Tuple[Vertex, float]


@dataclass
class SolveState:
    task_i: int
    curr_places: np.ndarray
    remaining_caps: np.ndarray
    # cp_model: cp_model.CpModel


@dataclass(order=True)
class PqEntry:
    cost: float
    state: SolveState = dataclasses.field(compare=False)


def build_cpmodel(
    worker_caps: List[int], task_costs: List[int], allowed_workers: List[Set[int]]
) -> cp_model.CpModel:
    model = cp_model.CpModel()
    num_workers = len(worker_caps)
    num_tasks = len(task_costs)

    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f"x[{worker},{task}]")

    # Constraints
    # Each worker is allowed to take tasks up to its capacity
    for worker in range(num_workers):
        for worker, worker_cap in enumerate(worker_caps):
            model.Add(
                sum(
                    task_costs[task] * x[worker, task]
                    for task in range(num_tasks)
                    if worker in allowed_workers
                )
                <= worker_cap
            )

    # Each task is assigned to exactly one worker in its allowed list
    for task in range(num_tasks):
        model.AddExactlyOne(
            x[worker, task]
            for worker in range(num_workers)
            if worker in allowed_workers[task]
        )

    return x, model


class Solver:
    def __init__(
        self,
        worker_caps: List[int],
        task_costs: List[int],
        edges_v: List[EdgeCon],
        allowed_workers: List[Set[int] | Sequence[int]],
    ):
        self._worker_caps = worker_caps
        self._task_costs = task_costs
        self._edges_v = edges_v
        edges: List[Edge] = []
        for u, conns in enumerate(edges_v):
            assert u < len(self._task_costs), u
            for v, c in conns:
                assert v < len(self._task_costs), v
                edges.append((u, v, c))
        self._edges = edges
        self._allowed_workers = allowed_workers

        self._num_workers = len(self._worker_caps)

        self._cp_solver = cp_model.CpSolver()

    def is_solveable(self, state: SolveState) -> bool:
        status = self._cp_solver.Solve(state.cp_model)
        return status == cp_model.OPTIMAL or status == cp_model.FEASIBLE

    def run(self):
        t_sorted = top_sort(self._task_costs, self._edges.copy())
        print(t_sorted)

        pq: List[PqEntry] = []

        cur_caps = np.array(self._worker_caps.copy())
        cur_places = np.array([-1] * len(t_sorted))
        # model_vars, cur_model = build_cpmodel(
        #     cur_caps, self._task_costs, self._allowed_workers
        # )

        state = SolveState(
            task_i=0,
            curr_places=cur_places,
            remaining_caps=cur_caps,
            # cp_model=cur_model,
        )

        pq.append(PqEntry(0, state))
        heapq.heapify(pq)

        searchspace = 0

        while pq:
            head_entry = heapq.heappop(pq)
            cur_cost = head_entry.cost
            cur_state = head_entry.state

            searchspace += 1

            # print(f"cur cost={cur_cost}. cur_i={cur_state.task_i} qs={len(pq)}")

            if cur_state.task_i == len(t_sorted):
                print(f"solved. space={searchspace}")
                return cur_cost, cur_state.curr_places

            u = t_sorted[cur_state.task_i]
            u_cost = self._task_costs[u]

            for u_worker in self._allowed_workers[u]:
                worker_cap = cur_state.remaining_caps[u_worker]
                if u_cost > worker_cap:
                    continue

                # next_model = cp_model.CpModel()
                # next_model.CopyFrom(cur_state.cp_model)
                # # next_model.__model.CopyFrom(cur_state.cp_model.Proto())
                # next_model.Add(model_vars[u_worker, u] == 1)
                # next_status = self._cp_solver.Solve(next_model)
                # if next_status != cp_model.OPTIMAL
                #     print("skipping unsolveable ", u, u_worker)
                #     continue

                next_state = copy.deepcopy(cur_state)
                # next_state.cp_model = next_model
                next_state.task_i += 1
                assert next_state.curr_places[u] == -1
                next_state.curr_places[u] = u_worker
                next_state.remaining_caps[u_worker] -= u_cost

                next_cost = cur_cost

                for v, w in self._edges_v[u]:
                    assert next_state.curr_places[v] != -1
                    v_worker = next_state.curr_places[v]
                    worker_to_worker = 1 if v_worker == u_worker else 5
                    next_cost += w * worker_to_worker

                heapq.heappush(pq, PqEntry(next_cost, next_state))
