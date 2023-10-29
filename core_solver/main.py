import sys
import time
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Iterable, List, Set, Tuple

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from .solver import EdgeCon, Solver

worker_caps = [5, 5]
task_costs = [1, 2, 2, 3, 2]
num_workers = len(worker_caps)
num_tasks = len(task_costs)
allowed_workers: List[Set[int]] = [
    {0, 1},
    {0, 1},
    {0, 1},
    {1},
    {0, 1},
]


edges_v: List[EdgeCon] = [
    [(1, 1)],
    [],
    [(1, 0.1)],
    [(2, 2)],
    [(2, 10)],
]


def getsize(obj):
    BLACKLIST = type, ModuleType, FunctionType
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def test_mipmodel():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    assert isinstance(solver, pywraplp.Solver)

    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = solver.BoolVar(f"x[{worker},{task}]")

    # Constraints
    # Each worker is tasks up to its capacity
    for worker, worker_cap in enumerate(worker_caps):
        solver.Add(
            solver.Sum(
                task_costs[task] * x[worker, task]
                for task in range(num_tasks)
                if worker in allowed_workers[task]
            )
            <= worker_cap
        )

    # Each task is assigned to exactly one worker in its allowed list
    for task in range(num_tasks):
        solver.Add(
            solver.Sum(
                x[worker, task]
                for worker in range(num_workers)
                if worker in allowed_workers[task]
            )
            == 1
        )

    # Objective
    cost_terms = []
    for u, es in enumerate(edges_v):
        for v, w in es:
            for worker_u in range(num_workers):
                for worker_v in range(num_workers):
                    worker_to_worker = 5 if worker_u != worker_v else 1
                    cost_terms.append(
                        (x[worker_u, u] + x[worker_v, v]) * w * worker_to_worker
                    )
    # print(cost_terms)
    solver.Minimize(solver.Sum(cost_terms))

    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost = {solver.Objective().Value()}\n")
        for worker in range(num_workers):
            for task in range(num_tasks):
                if x[worker, task].solution_value() > 0.5:
                    print(
                        f"Worker {worker} assigned to task {task}."
                        + f" Cost: {task_costs[task]}"
                    )
    else:
        print("No solution found.")


def test_cpmodel() -> cp_model.CpModel:
    model = cp_model.CpModel()

    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f"x[{worker},{task}]")

    # Constraints
    # Each worker is tasks up to its capacity
    for worker, worker_cap in enumerate(worker_caps):
        model.Add(
            sum(
                task_costs[task] * x[worker, task]
                for task in range(num_tasks)
                if worker in allowed_workers[task]
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

    # cost_terms = []
    # for u, es in enumerate(edges_v):
    #     for v, w in es:
    #         for worker_u in range(num_workers):
    #             for worker_v in range(num_workers):
    #                 worker_to_worker = 5 if worker_u != worker_v else 1
    #                 print(u, v, worker_u, worker_v, w * worker_to_worker)
    #                 cost_terms.append(
    #                     (x[worker_u, u] + x[worker_v, v]) * w * worker_to_worker
    #                 )
    # print(cost_terms)
    # model.Minimize(sum(cost_terms))

    # model.Add(x[0, 1] == 1)
    # print(model)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total cost = {solver.ObjectiveValue()} status={status}\n")
        for worker in range(num_workers):
            for task in range(num_tasks):
                if solver.BooleanValue(x[worker, task]):
                    print(
                        f"Worker {worker} assigned to task {task}."
                        + f" Cost = {task_costs[task]}"
                    )
    else:
        print("No solution found.", status)

    print("size:", getsize(model))

    return model


def run():
    # start = time.time()
    # test_cpmodel()
    # end_ts = time.time()
    # print("solver:", end_ts - start)

    # start = time.time()
    # test_mipmodel()
    # end_ts = time.time()
    # print("solver:", end_ts - start)
    # return

    solver = Solver(
        worker_caps=worker_caps,
        task_costs=task_costs,
        edges_v=edges_v,
        allowed_workers=allowed_workers,
    )
    start_ts = time.time()
    cost, placement = solver.run()
    end_ts = time.time()
    print("runtime:", end_ts - start_ts)
    print("sol ", cost, " ", placement)
