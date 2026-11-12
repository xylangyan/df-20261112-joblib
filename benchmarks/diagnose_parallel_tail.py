"""Diagnose tail waits in joblib.Parallel without changing joblib internals.

Run from the repository root:

    python benchmarks/diagnose_parallel_tail.py
    python benchmarks/diagnose_parallel_tail.py --include-auto-growth

The main experiment intentionally uses the threading backend so every event
shares one monotonic clock and the two-worker startup barrier is easy to
control. It keeps n_jobs fixed and compares:

* fixed batch size vs batch_size="auto";
* ordered generator vs generator_unordered.

The optional process-backend section is separate because process backends
actually grow auto batches, while the threading backend keeps auto batches at
size one. Conclusions from each section are specific to the backend used.
"""

import argparse
import csv
import platform
import threading
import time
from collections import namedtuple

import joblib
from joblib import Parallel, delayed
from joblib.parallel import BatchCompletionCallBack


Event = namedtuple("Event", "t phase target detail")

N_JOBS = 2
N_TASKS = 8
PRE_DISPATCH = "2*n_jobs"
BARRIER_TASKS = (0, 2)
SLOW_TASK = 0
BARRIER_TIMEOUT = 30.0
GROWTH_N_TASKS = 24
GROWTH_SLOW_TASK = 18
GROWTH_FAST_SECONDS = 0.003
GROWTH_SLOW_SECONDS = 0.50


class EventSink:
    def __init__(self):
        self._events = []
        self._lock = threading.Lock()
        self.t0 = time.perf_counter()

    def add(self, phase, target="", detail=""):
        event = Event(
            time.perf_counter() - self.t0,
            phase,
            str(target),
            detail,
        )
        with self._lock:
            self._events.append(event)
        return event

    def rows(self):
        with self._lock:
            return sorted(self._events, key=lambda event: event.t)


class InstrumentedCallback(BatchCompletionCallBack):
    def __call__(self, *args, **kwargs):
        self.log.add("callback_start", self.tag, f"batch_size={self.batch_size}")
        result = super().__call__(*args, **kwargs)
        self.log.add("callback_end", self.tag, f"batch_size={self.batch_size}")
        return result


class InstrumentedParallel(Parallel):
    def _dispatch(self, batch):
        task_ids = tuple(call_args[0] for _, call_args, _ in batch.items)
        tag = "/".join(str(task_id) for task_id in task_ids)
        self.log.add("submit", tag, f"batch_size={len(batch)}")
        if not hasattr(self, "dispatch_rows"):
            self.dispatch_rows = []
        self.dispatch_rows.append(
            {
                "t": time.perf_counter() - self.log.t0,
                "task_ids": task_ids,
                "size": len(batch),
                "label": tag,
            }
        )

        batch_size = len(batch)
        self.n_dispatched_tasks += batch_size
        self.n_dispatched_batches += 1

        callback = InstrumentedCallback(time.time(), batch_size, self)
        callback.tag = tag
        callback.log = self.log
        self._register_new_job(callback)
        job = self._backend.submit(batch, callback=callback)
        callback.register_job(job)


def diagnostic_task(task_id, sink, barrier, fast_seconds, slow_seconds):
    sink.add(
        "worker_start",
        task_id,
        f"worker={threading.current_thread().name}",
    )
    if task_id in BARRIER_TASKS:
        barrier.wait(timeout=BARRIER_TIMEOUT)
        sink.add("barrier_pass", task_id)

    duration = slow_seconds if task_id == SLOW_TASK else fast_seconds
    time.sleep(duration)
    sink.add("worker_end", task_id, f"duration={duration:.3f}")
    return task_id


def make_tasks(sink, barrier, fast_seconds, slow_seconds):
    for task_id in range(N_TASKS):
        sink.add("produced", task_id)
        yield delayed(diagnostic_task)(
            task_id,
            sink,
            barrier,
            fast_seconds,
            slow_seconds,
        )


def run_combination(batch_size, return_as, fast_seconds, slow_seconds):
    sink = EventSink()
    barrier = threading.Barrier(parties=len(BARRIER_TASKS))
    parallel = InstrumentedParallel(
        n_jobs=N_JOBS,
        backend="threading",
        pre_dispatch=PRE_DISPATCH,
        batch_size=batch_size,
        return_as=return_as,
    )
    parallel.log = sink

    output = parallel(
        make_tasks(sink, barrier, fast_seconds, slow_seconds)
    )
    initial_batches = parallel.n_dispatched_batches
    initial_tasks = parallel.n_dispatched_tasks
    sink.add(
        "generator_return",
        "",
        f"initial_batches={initial_batches} "
        f"initial_tasks={initial_tasks}",
    )

    read_rows = []
    for value in output:
        read_rows.append(
            sink.add(
                "read",
                value,
                f"dispatched={parallel.n_dispatched_tasks} "
                f"completed={parallel.n_completed_tasks}",
            )
        )

    rows = sink.rows()
    first_read = read_rows[0]
    head_callback = next(
        row
        for row in rows
        if row.phase == "callback_start"
        and int(row.target.split("/")[0]) == SLOW_TASK
    )
    later_ready = next(
        row
        for row in rows
        if row.phase == "callback_start"
        and int(row.target.split("/")[0]) != SLOW_TASK
    )
    head_batch = next(
        row
        for row in rows
        if row.phase == "submit" and int(row.target.split("/")[0]) == SLOW_TASK
    )
    peer_coupling = "/" in head_batch.target
    if peer_coupling and return_as == "generator":
        wait_point = (
            "batch peer tasks wait for slow task 0; then ordered "
            "retrieval withholds completed later batches"
        )
    elif peer_coupling:
        wait_point = (
            "batch peer tasks wait for slow task 0; no ordered "
            "retrieval wait"
        )
    elif return_as == "generator":
        wait_point = "ordered retrieval waits for slow task 0's result"
    else:
        wait_point = "only slow task 0 itself remains at the tail"

    return {
        "label": f"{batch_size}/{return_as}",
        "rows": rows,
        "initial_batches": initial_batches,
        "initial_tasks": initial_tasks,
        "first_read": first_read,
        "later_ready": later_ready,
        "head_callback": head_callback,
        "wait_point": wait_point,
        "effective_batch_note": (
            "threading auto uses the base backend batch size of 1"
            if batch_size == "auto"
            else f"fixed batch_size={batch_size}"
        ),
    }


def growth_task(task_id, slow_task_id, fast_seconds, slow_seconds, events, t0):
    duration = slow_seconds if task_id == slow_task_id else fast_seconds
    time.sleep(duration)
    events.append((time.perf_counter() - t0, "worker_end", task_id, duration))
    return task_id


def run_growth_combination(
    batch_size,
    n_tasks,
    slow_task_id,
    fast_seconds,
    slow_seconds,
    manager,
):
    events = manager.list()
    t0 = time.perf_counter()
    submit_rows = []
    read_rows = []

    parallel = InstrumentedParallel(
        n_jobs=N_JOBS,
        backend="loky",
        pre_dispatch=PRE_DISPATCH,
        batch_size=batch_size,
        return_as="generator_unordered",
    )
    parallel.log = _ProxyEventLog(events, t0)

    def producer():
        for task_id in range(n_tasks):
            events.append(
                (
                    time.perf_counter() - t0,
                    "produced",
                    task_id,
                    "",
                )
            )
            yield delayed(growth_task)(
                task_id,
                slow_task_id,
                fast_seconds,
                slow_seconds,
                events,
                t0,
            )

    output = parallel(producer())
    events.append(
        (
            time.perf_counter() - t0,
            "generator_return",
            "",
            f"initial_batches={parallel.n_dispatched_batches} "
            f"initial_tasks={parallel.n_dispatched_tasks}",
        )
    )

    for value in output:
        event = (
            time.perf_counter() - t0,
            "read",
            value,
            f"dispatched={parallel.n_dispatched_tasks} "
            f"completed={parallel.n_completed_tasks}",
        )
        events.append(event)
        read_rows.append(event)

    for row in parallel.dispatch_rows:
        submit_rows.append(row)

    slow_batch = next(
        row for row in submit_rows if slow_task_id in row["task_ids"]
    )
    max_batch_size = max(row["size"] for row in submit_rows)
    rows = [Event(*row) for row in sorted(events, key=lambda item: item[0])]
    submit_labels = [row["label"] for row in submit_rows]
    read_order = [row[2] for row in read_rows]

    return {
        "label": f"loky {batch_size}",
        "rows": rows,
        "submit_rows": submit_rows,
        "read_rows": read_rows,
        "read_order": read_order,
        "submit_labels": submit_labels,
        "slow_batch": slow_batch,
        "max_batch_size": max_batch_size,
    }


class _ProxyEventLog:
    def __init__(self, events, t0):
        self.events = events
        self.t0 = t0

    def add(self, phase, target="", detail=""):
        self.events.append(
            (
                time.perf_counter() - self.t0,
                phase,
                target,
                detail,
            )
        )


def print_event_table(title, rows):
    print(f"\n## {title}")
    print(f"{'time_s':>7}  {'phase':<16}  {'target':<12}  detail")
    print("-" * 78)
    for event in rows:
        print(
            f"{event.t:7.3f}  {event.phase:<16}  "
            f"{event.target:<12}  {event.detail}"
        )


def print_main_summary(results):
    print("\n## Four-combination event table")
    print(
        f"{'combo':<24} {'initial':>8} {'submit_s':>9} "
        f"{'batch_end_s':>12} {'first_read':>10} {'read_s':>8}  "
        "first mutual wait"
    )
    print("-" * 130)
    for result in results:
        initial = (
            f"{result['initial_batches']}b/"
            f"{result['initial_tasks']}t"
        )
        first_submit = next(
            row for row in result["rows"] if row.phase == "submit"
        )
        print(
            f"{result['label']:<24} {initial:>8} "
            f"{first_submit.t:9.3f} {result['later_ready'].t:12.3f} "
            f"{result['first_read'].target:>10} "
            f"{result['first_read'].t:8.3f}  {result['wait_point']}"
        )
    print("\nDetailed event rows for all four combinations are printed above.")
    print("`batch_end_s` is the first completion callback of a non-head batch.")


def print_growth_summary(result):
    print("\n## Process-backend auto-growth events")
    print("submit batches:")
    for row in result["submit_rows"]:
        print(
            f"  {row['t']:7.3f} size={row['size']:<2} "
            f"tasks={row['label']}"
        )
    print("\nread order (last 10 values):")
    for row in result["read_rows"][-10:]:
        print(
            f"  {row[0]:7.3f} task={row[2]:<3} "
            f"dispatched/completed detail: {row[3]}"
        )
    print(f"\nsubmission labels: {result['submit_labels']}")
    print(f"unordered read order: {result['read_order']}")
    print(
        f"slow task {GROWTH_SLOW_TASK} was submitted in "
        f"size={result['slow_batch']['size']} batch "
        f"{result['slow_batch']['label']}; max auto batch observed: "
        f"{result['max_batch_size']}."
    )


def write_csv(path, labeled_results):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["combination", "time_s", "phase", "target", "detail"])
        for label, rows in labeled_results:
            for event in rows:
                writer.writerow(
                    [label, f"{event.t:.6f}", event.phase, event.target, event.detail]
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Parallel tail waits and batch/result ordering."
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=2,
        help="Fixed batch_size used in the main experiment.",
    )
    parser.add_argument(
        "--fast-seconds",
        type=float,
        default=0.02,
        help="Duration of every non-slow task in the main experiment.",
    )
    parser.add_argument(
        "--slow-seconds",
        type=float,
        default=0.50,
        help="Duration of the slow head task in the main experiment.",
    )
    parser.add_argument(
        "--include-auto-growth",
        action="store_true",
        help="Add a loky process-backend check for auto batch growth.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Optional path for all timestamped events.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Parallel tail-wait diagnostic")
    print(f"python={platform.python_version()}")
    print(f"joblib={joblib.__version__}")
    print(
        f"backend=threading n_jobs={N_JOBS} pre_dispatch={PRE_DISPATCH!r} "
        f"tasks={N_TASKS} barrier_tasks={BARRIER_TASKS} "
        f"slow_task={SLOW_TASK}"
    )
    print(
        "The worker count is deliberately fixed; this script does not test "
        "or recommend increasing n_jobs."
    )

    combinations = [
        (args.fixed_batch_size, "generator"),
        (args.fixed_batch_size, "generator_unordered"),
        ("auto", "generator"),
        ("auto", "generator_unordered"),
    ]
    results = []
    for batch_size, return_as in combinations:
        result = run_combination(
            batch_size,
            return_as,
            args.fast_seconds,
            args.slow_seconds,
        )
        results.append(result)
        print_event_table(
            f"{result['label']} - {result['effective_batch_note']}",
            result["rows"],
        )

    print_main_summary(results)

    labeled_results = [(result["label"], result["rows"]) for result in results]

    if args.include_auto_growth:
        import multiprocessing as mp

        manager = mp.Manager()
        try:
            growth = run_growth_combination(
                "auto",
                GROWTH_N_TASKS,
                GROWTH_SLOW_TASK,
                GROWTH_FAST_SECONDS,
                GROWTH_SLOW_SECONDS,
                manager,
            )
        finally:
            manager.shutdown()
        print_growth_summary(growth)
        labeled_results.append((growth["label"], growth["rows"]))

    print("\n## Backend-specific interpretation")
    print(
        "- threading: auto stays at batch size 1 because ThreadingBackend "
        "does not use AutoBatchingMixin; fixed=2 batches co-locate tasks."
    )
    print(
        "- generator: even completed later batches remain unreadable until "
        "the oldest submitted batch is retrievable."
    )
    print(
        "- generator_unordered: completion order controls readability, so "
        "later batches can be consumed before the slow head batch."
    )
    if args.include_auto_growth:
        print(
            "- loky only: auto can grow batches on many fast process tasks, "
            "allowing a late slow task to trap several co-batched fast tasks."
        )

    if args.csv_path:
        write_csv(args.csv_path, labeled_results)
        print(f"\nWrote event CSV: {args.csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
