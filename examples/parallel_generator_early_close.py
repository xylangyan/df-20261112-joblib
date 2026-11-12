"""
=====================================================
Early-closing a Parallel generator and reusing it
=====================================================

Minimal reproduction of a partial-sampling usage pattern: round A only
consumes the first result and closes the generator, then round B is run
on the same managed backend immediately afterwards.

Closing the generator early tells joblib to:

* cancel the tasks that were still waiting to start as soon as possible,
* let the tasks already running finish according to the backend's
  capabilities (threads cannot be killed, so they run until they return),
* keep the backend ready for the next call and isolate each call so that
  round B only sees its own results and exceptions.

A threading.Event is used to block the running tasks deterministically,
instead of relying on fixed sleeps.
"""

import threading
import time
import warnings

from joblib import Parallel, delayed


events_log = []
events_log_lock = threading.Lock()
t0 = time.time()


def _record(event_name, round_id, task_id):
    with events_log_lock:
        events_log.append((time.time() - t0, event_name, round_id, task_id))


def task(round_id, task_id, block_event):
    _record("start", round_id, task_id)

    # Controlled blocking: for the slow tasks of round A this event is set
    # only after the generator has been closed and round B has started.
    block_event.wait()

    _record("end", round_id, task_id)
    return (round_id, task_id)


class _UnblockedEvent:
    def wait(self, timeout=None):
        return True


def main():
    def elapsed():
        return f"{time.time() - t0:6.3f}s"

    n_jobs = 2
    round_a_block = threading.Event()

    with Parallel(
        n_jobs=n_jobs,
        backend="threading",
        return_as="generator",
        pre_dispatch="all",
    ) as parallel:
        # Round A: 6 tasks, the 2 workers grab A1 and A2 which block,
        # A3..A5 stay queued until the generator is closed.
        generator_a = parallel(
            delayed(task)(
                "A", i, round_a_block if i > 0 else _UnblockedEvent()
            )
            for i in range(6)
        )
        first_result = next(generator_a)
        print(f"[{elapsed()}] round A first result: {first_result}")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            generator_a.close()
        for warning in caught_warnings:
            print(f"[{elapsed()}] expected warning: {warning.message}")
        print(f"[{elapsed()}] round A generator closed after one result")

        # Unblock the tasks that were already running when A was closed.
        round_a_block.set()

        # Round B runs immediately on the same backend and must only see
        # its own results.
        results_b = list(
            parallel(
                delayed(task)("B", i, _UnblockedEvent()) for i in range(2)
            )
        )
        print(f"[{elapsed()}] round B results: {results_b}")

    print("\ntask lifecycle log (chronological):")
    for timestamp, event_name, round_id, task_id in sorted(events_log):
        print(f"  {timestamp:6.3f}s {event_name:5s} {round_id}{task_id}")

    started = {
        (round_id, task_id)
        for _, event_name, round_id, task_id in events_log
        if event_name == "start"
    }
    cancelled_a = {
        ("A", i) for i in range(6) if ("A", i) not in started
    }
    assert cancelled_a, "at least one queued round A task must be skipped"
    assert sorted(results_b) == [("B", 0), ("B", 1)]
    print(
        "\nqueued round A tasks were skipped and round B stayed isolated "
        f"({len(cancelled_a)} task(s) never started)"
    )


if __name__ == "__main__":
    main()
