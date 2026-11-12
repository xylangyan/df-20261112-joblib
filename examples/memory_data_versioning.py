"""
==============================================================
Versioning cached results when a file content changes in place
==============================================================

This example shows how to make the cache aware of a *data version* when the
cached function only receives a filename and another job replaces the content
of the file in place. In that situation, the function arguments do not change
but the underlying data does, so a plain :class:`joblib.Memory` cache would
keep returning the stale result.

The ``version`` parameter of :meth:`joblib.Memory.cache` (or the ``version``
keyword of a cached call) participates in the cache identity without being
added to the function's signature:

- the same version string always yields the same cache identity, including
  across processes,
- switching version triggers a recomputation,
- the cache of older versions is kept so results can still be replayed.

A counter records every actual execution of the function, which makes the
recomputation stages visible.

"""

import os
import subprocess
import sys
import tempfile

from joblib import Memory

###############################################################################
# The conversion function only receives a filename. On every execution it
# appends a line to a counter file so that recomputations can be counted,
# including from another process.

workdir = tempfile.mkdtemp(prefix="joblib_versioning_")
counter_file = os.path.join(workdir, "counter.log")
data_file = os.path.join(workdir, "data.txt")
cache_dir = os.path.join(workdir, "cache")

# The function lives in its own importable module so that it has the same
# identifier in the main process and in the subprocess used at the end.
with open(os.path.join(workdir, "convert_module.py"), "w") as f:
    f.write(
        "COUNTER_FILE = %r\n"
        "def convert(filename):\n"
        "    with open(COUNTER_FILE, 'a') as counter:\n"
        "        counter.write('computed\\n')\n"
        "    with open(filename) as data:\n"
        "        return data.read().upper()\n" % counter_file
    )

sys.path.insert(0, workdir)
import convert_module  # noqa: E402

memory = Memory(location=cache_dir, verbose=0)
convert = memory.cache(convert_module.convert, version="v1")


def call_count():
    if not os.path.exists(counter_file):
        return 0
    with open(counter_file) as counter:
        return sum(1 for _ in counter)


def stage(label, *args, **kwargs):
    before = call_count()
    result = convert(*args, **kwargs)
    after = call_count()
    status = "RECOMPUTED" if after > before else "cache hit"
    print(f"{label:<34} result={result!r:<8} {status}")


###############################################################################
# Stage 1 - data version v1. The first call computes the result; the second
# one, with the same filename, reuses it.

with open(data_file, "w") as f:
    f.write("alpha")

print("--- data version v1 ---")
stage("v1, first call", data_file)
stage("v1, repeated call", data_file)

###############################################################################
# Stage 2 - another job replaces the content of the same file in place. The
# filename is unchanged, so without a version the stale result would be
# returned. Passing ``version="v2"`` selects a new cache generation and
# triggers exactly one recomputation.

with open(data_file, "w") as f:
    f.write("beta")

print("--- data version v2 (file swapped in place) ---")
stage("v2, first call", data_file, version="v2")
stage("v2, repeated call", data_file, version="v2")

###############################################################################
# Stage 3 - explicitly read the v1 result back. The old generation was not
# deleted when v2 was computed, so this is a cache hit and returns the value
# computed against the v1 data.

print("--- replay of the previous version ---")
stage("v1, explicit read back", data_file, version="v1")

###############################################################################
# The cache identity only depends on the version string, not on the process.
# A different process reusing the same cache and the same version does not
# recompute anything: the counter file is untouched.

subprocess.check_call(
    [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, %r);"
        "from joblib import Memory;"
        "import convert_module;"
        "memory = Memory(%r, verbose=0);"
        "convert = memory.cache(convert_module.convert, version='v2');"
        "assert convert(%r) == 'BETA'" % (workdir, cache_dir, data_file),
    ]
)

print("--- cross-process reuse ---")
print(f"calls after a subprocess using version v2: {call_count()} (unchanged)")
