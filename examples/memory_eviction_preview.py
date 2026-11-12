"""
====================================
Previewing cache eviction with plans
====================================

This example illustrates how to preview which cache items
:class:`joblib.Memory` would evict when the cache limits are tightened,
before actually deleting anything. The same plan can then be applied
later, e.g. after a review by another operator, and items that were
rewritten in the meantime are kept instead of being deleted.

"""

###############################################################################
# Setup a small cache in a temporary directory
###############################################################################
#
# We cache four results of a toy function in a temporary directory.

import json
import os
import pickle
import shutil
import tempfile

from joblib import Memory

work_dir = tempfile.mkdtemp(prefix="joblib_eviction_")
cache_dir = os.path.join(work_dir, "cache")
memory = Memory(location=cache_dir, verbose=0)


@memory.cache
def costly_compute(index):
    return "x" * 1000 * (index + 1)


for index in range(4):
    costly_compute(index)

###############################################################################
# Preview the eviction with a plan
###############################################################################
#
# Tightening the limits (here: keep at most 1 item) produces a
# ``CacheEvictionPlan``. The plan lists the candidate keys, is ordered by
# last access time (oldest first, with the key as a stable tie-breaker)
# and reports the size expected to be reclaimed. Building the plan is
# read-only: it does not refresh access times nor modify any metadata.

plan = memory.preview_reduce_size(items_limit=1)
for entry in plan.entries:
    print(f"would evict {entry.key} ({entry.size} bytes, {entry.last_access})")
print(f"planned reclaimed size: {plan.planned_reclaimed_size} bytes")

###############################################################################
# Keep the plan and the pre-application listing for review
###############################################################################
#
# The plan and the current store listing are saved so that another shift
# can review them before the plan is applied.

plan_file = os.path.join(work_dir, "eviction_plan.pkl")
listing_before_file = os.path.join(work_dir, "listing_before.json")
listing_after_file = os.path.join(work_dir, "listing_after.json")

with open(plan_file, "wb") as f:
    pickle.dump(plan, f)
with open(listing_before_file, "w") as f:
    json.dump([entry.key for entry in plan.entries], f, indent=2)

###############################################################################
# Meanwhile, the store changes before the plan is applied
###############################################################################
#
# A concurrent process recomputes one of the planned items (it is
# rewritten and must not be evicted anymore) while another planned item
# gets removed directly (it no longer exists). The third candidate is
# left untouched. The oldest planned entry (first in the plan) corresponds
# to ``costly_compute(0)`` and is the one being recomputed.

rewritten, missing, unchanged = plan.entries

costly_compute(0)
shutil.rmtree(missing.path)

###############################################################################
# Apply the very same plan
###############################################################################
#
# Applying the plan distinguishes three cases per entry: unchanged items
# are deleted, items that disappeared are reported as missing, and items
# that were rewritten since the preview are kept and reported.

with open(plan_file, "rb") as f:
    plan = pickle.load(f)

report = memory.apply_reduce_size_plan(plan)
for entry in report.entries:
    print(f"{entry.status}: {entry.key}")

remaining = [
    os.path.relpath(item.path, memory.store_backend.location)
    for item in memory.store_backend.get_items()
]
with open(listing_after_file, "w") as f:
    json.dump(remaining, f, indent=2)

assert report.deleted == (unchanged.key,)
assert report.missing == (missing.key,)
assert report.rewritten == (rewritten.key,)

###############################################################################
# Reconcile planned and actual reclaimed space
###############################################################################
#
# The difference between the planned and the actually reclaimed size
# accounts for the items that were missing or rewritten between the
# preview and the application.

print(f"actually reclaimed: {report.reclaimed_size} bytes")
print(
    "difference with the plan: "
    f"{report.planned_reclaimed_size - report.reclaimed_size} bytes"
)

shutil.rmtree(work_dir)
