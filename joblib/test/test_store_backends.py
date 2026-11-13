import functools
import os
import pickle as cpickle
import time
import types
import warnings
from pickle import PicklingError

import pytest

from joblib import Parallel, delayed, numpy_pickle
from joblib._store_backends import (
    CacheWarning,
    FileSystemStoreBackend,
    _CallLockTimeout,
    _old_split_id,
)
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, timeout, warns


def write_func(output, filename):
    with open(filename, "wb") as f:
        cpickle.dump(output, f)


def load_func(expected, filename):
    for i in range(10):
        try:
            with open(filename, "rb") as f:
                reloaded = cpickle.load(f)
            break
        except (OSError, IOError):
            # On Windows you can have WindowsError ([Error 5] Access
            # is denied or [Error 13] Permission denied) when reading the file,
            # probably because a writer process has a lock on the file
            time.sleep(0.1)
    else:
        raise
    assert expected == reloaded


@timeout(0)  # No timeout as this test can be long
@with_multiprocessing
@parametrize("backend", ["multiprocessing", "loky", "threading"])
def test_concurrency_safe_write(tmpdir, backend):
    store_backend = FileSystemStoreBackend()
    store_backend.configure(tmpdir.join("store").strpath)

    # Add one item to cache
    filename = tmpdir.join("test.pkl").strpath

    obj = {str(i): i for i in range(int(1e5))}
    funcs = [
        functools.partial(store_backend._concurrency_safe_write, write_func=write_func)
        if i % 3 != 2
        else load_func
        for i in range(12)
    ]
    Parallel(n_jobs=2, backend=backend)(delayed(func)(obj, filename) for func in funcs)


def test_store_cached_func_code_is_not_written_in_place(tmpdir):
    # Non-regression test for #1694: func_code.py used to be truncated and
    # rewritten in place, so a concurrent reader could catch it half-written,
    # read that as a changed function and wipe the whole cache directory.
    backend = FileSystemStoreBackend()
    backend.configure(tmpdir.join("store").strpath)

    call_id = ("mod", "func")
    backend.store_cached_func_code(call_id, "# first line: 1\ndef f(): pass\n")
    original = backend.get_cached_func_code(call_id)

    def exploding_open(name, mode):
        open(name, mode).close()  # truncates whatever it is pointed at
        raise RuntimeError("boom")

    old_open = backend._open_item
    backend._open_item = exploding_open
    with pytest.raises(RuntimeError, match="boom"):
        backend.store_cached_func_code(call_id, "# first line: 1\ndef g(): pass\n")

    backend._open_item = old_open
    assert backend.get_cached_func_code(call_id) == original


def test_warning_on_dump_failure(tmp_path):
    # Check that a warning is raised when the dump fails for any reason but
    # a PicklingError.
    class UnpicklableObject(object):
        def __reduce__(self):
            raise RuntimeError("some exception")

    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path))

    with pytest.warns(CacheWarning, match="some exception"):
        backend.dump_item(("func", "input"), UnpicklableObject())
    assert not list(tmp_path.rglob("output.pkl.*"))


def test_warning_on_pickling_error(tmp_path):
    # This is separate from test_warning_on_dump_failure because in the
    # future we will turn this into an exception.
    class UnpicklableObject(object):
        def __reduce__(self):
            raise PicklingError("not picklable")

    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path))

    with pytest.warns(FutureWarning, match="not picklable"):
        backend.dump_item(("func", "input"), UnpicklableObject())


def test_cache_tree_versions(tmpdir):
    # Makes a backend using old cache tree
    old_store = FileSystemStoreBackend()
    old_store.configure(tmpdir.strpath)
    old_store._split_id = types.MethodType(_old_split_id, old_store)
    os.remove(os.path.join(old_store.location, "store_backend_info.json"))

    # Cache some items
    funs = ["fun1", "fun2", os.path.join("z" * 32, "fun3")]
    args = ["012" + "3" * 29, "012" + "4" * 29, "abcd" * 8]
    items = [0, "xyz", [42, 6.9]]
    for fun in funs:
        for arg, item in zip(args, items):
            old_store.dump_item((fun, arg), item)

    # Assert store_backend finds items from old cache tree
    old_items = old_store.get_items()
    assert len(old_items) == len(funs) * len(args)
    assert set(item.path for item in old_items) == set(
        tmpdir.join(fun, arg).strpath for fun in funs for arg in args
    )

    # Assert a warning is raised when configuring a FileSystemStoreBackend
    # with an old cache tree
    with warns(UserWarning, match="using old cache storage tree"):
        new_store = FileSystemStoreBackend()
        new_store.configure(tmpdir.strpath)

    # Pollute the backend
    # Creates items with new tree. After the creation:
    # (funs[0], args[0]) old tree version is more recent
    # (funs[1], args[0]) new tree version is more recent
    conflict_item = "bad"
    for fun in funs[:2]:
        arg = args[0]
        conflict_dir = tmpdir.join(fun, arg[:3], arg[3:]).strpath
        os.makedirs(conflict_dir)
        numpy_pickle.dump(conflict_item, os.path.join(conflict_dir, "output.pkl"))
    time.sleep(1.5)
    new_store.dump_item((funs[0], args[0]), items[0])

    # Assert (funs[0], args[0]) old tree version is more recent
    assert os.path.getmtime(
        tmpdir.join(funs[0], arg[:3], arg[3:], "output.pkl").strpath
    ) < os.path.getmtime(tmpdir.join(funs[0], arg, "output.pkl").strpath)

    # Assert update_cache_tree correctly updates the cache tree
    new_store.update_cache_tree()
    new_items = new_store.get_items()
    assert len(new_items) == len(funs) * len(args)
    assert set(item.path for item in new_items) == set(
        tmpdir.join(fun, arg[:3], arg[3:]).strpath for fun in funs for arg in args
    )

    # Assert no warning is raised when caching a function with a new cache tree
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        FileSystemStoreBackend().configure(tmpdir.strpath)
        assert len(ws) == 0

    # Check that old_store shifted to new cache tree
    assert old_store.load_item((funs[0], args[0])) == items[0]
    assert old_store.load_item((funs[1], args[0])) == conflict_item
    with raises(KeyError):
        old_store.load_item(("not_a_function", args[0]))


def test_exclusive_compute_option_is_opt_in(tmp_path):
    # By default, the backend does not provide any per-call lock, so that
    # existing code and existing cache trees are left untouched.
    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path))
    assert backend.call_lock(("func", "a" * 32)) is None
    assert backend.exclusive_compute["enabled"] is False


def test_exclusive_compute_option_configuration(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(
        str(tmp_path),
        backend_options={
            "exclusive_compute": {
                "enabled": True,
                "timeout": 5,
                "stale_timeout": 2,
                "poll_interval": 0.01,
            }
        },
    )
    assert backend.exclusive_compute == {
        "enabled": True,
        "timeout": 5,
        "stale_timeout": 2,
        "poll_interval": 0.01,
    }
    # Boolean shorthand
    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path), backend_options={"exclusive_compute": True})
    assert backend.exclusive_compute["enabled"] is True
    assert backend.call_lock(("func", "a" * 32)) is not None


def test_exclusive_compute_lock_is_keyed_by_normalized_call_id(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path), backend_options={"exclusive_compute": True})

    # The lock lives inside the item directory, i.e. it uses the same
    # normalized (split, 3+29) call id as the cached item itself.
    args_id = "a" * 32
    lock = backend.call_lock(("func", args_id))
    assert lock.path == os.path.join(
        backend.location, "func", args_id[:3], args_id[3:], ".joblib_call.lock"
    )
    # Different arguments get different locks and hence can run in parallel.
    other_lock = backend.call_lock(("func", "b" * 32))
    assert other_lock.path != lock.path


def test_exclusive_compute_lock_roundtrip(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(
        str(tmp_path),
        backend_options={"exclusive_compute": {"enabled": True, "poll_interval": 0.01}},
    )
    lock = backend.call_lock(("func", "a" * 32))
    assert lock.acquire()
    assert os.path.exists(lock.path)
    lock.release()
    assert not os.path.exists(lock.path)


def test_exclusive_compute_stale_lock_is_recovered(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(
        str(tmp_path),
        backend_options={
            "exclusive_compute": {
                "enabled": True,
                "stale_timeout": 0.1,
                "poll_interval": 0.01,
            }
        },
    )
    lock = backend.call_lock(("func", "a" * 32))
    os.makedirs(os.path.dirname(lock.path), exist_ok=True)
    # Simulate a process that died while holding the lock long ago.
    with open(lock.path, "w") as f:
        f.write("0 1234")
    assert lock.acquire()
    assert lock.acquired
    lock.release()


def test_exclusive_compute_lock_timeout(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(
        str(tmp_path),
        backend_options={
            "exclusive_compute": {
                "enabled": True,
                "timeout": 0.1,
                "stale_timeout": None,
                "poll_interval": 0.01,
            }
        },
    )
    held = backend.call_lock(("func", "a" * 32))
    held.acquire()
    other = backend.call_lock(("func", "a" * 32))
    with raises(_CallLockTimeout):
        other.acquire()
    held.release()


def test_backend_pickle_preserves_exclusive_compute(tmp_path):
    backend = FileSystemStoreBackend()
    backend.configure(str(tmp_path), backend_options={"exclusive_compute": True})
    unpickled = cpickle.loads(cpickle.dumps(backend))
    assert unpickled.exclusive_compute["enabled"] is True
