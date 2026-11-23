"""Process-death-safe exclusive file locks shared by CI asset stagers.

Uses OS-managed advisory locks (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows) so the
kernel releases the lock immediately if a holder crashes or is killed (`SIGKILL`).

When `remove_on_exit=True` (used for placement locks inside asset directories where no sidecar
`.lock` file may remain behind), POSIX callers verify `(st_dev, st_ino)` after acquiring `flock`
and unlink the lockfile while still holding the lock to prevent the classic `flock`+unlink inode
race. If `fcntl.flock` itself raises `ENOTSUP`/`EOPNOTSUPP` on an exotic filesystem and
`fallback_excl_on_notsup=True`, callers fall back to `O_CREAT | O_EXCL` with the owner PID
recorded in the lockfile and automatically reclaim locks whose recorded PID is no longer alive.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time

# Only these mean "another holder has it"; any other lock error (ENOLCK, EBADF, EINVAL, or
# ENOTSUP when fallback is disabled) is raised instead of spinning forever.
POSIX_LOCK_BUSY = frozenset((errno.EWOULDBLOCK, errno.EAGAIN))
POSIX_LOCK_UNSUPPORTED = frozenset((errno.ENOTSUP, errno.EOPNOTSUPP))
MSVCRT_LOCK_BUSY = frozenset((errno.EACCES, errno.EDEADLK))
# Grace period before treating an empty/unparseable `O_CREAT | O_EXCL` fallback lockfile as stale
# (handles an owner killed between `os.open` and writing its PID).
STALE_EMPTY_LOCK_GRACE_SECONDS = 1.0


def try_lock_handle(handle, *, is_windows: bool | None = None) -> bool:
    """Attempt a non-blocking exclusive OS lock on `handle`.

    Returns `True` when locked, `False` on contention, and raises any other `OSError`.
    """
    windows = (os.name == "nt") if is_windows is None else is_windows
    try:
        if windows:
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno in (MSVCRT_LOCK_BUSY if windows else POSIX_LOCK_BUSY):
            return False
        raise


def unlock_handle(handle, *, is_windows: bool | None = None) -> None:
    """Release an OS lock previously acquired by `try_lock_handle`."""
    windows = (os.name == "nt") if is_windows is None else is_windows
    if windows:
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _pid_is_alive(pid: int) -> bool:
    """Whether `pid` currently names a live OS process on this host."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists, owned by another user.
        return True


def _reclaim_if_dead_owner(path: Path, now_wall: float) -> bool:
    """Unlink a stale `O_CREAT | O_EXCL` fallback lockfile if its owner PID is dead.

    Returns `True` when the lockfile was removed (or already gone) so the caller should retry
    acquisition immediately, and `False` when a live owner still holds it.
    """
    try:
        st = os.stat(path)
        raw = path.read_bytes()[:64]
    except FileNotFoundError:
        return True
    except OSError:
        return False
    text = raw.decode("ascii", errors="ignore").strip()
    stale = False
    if text:
        try:
            pid = int(text.splitlines()[0].strip())
        except ValueError:
            stale = (now_wall - st.st_mtime) >= STALE_EMPTY_LOCK_GRACE_SECONDS
        else:
            stale = not _pid_is_alive(pid)
    else:
        stale = (now_wall - st.st_mtime) >= STALE_EMPTY_LOCK_GRACE_SECONDS
    if not stale:
        return False
    try:
        current = os.stat(path)
        if (current.st_dev, current.st_ino) == (st.st_dev, st.st_ino):
            path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _same_inode(path: Path, fileno: int) -> bool:
    try:
        st_path = os.stat(path)
        st_fd = os.fstat(fileno)
    except FileNotFoundError:
        return False
    return (st_path.st_dev, st_path.st_ino) == (st_fd.st_dev, st_fd.st_ino)


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    poll_seconds: float = 0.05,
    timeout_seconds: float | None = None,
    remove_on_exit: bool = False,
    fallback_excl_on_notsup: bool = False,
    on_wait=None,
    try_lock=None,
    monotonic=None,
    sleep=None,
    open_persistent=None,
):
    """Acquire a process-death-safe exclusive lock on `path`.

    A persistent-lock caller may supply a guarded opener that also creates its parent.
    """
    path = Path(path)
    if open_persistent is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif remove_on_exit or fallback_excl_on_notsup:
        raise ValueError("a guarded opener requires a persistent OS lock")
    lock_fn = try_lock_handle if try_lock is None else try_lock
    clock = time.monotonic if monotonic is None else monotonic
    sleeper = time.sleep if sleep is None else sleep
    deadline = None if timeout_seconds is None else (clock() + timeout_seconds)

    if not remove_on_exit and not fallback_excl_on_notsup:
        with (open(path, "a+b") if open_persistent is None else open_persistent(path)) as handle:
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            wait_count = 0
            while not lock_fn(handle):
                if on_wait is not None:
                    on_wait(wait_count)
                wait_count += 1
                if deadline is not None and clock() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock {path}")
                sleeper(poll_seconds)
            try:
                yield
            finally:
                unlock_handle(handle)
        return

    if os.name == "nt":
        wait_count = 0
        with open(path, "a+b") as handle:
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            while not lock_fn(handle):
                if on_wait is not None:
                    on_wait(wait_count)
                wait_count += 1
                if deadline is not None and clock() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock {path}")
                sleeper(poll_seconds)
            try:
                yield
            finally:
                unlock_handle(handle)
        if remove_on_exit:
            try:
                path.unlink()
            except OSError:
                pass
        return

    # POSIX path with `remove_on_exit` / optional `ENOTSUP` PID-aware `O_CREAT | O_EXCL` fallback.
    wait_count = 0
    while True:
        created_exclusively = False
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            created_exclusively = True
        except FileExistsError:
            if on_wait is not None:
                on_wait(wait_count)
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                continue
        handle = os.fdopen(fd, "r+b")
        try:
            try:
                locked = lock_fn(handle)
            except OSError as error:
                if fallback_excl_on_notsup and error.errno in POSIX_LOCK_UNSUPPORTED:
                    if created_exclusively:
                        # We already hold the exclusive `O_CREAT | O_EXCL` lockfile; record our
                        # PID so a future publisher can recover if we die before unlinking.
                        handle.seek(0)
                        handle.write(f"{os.getpid()}\n".encode("ascii"))
                        handle.flush()
                        os.fsync(handle.fileno())
                        try:
                            yield
                        finally:
                            try:
                                handle.close()
                            finally:
                                if remove_on_exit:
                                    try:
                                        path.unlink()
                                    except FileNotFoundError:
                                        pass
                        return
                    handle.close()
                    if _reclaim_if_dead_owner(path, time.time()):
                        continue
                    if on_wait is not None:
                        on_wait(wait_count)
                    wait_count += 1
                    if deadline is not None and clock() >= deadline:
                        raise TimeoutError(f"timed out waiting for lock {path}")
                    sleeper(poll_seconds)
                    continue
                handle.close()
                if created_exclusively and remove_on_exit:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                raise
            if not locked:
                handle.close()
                if on_wait is not None:
                    on_wait(wait_count)
                wait_count += 1
                if deadline is not None and clock() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock {path}")
                sleeper(poll_seconds)
                continue
            if remove_on_exit and not _same_inode(path, handle.fileno()):
                try:
                    unlock_handle(handle)
                finally:
                    handle.close()
                if on_wait is not None:
                    on_wait(wait_count)
                continue
            try:
                yield
            finally:
                try:
                    if remove_on_exit:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
                finally:
                    try:
                        unlock_handle(handle)
                    finally:
                        handle.close()
            return
        except BaseException:
            if not handle.closed:
                handle.close()
            raise
