# Thread-safe mini queue for freshly inserted keys (ids)
from collections import deque
from threading import RLock
from typing import Optional, Any

_q = deque()
_lock = RLock()

def push(key: Any) -> None:
    with _lock:
        _q.append(key)

def pop_last() -> Optional[Any]:
    with _lock:
        return _q.pop() if _q else None
