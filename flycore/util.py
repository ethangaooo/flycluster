"""通用工具：确定性随机源、内容哈希、JSON 序列化。"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


@contextmanager
def blas_quiet() -> Iterator[None]:
    """屏蔽 BLAS 在部分形状/dtype 组合上误报的浮点标志位。

    Apple Accelerate 等后端会对完全正常的 float32 matmul 设置 overflow / invalid /
    divide-by-zero 标志，触发 RuntimeWarning 而结果仍然有限。这些噪声会淹没真正的
    数值异常，所以这里**只**屏蔽这三个标志；调用方仍须自行校验结果有界。
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        yield


def rng_for(*parts: Any) -> np.random.Generator:
    """由任意可序列化片段派生确定性随机源。

    同一个 (seed, tag) 组合在任何机器、任何进程上都必须产生同一序列，
    否则"相同版本、种子和输入可重复结果"这道门就无法通过。
    """
    raw = json.dumps([str(p) for p in parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def sha256_of(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def graph_checksum(src: np.ndarray, dst: np.ndarray, weight: np.ndarray) -> str:
    order = np.lexsort((dst, src))
    h = hashlib.sha256()
    for arr in (src[order], dst[order]):
        h.update(np.ascontiguousarray(arr.astype(np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.round(weight[order], 6).astype(np.float32)).tobytes())
    return h.hexdigest()


def content_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    """ISO-8601 时间戳，**毫秒精度**且可直接字符串排序。

    必须有毫秒精度：一次运行每秒产生几十条经验，快照、写入、过期可能落在同一秒内。
    秒级精度会让 valid_to 与快照时间相等，"快照回溯"和"过期排除"的判定就会互相打架。
    """
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{ms:03d}" + time.strftime("%z", lt)


def now_ms() -> float:
    return time.time() * 1000.0


def jdump(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=_default)


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


class Telemetry:
    """滚动遥测累加器。任何性能数字都必须与它同时上报。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.sim_ms = 0.0
        self.wall_ms = 0.0
        self.spikes = 0
        self.dropped_spikes = 0
        self.nan_events = 0
        self.restarts = 0
        self.saturated_steps = 0

    def observe(self, *, sim_ms: float, wall_ms: float, spikes: int,
                dropped: int, nan_events: int = 0, saturated: int = 0) -> None:
        self.steps += 1
        self.sim_ms += sim_ms
        self.wall_ms += wall_ms
        self.spikes += spikes
        self.dropped_spikes += dropped
        self.nan_events += nan_events
        self.saturated_steps += saturated

    @property
    def realtime_ratio(self) -> float:
        """模拟时间 / 现实时间。>1 表示快于实时。"""
        if self.wall_ms <= 0:
            return 0.0
        return self.sim_ms / self.wall_ms

    @property
    def drop_rate(self) -> float:
        total = self.spikes + self.dropped_spikes
        return (self.dropped_spikes / total) if total else 0.0

    def as_dict(self, neurons: int = 0) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "sim_ms": round(self.sim_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "realtime_ratio": round(self.realtime_ratio, 4),
            "steps_per_second": round(self.steps / (self.wall_ms / 1000.0), 2) if self.wall_ms > 0 else 0.0,
            "spikes": self.spikes,
            "dropped_spikes": self.dropped_spikes,
            "drop_rate": round(self.drop_rate, 6),
            "mean_spikes_per_neuron_step": round(self.spikes / max(1, self.steps * neurons), 6) if neurons else 0.0,
            "nan_events": self.nan_events,
            "restarts": self.restarts,
            "saturated_steps": self.saturated_steps,
        }
