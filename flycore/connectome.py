"""连接组层：加载 / 合成 / 重连对照 / 子图提取。

四种图来源：
  synthetic  — 程序化合成，对齐重尾度分布、模块化、富俱乐部、递归性、抑制比例
  imported   — 外部 .npz（FlyWire / flychip / MaleCNS 子图，经 tools/fetch_connectome.py 产出）
  rewired    — 保度序列双边交换（Maslov–Sneppen）得到的对照图

公平对照纪律：重连图保持节点数、二值化入/出度序列、抑制神经元集合（符号随源神经元
天然保持）、权重多重集合（重排而非重抽），并重新校验输入→输出可达性；不可达则换
种子重试并记录接受轮次。任何一处做不到，都必须写进 meta.notes。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .schemas import ConnectomeMeta
from .util import graph_checksum, jdump, now_iso, rng_for

# 合成图的目标统计量：对齐 FlyWire 全脑公开统计的近似，用于生成"像果蝇"的拓扑
DEFAULT_MEAN_INDEGREE = 40.0
DEFAULT_INHIBITORY_FRACTION = 0.12
DEFAULT_WITHIN_MODULE_P = 0.70
DEFAULT_MODULES = 8
DEFAULT_HUB_FRACTION = 0.05
DEFAULT_HUB_GAIN = 6.0
DEFAULT_WEIGHT_SIGMA = 0.55
DEFAULT_IO_BACKBONE_EDGES = 0.05     # 上行感觉→运动通路占总边数的比例
DEFAULT_ASCENDING_PER_OUTPUT = 32    # 每个读出神经元直接接收的上行输入条数
DEFAULT_ASCENDING_GAIN = 4.0         # 上行突触相对平均突触的权重倍数
SENSORY_MODULE = 0
MOTOR_MODULE = 1


@dataclass
class NeuronGraph:
    """一张神经元连接图。CSR 惰性构造，避免重复转置。"""

    n: int
    src: np.ndarray          # int32 (m,) 预神经元
    dst: np.ndarray          # int32 (m,) 后神经元
    weight: np.ndarray       # float32 (m,) 幅值，恒正
    sign: np.ndarray         # float32 (m,) 预神经元符号：+1 兴奋 / -1 抑制
    inhibitory: np.ndarray   # bool (n,)
    input_ids: np.ndarray    # int32 感觉输入组
    output_ids: np.ndarray   # int32 动作读出组
    meta: ConnectomeMeta
    module: np.ndarray | None = None
    _w: sparse.csr_matrix | None = field(default=None, repr=False, compare=False)

    @property
    def signed_weights(self) -> np.ndarray:
        return self.weight * self.sign

    def matrix(self) -> sparse.csr_matrix:
        """邻接矩阵 A，A[src, dst] = 带符号权重。

        这个方向是给**图算法**用的：行向量右乘 `f @ A` 才表示"沿边向后继传播"，
        可达闭包与连通分量分析都依赖它。
        """
        if self._w is None:
            csr = sparse.csr_matrix(
                (self.signed_weights.astype(np.float32), (self.src, self.dst)),
                shape=(self.n, self.n),
            )
            csr.sum_duplicates()
            self._w = csr
        return self._w

    def propagation_matrix(self) -> sparse.csr_matrix:
        """突触传播矩阵 W，W[dst, src] = 带符号权重。

        脉冲仿真必须用这个方向：`W @ s` 的第 i 个分量是"所有预神经元本步发放
        对神经元 i 的突触电流贡献"。它等于 `matrix().T`，但直接构造更省一次转置。
        用错方向会让脉冲沿反向传播，症状是"输入对输出毫无影响"。
        """
        csr = sparse.csr_matrix(
            (self.signed_weights.astype(np.float32), (self.dst, self.src)),
            shape=(self.n, self.n),
        )
        csr.sum_duplicates()
        return csr

    def degree_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        """二值化出/入度（去重后计数）。"""
        out = np.bincount(self.src, minlength=self.n).astype(np.int32)
        inn = np.bincount(self.dst, minlength=self.n).astype(np.int32)
        return out, inn

    def edge_count(self) -> int:
        return int(self.src.size)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            src=self.src, dst=self.dst, weight=self.weight, sign=self.sign,
            inhibitory=self.inhibitory, input_ids=self.input_ids, output_ids=self.output_ids,
            module=self.module if self.module is not None else np.zeros(0, dtype=np.int32),
            meta=np.array([jdump(self.meta.model_dump(), indent=None)], dtype=object),
        )
        return path


# ------------------------------------------------------------------ 统计量


def _csr(src, dst, sign, weight, n) -> sparse.csr_matrix:
    csr = sparse.csr_matrix((weight * sign, (src, dst)), shape=(n, n))
    csr.sum_duplicates()
    return csr


def _stats(src, dst, inhibitory, module, n) -> dict[str, float]:
    out, inn = np.bincount(src, minlength=n), np.bincount(dst, minlength=n)
    adj = sparse.csr_matrix((np.ones(src.size, np.float32), (src, dst)), shape=(n, n))
    _, labels = csgraph.connected_components(adj, directed=True, connection="strong")
    scc = np.bincount(labels)
    modularity = 0.0
    if module is not None and module.size == n and module.max() > 0:
        k = int(module.max()) + 1
        m = max(1, src.size)
        e = np.zeros((k, k))
        np.add.at(e, (module[src], module[dst]), 1.0)
        e /= m
        a = e.sum(axis=1)
        modularity = float(np.trace(e) - np.sum(a * a))
    return {
        "mean_indegree": float(inn.mean()),
        "mean_outdegree": float(out.mean()),
        "largest_scc_fraction": float(scc.max() / n) if n else 0.0,
        "modularity": modularity,
        "inhibitory_fraction": float(inhibitory.mean()),
    }


def io_reachable(matrix: sparse.csr_matrix, input_ids: np.ndarray, output_ids: np.ndarray,
                 hops: int = 12) -> bool:
    """输入组能否在 hops 跳内到达输出组。用 binarize 后的稀疏乘幂做可达闭包。"""
    if input_ids.size == 0 or output_ids.size == 0:
        return False
    adj = (matrix != 0).astype(np.float32)
    frontier = sparse.csr_matrix(
        (np.ones(input_ids.size, np.float32), (np.zeros(input_ids.size, np.int32), input_ids)),
        shape=(1, matrix.shape[0]),
    )
    seen = np.zeros(matrix.shape[0], dtype=bool)
    for _ in range(hops):
        frontier = (frontier @ adj).astype(np.float32)
        if frontier.indices.size == 0:
            break
        seen[frontier.indices] = True
        frontier.data[:] = 1.0
    return bool(seen[output_ids].any())


# ------------------------------------------------------------------ 合成


def generate_synthetic_connectome(
    n_neurons: int = 2000,
    seed: int = 0,
    modules: int = DEFAULT_MODULES,
    mean_indegree: float = DEFAULT_MEAN_INDEGREE,
    inhibitory_fraction: float = DEFAULT_INHIBITORY_FRACTION,
    within_module_p: float = DEFAULT_WITHIN_MODULE_P,
    hub_fraction: float = DEFAULT_HUB_FRACTION,
    hub_gain: float = DEFAULT_HUB_GAIN,
    weight_sigma: float = DEFAULT_WEIGHT_SIGMA,
    io_backbone_edges: float = DEFAULT_IO_BACKBONE_EDGES,
    ascending_per_output: int = DEFAULT_ASCENDING_PER_OUTPUT,
    ascending_gain: float = DEFAULT_ASCENDING_GAIN,
    n_input: int = 96,
    n_output: int = 36,
    name: str | None = None,
) -> NeuronGraph:
    """生成"连接组约束"的合成子图。

    刻意对齐的拓扑特征（**不是真实连接数据**，报告必须如实标注）：
      重尾入度、模块化（模块内连接概率显著高于跨模块）、富俱乐部（少数高连接度枢纽）、
      高递归（大强连通分量）、约 12% 抑制性神经元，以及一条显式的
      感觉组 → 读出组 上行通路。

    为什么必须有显式上行通路：若输入到读出之间只靠多跳递归传播，感觉扰动会被
    2000 个神经元的平均效应吞掉，任务在结构上无法完成——那不是"结构劣势"，
    而是把模型建坏了。真实果蝇连接组中，感觉上行神经元到下降神经元存在明确、
    相对直接的通路。因此这里按比例构造一条上行通路；**保度序列重连会保留每个
    读出神经元的入度，但破坏其预神经元的具体身份**，即破坏"输入是否来自感觉区"
    这一结构信息。这正是「真实结构 vs 重连结构」对照要检验的差异。
    """
    rng = rng_for("synthetic", seed, n_neurons, modules)
    n = int(n_neurons)
    module = rng.integers(0, modules, n).astype(np.int32)
    n_io = min(n, max(2 * (n_input + n_output), int(0.02 * n)))
    module[:n_io] = np.where(np.arange(n_io) % 2 == 0, SENSORY_MODULE, MOTOR_MODULE)
    members = [np.flatnonzero(module == k) for k in range(modules)]
    members = [m for m in members if m.size]

    inhibitory = np.zeros(n, dtype=bool)
    pool = np.flatnonzero((module != SENSORY_MODULE) & (module != MOTOR_MODULE))
    n_inh = int(round(inhibitory_fraction * n))
    if pool.size and n_inh > 0:
        inhibitory[rng.choice(pool, size=min(n_inh, pool.size), replace=False)] = True

    hub_score = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    n_hub = max(1, int(hub_fraction * n))
    hubs = np.argsort(hub_score)[-n_hub:]
    boost = np.ones(n)
    boost[hubs] = hub_gain
    pick_p = boost / boost.sum()

    indeg = np.clip(
        rng.lognormal(mean=np.log(max(2.0, mean_indegree)), sigma=0.85, size=n),
        3, 8 * mean_indegree,
    ).astype(np.int64)
    dst = np.repeat(np.arange(n, dtype=np.int32), indeg)
    m = dst.size

    same = rng.random(m) < within_module_p
    src = np.empty(m, dtype=np.int32)
    filled = np.zeros(m, dtype=bool)
    same_idx = np.flatnonzero(same)
    if same_idx.size:
        mod_of_dst = module[dst[same_idx]]
        for k, mem in enumerate(members):
            sel = same_idx[mod_of_dst == k]
            if sel.size:
                src[sel] = rng.choice(mem, size=sel.size, replace=True).astype(np.int32)
                filled[sel] = True
    rest = np.flatnonzero(~filled)
    if rest.size:
        src[rest] = rng.choice(n, size=rest.size, p=pick_p).astype(np.int32)

    # 感觉模块 → 运动模块上行骨架（模块级）
    sens_pool, moto_pool = members[SENSORY_MODULE], members[MOTOR_MODULE]
    n_bone = max(n_input, int(io_backbone_edges * (indeg.sum() + 1)))
    bone_src = rng.choice(sens_pool, size=n_bone, replace=True)
    bone_dst = rng.choice(moto_pool, size=n_bone, replace=True)
    src = np.concatenate([src, bone_src.astype(np.int32)])
    dst = np.concatenate([dst, bone_dst.astype(np.int32)])

    # 显式上行通路：感觉组 → 读出组，每个读出神经元固定数量的上行输入。
    # 上行突触单独乘以 ascending_gain：果蝇中感觉上行神经元对下降神经元的驱动
    # 显著强于平均突触；若不做这一步，读出组会被局部递归活动淹没，任务在结构上
    # 无法完成。保度重连会保留每个读出神经元的入度，但抹掉"预神经元是否来自感觉组"
    # 这一身份信息——这正是本对照要检验的结构差异。
    input_ids = rng.choice(sens_pool, size=min(n_input, sens_pool.size), replace=False).astype(np.int32)
    output_ids = rng.choice(moto_pool, size=min(n_output, moto_pool.size), replace=False).astype(np.int32)
    asc_dst = np.repeat(output_ids, ascending_per_output)
    asc_src = rng.choice(input_ids, size=asc_dst.size, replace=True).astype(np.int32)

    n_base = src.size
    src = np.concatenate([src, asc_src])
    dst = np.concatenate([dst, asc_dst])
    gain_flag = np.concatenate([np.zeros(n_base, dtype=bool), np.ones(asc_src.size, dtype=bool)])

    keep = src != dst
    src, dst, gain_flag = src[keep], dst[keep], gain_flag[keep]
    _, uniq = np.unique(src.astype(np.int64) * n + dst, return_index=True)
    uniq.sort()
    src, dst, gain_flag = src[uniq].astype(np.int32), dst[uniq].astype(np.int32), gain_flag[uniq]

    sign = np.where(inhibitory[src], -1.0, 1.0).astype(np.float32)
    weight = rng.lognormal(mean=-0.5, sigma=weight_sigma, size=src.size).astype(np.float32)
    weight[gain_flag] *= ascending_gain
    weight = weight.astype(np.float32)

    reach = io_reachable(_csr(src, dst, sign, weight, n), input_ids, output_ids)
    meta = ConnectomeMeta(
        name=name or f"synthetic-{n}-s{seed}",
        source="synthetic",
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        neuron_count=n,
        edge_count=int(src.size),
        synapse_count=None,
        includes_optic_lobe=False,
        io_reachable=bool(reach),
        input_count=int(input_ids.size),
        output_count=int(output_ids.size),
        checksum=graph_checksum(src, dst, weight),
        created_at=now_iso(),
        notes=(
            "程序化生成的连接组约束子图：对齐重尾入度、模块化、富俱乐部、高递归、"
            f"约 {inhibitory_fraction:.0%} 抑制比例。非任何真实果蝇连接数据；"
            "神经元动力学、递质符号、权重分布均为人为建模选择。"
        ),
        **_stats(src, dst, inhibitory, module, n),
    )
    return NeuronGraph(n, src, dst, weight, sign, inhibitory, input_ids, output_ids, meta, module)


# ------------------------------------------------------------------ 重连对照


def rewire_matched(
    graph: NeuronGraph,
    seed: int = 0,
    swap_multiplier: int = 10,
    max_attempts: int = 25,
    require_io_reachable: bool = True,
) -> NeuronGraph:
    """保度序列重连：有向图 Maslov–Sneppen 双边交换。

    保持：节点数、二值化入度序列、二值化出度序列、抑制神经元集合（符号随源神经元走）、
          权重多重集合（整体重排，不重新抽样）。
    校验：重连后输入→输出可达性；不可达则换种子重试，并记录接受轮次与交换次数。
    """
    n = graph.n
    for attempt in range(max_attempts):
        rng = rng_for("rewire", seed, attempt, graph.meta.checksum)
        src_l = graph.src.tolist()
        dst_l = graph.dst.tolist()
        m = len(src_l)
        edges = set(zip(src_l, dst_l))
        target = swap_multiplier * m
        swaps = 0
        probe = 0
        max_probe = 300 * target + 1000
        while swaps < target and probe < max_probe:
            probe += 1
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m))
            if i == j:
                continue
            a, b, c, d = src_l[i], dst_l[i], src_l[j], dst_l[j]
            if a == d or c == b or a == c or b == d:
                continue
            if (a, d) in edges or (c, b) in edges:
                continue
            edges.discard((a, b))
            edges.discard((c, d))
            edges.add((a, d))
            edges.add((c, b))
            src_l[i], dst_l[i] = a, d
            src_l[j], dst_l[j] = c, b
            swaps += 1

        src = np.asarray(src_l, dtype=np.int32)
        dst = np.asarray(dst_l, dtype=np.int32)
        weight = graph.weight[rng.permutation(graph.weight.size)].astype(np.float32)
        sign = np.where(graph.inhibitory[src], -1.0, 1.0).astype(np.float32)

        reach = io_reachable(_csr(src, dst, sign, weight, n), graph.input_ids, graph.output_ids)
        if reach or not require_io_reachable:
            meta = graph.meta.model_copy(deep=True)
            meta.name = f"{graph.meta.name}-rewired"
            meta.checksum = graph_checksum(src, dst, weight)
            meta.derived_from = graph.meta.checksum
            meta.rewire_swaps = swaps
            meta.io_reachable = bool(reach)
            meta.created_at = now_iso()
            meta.notes = (
                f"保度序列重连对照（有向 Maslov–Sneppen，完成双边交换 {swaps} 次，"
                f"接受轮次 {attempt + 1}/{max_attempts}）。入/出度序列、抑制神经元集合、"
                "权重多重集合均与父图一致；权重为多重集合重排（分布一致、配对不同）。"
            )
            for k, v in _stats(src, dst, graph.inhibitory, graph.module, n).items():
                setattr(meta, k, v)
            return NeuronGraph(n, src, dst, weight, sign, graph.inhibitory,
                               graph.input_ids, graph.output_ids, meta, graph.module)
    raise RuntimeError(
        f"重连失败：{max_attempts} 轮内未得到输入→输出可达的保度图；"
        "请放宽 hops 或提高 max_attempts，不要降低可达性要求。"
    )


# ------------------------------------------------------------------ 子图提取


def _reach_sets(adj: sparse.csr_matrix, seeds: np.ndarray, hops: int) -> np.ndarray:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    f = sparse.csr_matrix(
        (np.ones(seeds.size, np.float32), (np.zeros(seeds.size, np.int32), seeds)), shape=(1, n)
    )
    for _ in range(hops):
        f = (f @ adj).astype(np.float32)
        if f.indices.size == 0:
            break
        seen[f.indices] = True
        f.data[:] = 1.0
    return seen


def extract_subgraph(graph: NeuronGraph, hops: int = 6, max_nodes: int = 0,
                     seed: int = 0) -> NeuronGraph:
    """保留"输入→输出"通路及其周边局部循环。

    构思文档要求：选取时优先保留从输入到输出的路径和局部循环，
    不能任意删除节点后仍称其为原有生物回路。
    """
    n = graph.n
    adj = (graph.matrix() != 0).astype(np.float32)
    fwd = _reach_sets(adj, graph.input_ids, hops)
    bwd = _reach_sets(adj.T.tocsr(), graph.output_ids, hops)
    forced = np.zeros(n, dtype=bool)
    forced[graph.input_ids] = True
    forced[graph.output_ids] = True
    keep = forced | (fwd & bwd)
    if max_nodes and keep.sum() > max_nodes:
        deg = np.bincount(graph.src, minlength=n) + np.bincount(graph.dst, minlength=n)
        rest = np.flatnonzero(keep & ~forced)
        take = rest[np.argsort(-deg[rest])][: max(0, max_nodes - int(forced.sum()))]
        keep = forced.copy()
        keep[take] = True

    remap = -np.ones(n, dtype=np.int64)
    idx = np.flatnonzero(keep)
    remap[idx] = np.arange(idx.size)
    ek = keep[graph.src] & keep[graph.dst]
    src = remap[graph.src[ek]].astype(np.int32)
    dst = remap[graph.dst[ek]].astype(np.int32)
    weight = graph.weight[ek].astype(np.float32)
    inhibitory = graph.inhibitory[idx]
    sign = np.where(inhibitory[src], -1.0, 1.0).astype(np.float32)
    module = graph.module[idx] if graph.module is not None else None

    meta = graph.meta.model_copy(deep=True)
    meta.name = f"{graph.meta.name}-sub{idx.size}"
    meta.neuron_count = int(idx.size)
    meta.edge_count = int(src.size)
    meta.checksum = graph_checksum(src, dst, weight)
    meta.derived_from = graph.meta.checksum
    meta.created_at = now_iso()
    meta.notes = f"由 {graph.meta.name} 按 输入↔输出 {hops} 跳通路提取（保留局部循环，seed={seed}）。"
    for k, v in _stats(src, dst, inhibitory, module, idx.size).items():
        setattr(meta, k, v)
    in_new, out_new = remap[graph.input_ids].astype(np.int32), remap[graph.output_ids].astype(np.int32)
    meta.input_count, meta.output_count = int(in_new.size), int(out_new.size)
    meta.io_reachable = io_reachable(_csr(src, dst, sign, weight, idx.size), in_new, out_new)
    return NeuronGraph(idx.size, src, dst, weight, sign, inhibitory, in_new, out_new, meta, module)


# ------------------------------------------------------------------ 持久化与注册表


def load_graph(path: str | Path) -> NeuronGraph:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        meta = ConnectomeMeta(**json.loads(str(z["meta"][0])))
        module = z["module"] if z["module"].size == int(meta.neuron_count) else None
        return NeuronGraph(
            n=int(meta.neuron_count), src=z["src"].astype(np.int32), dst=z["dst"].astype(np.int32),
            weight=z["weight"].astype(np.float32), sign=z["sign"].astype(np.float32),
            inhibitory=z["inhibitory"].astype(bool), input_ids=z["input_ids"].astype(np.int32),
            output_ids=z["output_ids"].astype(np.int32), meta=meta, module=module,
        )


_REGISTRY: dict[str, NeuronGraph] = {}


def parse_spec(spec: str) -> tuple[str, int, int]:
    """解析 `synthetic-<n>[-s<seed>]` / `synthetic-<n>-rewired[-s<seed>]`。"""
    parts = spec.split("-")
    if not spec.startswith("synthetic"):
        raise ValueError(spec)
    rewired = "rewired" in parts
    nums = [p for p in parts if p.isdigit()]
    seed = 0
    for p in parts:
        if p.startswith("s") and p[1:].isdigit():
            seed = int(p[1:])
    return ("rewired" if rewired else "plain"), (int(nums[0]) if nums else 2000), seed


def get_graph(spec: str, data_dir: str | Path = "data") -> NeuronGraph:
    """按规格取图，带进程内缓存。

    规格：
      `synthetic-<n>[-s<seed>][-rewired]`   程序化合成
      `<name>` / `<name>.npz` / `<path>`    磁盘上的真实连接组子图
      `<name>-rewired`                      对磁盘图做保度序列重连（对照用）

    外部文件不存在时**报错**，不会静默回退成合成图——避免把合成数据当真实数据用。
    """
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    data_dir = Path(data_dir)

    def resolve(name: str) -> Path | None:
        cand = Path(name)
        if cand.exists():
            return cand
        cand = data_dir / (name if name.endswith(".npz") else f"{name}.npz")
        return cand if cand.exists() else None

    path = resolve(spec)
    if path is not None:
        g = load_graph(path)
    elif spec.endswith("-rewired") and resolve(spec[: -len("-rewired")]) is not None:
        # 真实连接组的重连对照：先载父图，再做保度序列重连。
        # 只靠 `-rewired` 后缀找不到父图时会走下面的 synthetic 分支并报错，
        # 因此这里必须显式处理，否则"真实结构 vs 重连结构"的对照根本跑不起来。
        base = load_graph(resolve(spec[: -len("-rewired")]))  # type: ignore[arg-type]
        g = rewire_matched(base, seed=991)
    elif spec.startswith("synthetic"):
        kind, n, seed = parse_spec(spec)
        g = generate_synthetic_connectome(n_neurons=n, seed=seed)
        if kind == "rewired":
            g = rewire_matched(g, seed=seed + 991)
    else:
        raise FileNotFoundError(
            f"找不到连接组 {spec}（已查找 {data_dir / (spec + '.npz')}）。"
            "请先运行 tools/fetch_connectome.py，或使用 synthetic-<n> 规格。"
        )
    _REGISTRY[spec] = g
    return g


def registry_clear() -> None:
    _REGISTRY.clear()
