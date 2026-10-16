#!/usr/bin/env python3
"""把真实果蝇连接组接入实验：官方 neuPrint API 与本地导出文件两条通道。

用法
----
探测（不改任何文件，只报告能查到什么）::

    export NEUPRINT_TOKEN=xxxx          # 在 neuprint.janelia.org 账户页生成
    python tools/fetch_connectome.py probe --dataset male-cns:v1.0

在线抓取（默认走官方 API）::

    python tools/fetch_connectome.py online --dataset male-cns:v1.0 \
        --out data/male-cns-gateway.npz --descending-regex '^DN.*' --ascending-regex '^AN.*'

**neuPrint 的 type 正则是全串匹配**：`^DN` 匹配不到 `DN1a`，必须写 `^DN.*`。

离线导入（先在有网的机器上导出，或在本地保存好的文件）::

    python tools/fetch_connectome.py offline \
        --connections data/male-cns/connections.csv \
        --neurons data/male-cns/neurons.csv \
        --out data/male-cns-gateway.npz

设计纪律
--------
* **绝不静默回退**：拿不到真实数据就报错退出，不会偷偷用合成图冒充真实连接组。
* **许可随数据走**：数据集名、许可与来源写进 meta，落盘留痕。
* **口径如实**：weight 是聚合突触数而非连接边数；是否含视叶、是否含感觉末梢，
  由 `--include-optic-lobe` 等开关决定并写进 notes。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flycore.connectome import NeuronGraph, _stats, io_reachable  # noqa: E402
from flycore.schemas import ConnectomeMeta  # noqa: E402
from flycore.util import graph_checksum, jdump, now_iso  # noqa: E402

#  neurotransitter → 抑制性判定。果蝇中枢以 GABA 为主要抑制性递质；
# 谷氨酸在部分回路中为兴奋性，是否计入抑制由开关控制，默认不计入。
INHIBITORY_NT = {"gaba", "gad", "gad1"}

PRE_COLS = ("bodyId_pre", "bodyid_pre", "pre", "source", "from", "src", "presynaptic")
POST_COLS = ("bodyId_post", "bodyid_post", "post", "target", "to", "dst", "postsynaptic")
WEIGHT_COLS = ("weight", "syn_count", "syncount", "count", "n_syn", "nsyn", "synapses")
ID_COLS = ("bodyId", "bodyid", "id", "segment", "segmentid")
TYPE_COLS = ("type", "celltype", "cell_type", "typename")
INSTANCE_COLS = ("instance", "name", "label")
NT_COLS = ("predictedNt", "predicted_nt", "nt", "neurotransmitter", "predictednttype")
ROI_COLS = ("roi", "region", "brainregion")


def _find(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[抓取失败] {msg}")


# --------------------------------------------------------------------- 读入


def read_tables(connections: str, neurons: str | None, glob_pat: bool = False):
    """读连接表与神经元注释表；列名自动识别，识别不出就报错而不是猜。"""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("需要 pandas：pip install -r requirements-connectome.txt") from exc

    paths: list[str] = []
    for p in connections.split(","):
        p = p.strip()
        if not p:
            continue
        paths.extend(sorted(glob.glob(p)) if glob_pat else [p])
    _require(bool(paths), f"找不到连接表文件：{connections}")

    frames = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            raise SystemExit(f"[抓取失败] 文件不存在：{p}")
        if fp.suffix.lower() in (".parquet", ".pq"):
            frames.append(pd.read_parquet(fp))
        else:
            frames.append(pd.read_csv(fp, low_memory=False))
    conn = pd.concat(frames, ignore_index=True)

    pre = _find(list(conn.columns), PRE_COLS)
    post = _find(list(conn.columns), POST_COLS)
    wcol = _find(list(conn.columns), WEIGHT_COLS)
    _require(pre is not None and post is not None,
             f"连接表里认不出 pre/post 列。实际列名：{list(conn.columns)}")

    nrn = None
    if neurons:
        fp = Path(neurons)
        _require(fp.exists(), f"神经元注释表不存在：{neurons}")
        nrn = pd.read_parquet(fp) if fp.suffix.lower() in (".parquet", ".pq") else pd.read_csv(fp, low_memory=False)
        _require(_find(list(nrn.columns), ID_COLS) is not None,
                 f"注释表里认不出 bodyId 列。实际列名：{list(nrn.columns)}")
    return conn, nrn, pre, post, wcol


def annotate(conn, nrn, pre: str, post: str, wcol: str | None, treat_glu_inhibitory: bool):
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit("需要 pandas：pip install -r requirements-connectome.txt") from exc

    agg = {wcol: "sum"} if wcol else {}
    if any(r.lower() in ROI_COLS for r in conn.columns):
        roi = _find(list(conn.columns), ROI_COLS)
        group = [pre, post, roi]
    else:
        group = [pre, post]
    g = conn.groupby(group, as_index=False).agg(agg) if agg else conn.groupby(group, as_index=False).size()
    if not wcol:
        g = g.rename(columns={"size": "weight"})
        wcol = "weight"

    info: dict[int, dict[str, str]] = {}
    if nrn is not None:
        idc = _find(list(nrn.columns), ID_COLS)
        tc = _find(list(nrn.columns), TYPE_COLS)
        ic = _find(list(nrn.columns), INSTANCE_COLS)
        nc = _find(list(nrn.columns), NT_COLS)
        for row in nrn[[c for c in (idc, tc, ic, nc) if c]].itertuples(index=False):
            vals = dict(zip([c for c in (idc, tc, ic, nc) if c], row))
            bid = int(vals[idc])
            nt = str(vals.get(nc, "") or "").strip().lower()
            info[bid] = {"type": str(vals.get(tc, "") or ""), "instance": str(vals.get(ic, "") or ""),
                         "nt": nt}
    if treat_glu_inhibitory:
        for v in info.values():
            if v["nt"].startswith("glu"):
                v["nt"] = "gaba"
    return g, info, wcol


# --------------------------------------------------------------------- 选组


def select_groups(ids: np.ndarray, info: dict[int, dict[str, str]],
                  descending_regex: str, ascending_regex: str,
                  max_input: int, max_output: int) -> tuple[np.ndarray, np.ndarray]:
    down = re.compile(descending_regex)
    up = re.compile(ascending_regex)
    out_ids = np.array([i for i in ids if down.search(info.get(int(i), {}).get("type", ""))], dtype=np.int64)
    in_ids = np.array([i for i in ids if up.search(info.get(int(i), {}).get("type", ""))], dtype=np.int64)
    _require(out_ids.size > 0, f"没有神经元匹配 --descending-regex {descending_regex!r}；"
                              "用 probe 子命令看实际 type 名称")
    _require(in_ids.size > 0, f"没有神经元匹配 --ascending-regex {ascending_regex!r}；"
                             "用 probe 子命令看实际 type 名称")
    if out_ids.size > max_output:
        out_ids = np.random.default_rng(0).choice(out_ids, size=max_output, replace=False)
    if in_ids.size > max_input:
        in_ids = np.random.default_rng(1).choice(in_ids, size=max_input, replace=False)
    return np.sort(in_ids), np.sort(out_ids)


def keep_io_reachable(src, dst, w, inhibitor, in_new, out_new, max_drop: float = 0.35):
    """只在**输入→输出不可达**时，按权重从大到小保留边直至可达；本来就可达则一条不删。

    这个顺序不能反：如果把"最小化边集"当成目标，本来就可达的图会被砍到只剩
    一条最短通路，891 个节点的子图会退化成平均入度 0.59 的树——递归结构、
    富俱乐部、局部循环全部消失，那样的"真实连接组"拿去做实验毫无意义。
    """
    from scipy import sparse

    def make(keep_mask: np.ndarray):
        n = int(max(src.max(), dst.max())) + 1
        return sparse.csr_matrix((np.ones(int(keep_mask.sum()), np.float32),
                                 (src[keep_mask], dst[keep_mask])), shape=(n, n))

    full = np.ones(src.size, dtype=bool)
    if io_reachable(make(full), in_new, out_new, hops=10):
        print(f"[可达] 全量 {src.size} 条边下输入→输出已可达，不做任何剪枝")
        return full

    order = np.argsort(-w)
    lo, hi = 1, src.size
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        keep = np.zeros(src.size, dtype=bool)
        keep[order[:mid]] = True
        if io_reachable(make(keep), in_new, out_new, hops=10):
            best = keep
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        print("[警告] 即使保留全部边，输入→输出在 10 跳内仍不可达；"
              "请放宽选组范围或增大中间神经元上限，不要降低可达性要求")
        return full
    dropped = 1.0 - best.mean()
    print(f"[可达] 原图不可达：按权重保留 {best.sum()}/{src.size} 条边后可达（丢弃 {dropped:.1%}）")
    if dropped > max_drop:
        print(f"[警告] 丢弃比例 {dropped:.1%} 超过 {max_drop:.0%} 阈值，"
              "结果应以「结构已被大幅裁剪」为前提解读")
    return best


# --------------------------------------------------------------------- 在线


def fetch_annotations_cypher(client, ids: list[int], batch: int = 800):
    """用 Cypher 取神经元注释。

    不用 `fetch_neurons` 的返回值：它的字段随版本变动，而这里只要 4 个稳定属性。
    递质优先取 `predictedNt`，缺失时退回 `consensusNt`（male-cns 两者都有）。
    """
    import pandas as pd

    frames = []
    for i in range(0, len(ids), batch):
        chunk = ",".join(str(int(x)) for x in ids[i: i + batch])
        cypher = (
            "MATCH (n:Neuron) "
            f"WHERE n.bodyId IN [{chunk}] "
            "RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, "
            "coalesce(n.predictedNt, n.consensusNt) AS nt, n.class AS cls"
        )
        frames.append(client.fetch_custom(cypher))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["bodyId", "type", "instance", "nt", "cls"])


def fetch_edges_cypher(client, ids: list[int], batch: int = 120, progress=print):
    """用 Cypher 取子图内部的连接边。

    为什么不用 `fetch_adjacencies`：neuprint-python 0.3.5 内部有一步
    `totals_df.eval('weight_all - weight_primary')`，在 `NotPrimary` 行为空时会抛
    `TypeError: unsupported operand type(s) for -: 'object' and 'int64'`
    ——与数据无关、与参数无关，属上游缺陷。

    另一层好处：neuPrint 的边属性 `weight` **本身已是跨 ROI 聚合的突触数**
    （`roiInfo` 只是附加映射），所以 `sum(e.weight)` 就是我们要的口径，
    不必像客户端那样先按 ROI 展开再合并——少一步就少一个出错的地方。

    分批策略：按「预神经元落在本批」切分，配对集合始终是全体 id，因此跨批的边不会漏。
    """
    import pandas as pd

    all_ids = ",".join(str(int(x)) for x in ids)
    frames = []
    n_batch = max(1, (len(ids) + batch - 1) // batch)
    for b, i in enumerate(range(0, len(ids), batch), start=1):
        chunk = ",".join(str(int(x)) for x in ids[i: i + batch])
        cypher = (
            "MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) "
            f"WHERE a.bodyId IN [{chunk}] AND b.bodyId IN [{all_ids}] "
            "RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, "
            "sum(e.weight) AS weight, count(e) AS n_edges"
        )
        df = client.fetch_custom(cypher)
        progress(f"[在线]   边查询 {b}/{n_batch}：本批 {len(df)} 行")
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit("[抓取失败] 子图内部没有任何连接边。请放宽选组范围或降低 --min-synapses。")
    return pd.concat(frames, ignore_index=True)


def online(args: argparse.Namespace) -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit("需要 pandas：pip install -r requirements-connectome.txt") from exc
    try:
        from neuprint import Client, fetch_neurons
        from neuprint.queries import SegmentCriteria
    except ImportError as exc:
        raise SystemExit("需要 neuprint-python：pip install -r requirements-connectome.txt") from exc

    token = os.environ.get(args.token_env, "")
    _require(bool(token), f"未设置环境变量 {args.token_env}。请在 {args.server} 注册后于账户页生成 token 并导出。")

    client = Client(args.server, dataset=args.dataset, token=token)
    print(f"[在线] 已连接 {args.server} / dataset={args.dataset}")

    def pick(pattern: str, label: str, limit: int) -> np.ndarray:
        df, _ = fetch_neurons(SegmentCriteria(type=pattern, regex=True), client=client)
        print(f"[在线] {label}（type =~ {pattern!r}）命中 {len(df)} 个神经元")
        if len(df) == 0:
            raise SystemExit(
                f"[抓取失败] 正则 {pattern!r} 没有匹配到任何神经元。\n"
                "  neuPrint 的 type 正则是**全串匹配**：`^DN` 匹配不到 `DN1a`，必须写 `^DN.*`。\n"
                "  先用 probe 子命令确认该数据集的真实类型命名（male-cns:v1.0 用 DN* / AN*）。")
        ids = df["bodyId"].astype(np.int64).to_numpy()
        if ids.size > limit:
            ids = np.random.default_rng(args.seed).choice(ids, size=limit, replace=False)
        return np.sort(ids)

    in_ids = pick(args.ascending_regex, "上行/感觉组", args.max_input)
    out_ids = pick(args.descending_regex, "下降/输出组", args.max_output)

    # 一跳中间神经元：同时接受上行组输入、并向下行组输出的神经元。
    # 用 Cypher 直接取，比"给种子集补前驱"更贴合"保留输入→输出通路"的要求。
    mids: list[int] = []
    if args.intermediate_limit > 0:
        a_list = ",".join(str(int(x)) for x in in_ids)
        d_list = ",".join(str(int(x)) for x in out_ids)
        cypher = ("MATCH (a:Neuron)-[:ConnectsTo]->(m:Neuron)-[:ConnectsTo]->(d:Neuron) "
                  f"WHERE a.bodyId IN [{a_list}] AND d.bodyId IN [{d_list}] "
                  f"RETURN DISTINCT m.bodyId AS bodyId LIMIT {int(args.intermediate_limit)}")
        try:
            mdf = client.fetch_custom(cypher)
            mids = [int(x) for x in mdf["bodyId"].tolist()]
            print(f"[在线] 位于 上行→中间→下降 路径上的中间神经元 {len(mids)} 个")
        except Exception as exc:
            print(f"[警告] 中间神经元查询失败（{type(exc).__name__}: {str(exc)[:140]}）；"
                  "退化为仅使用上行组与下降组的直接连接")

    node_ids = list(dict.fromkeys([int(x) for x in np.concatenate([in_ids, out_ids])] + mids))
    conn = fetch_edges_cypher(client, node_ids, batch=args.edge_batch)
    print(f"[在线] 节点 {len(node_ids)} 个；子图内部连接 {len(conn)} 行（weight 已含全部 ROI）")

    # 再补一跳：优先纳入下游，便于形成更深的输入→输出通路，同时引入真实递归结构
    for hop in range(max(0, args.expand_hops)):
        if len(node_ids) >= args.node_budget:
            break
        lst = ",".join(str(int(x)) for x in node_ids)
        try:
            down = client.fetch_custom(
                f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN [{lst}] "
                "RETURN b.bodyId AS bodyId, sum(e.weight) AS w ORDER BY w DESC "
                f"LIMIT {int(args.expand_batch)}")
            up = client.fetch_custom(
                f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) WHERE b.bodyId IN [{lst}] "
                "RETURN a.bodyId AS bodyId, sum(e.weight) AS w ORDER BY w DESC "
                f"LIMIT {int(args.expand_batch)}")
        except Exception as exc:
            print(f"[警告] 第 {hop + 1} 跳扩展失败（{str(exc)[:120]}），停止扩展")
            break
        extra = [int(x) for x in down["bodyId"].tolist()] + [int(x) for x in up["bodyId"].tolist()]
        known = set(node_ids)
        extra = [x for x in dict.fromkeys(extra) if x not in known]
        room = args.node_budget - len(node_ids)
        if room <= 0 or not extra:
            break
        add = extra[:room]
        node_ids = node_ids + add
        print(f"[在线] 第 {hop + 1} 跳扩展 {len(add)} 个节点 → 共 {len(node_ids)} 个")
        conn = fetch_edges_cypher(client, node_ids, batch=args.edge_batch)

    ann = fetch_annotations_cypher(client, node_ids)
    node_info: dict[int, dict[str, str]] = {}
    for r in ann.itertuples(index=False):
        d = r._asdict()
        node_info[int(d["bodyId"])] = {
            "type": str(d.get("type") or ""),
            "instance": str(d.get("instance") or ""),
            "nt": str(d.get("nt") or "").lower(),
            "cls": str(d.get("cls") or ""),
        }
    gaba = sum(1 for v in node_info.values() if v["nt"].startswith("gaba"))
    print(f"[在线] 注释覆盖 {len(node_info)}/{len(node_ids)} 个节点；GABA 抑制性 {gaba} 个")

    finalize(args, conn, node_info, in_ids, out_ids,
             dataset=args.dataset, source="male-cns" if "cns" in args.dataset else "flywire")


# --------------------------------------------------------------------- 离线


def offline(args: argparse.Namespace) -> None:
    conn, nrn, pre, post, wcol = read_tables(args.connections, args.neurons, args.glob)
    g, info, wcol = annotate(conn, nrn, pre, post, wcol, args.treat_glu_inhibitory)
    ids = np.union1d(g[pre].to_numpy(), g[post].to_numpy()).astype(np.int64)
    in_ids, out_ids = select_groups(ids, info, args.descending_regex, args.ascending_regex,
                                    args.max_input, args.max_output)
    print(f"[离线] 连接行 {len(g)}；节点 {ids.size}；上行组 {in_ids.size}；下降组 {out_ids.size}")
    finalize(args, g, info, in_ids, out_ids, dataset=args.dataset, source="imported",
             pre_col=pre, post_col=post, w_col=wcol)


# --------------------------------------------------------------------- 落盘


def finalize(args: argparse.Namespace, conn, info: dict[int, dict[str, str]],
             in_ids: np.ndarray, out_ids: np.ndarray, *, dataset: str, source: str,
             pre_col: str = "bodyId_pre", post_col: str = "bodyId_post", w_col: str = "weight") -> None:
    pre = _find(list(conn.columns), (pre_col,)) or _find(list(conn.columns), PRE_COLS)
    post = _find(list(conn.columns), (post_col,)) or _find(list(conn.columns), POST_COLS)
    wcol = _find(list(conn.columns), (w_col,)) or _find(list(conn.columns), WEIGHT_COLS)
    _require(pre is not None and post is not None, "连接表缺少 pre/post 列")

    if wcol:
        # neuPrint 的邻接表按 ROI 拆分。必须先按 (pre, post) 汇总，否则后面的
        # 去重会把多 ROI 的连接压成只保留其中一条，权重被系统性低估。
        conn = conn.groupby([pre, post], as_index=False)[wcol].sum()

    src_raw = conn[pre].astype(np.int64).to_numpy()
    dst_raw = conn[post].astype(np.int64).to_numpy()
    w_raw = (conn[wcol].astype(np.float32).to_numpy() if wcol
             else np.ones(src_raw.size, dtype=np.float32))

    node_ids = np.unique(np.concatenate([src_raw, dst_raw, in_ids, out_ids]))
    remap = {int(b): i for i, b in enumerate(node_ids)}
    src = np.array([remap[int(b)] for b in src_raw], dtype=np.int32)
    dst = np.array([remap[int(b)] for b in dst_raw], dtype=np.int32)
    in_new = np.array([remap[int(b)] for b in in_ids if int(b) in remap], dtype=np.int32)
    out_new = np.array([remap[int(b)] for b in out_ids if int(b) in remap], dtype=np.int32)
    _require(in_new.size > 0 and out_new.size > 0, "选定的上行/下降组在连接表里没有对应节点")

    if args.min_synapses > 0:
        k = w_raw >= args.min_synapses
        src, dst, w_raw = src[k], dst[k], w_raw[k]
    _, uniq = np.unique(src.astype(np.int64) * node_ids.size + dst, return_index=True)
    uniq.sort()
    src, dst, w_raw = src[uniq], dst[uniq], w_raw[uniq]
    _require(src.size > 0, "过滤后没有剩余连接边")

    if src.max() >= node_ids.size or dst.max() >= node_ids.size:
        raise SystemExit("[抓取失败] 节点索引越界，连接表里出现了自环或异常 bodyId")

    # 突触数 → 权重幅值：log1p 压缩重尾，再归一到均值 1，与合成图同一量纲。
    mag = np.log1p(w_raw.astype(np.float64))
    mag = mag / max(1e-9, mag.mean())
    weight = mag.astype(np.float32)

    n = int(node_ids.size)
    nts = np.array([info.get(int(b), {}).get("nt", "") for b in node_ids])
    # 模块标签用数据自带的 class（真实标注），重连会把它打散——
    # 这正是"真实结构 vs 重连结构"对照要检验的差异之一。缺失时留空并如实标注。
    cls_names = [info.get(int(b), {}).get("cls", "") for b in node_ids]
    uniq_cls = {c for c in cls_names if c}
    if len(uniq_cls) >= 2:
        cls_code = {c: i for i, c in enumerate(sorted(uniq_cls))}
        module = np.array([cls_code.get(c, len(cls_code)) for c in cls_names], dtype=np.int32)
        print(f"[模块] 用 class 标注得到 {len(uniq_cls)} 个模块（未标注节点归入第 {len(cls_code)} 组）")
    else:
        module = None
    inhibitory = np.array([any(nt.startswith(t) for t in INHIBITORY_NT) for nt in nts], dtype=bool)
    supplemented = 0
    if args.inhibitory_fraction is not None:
        # 递质预测缺失时，按指定比例随机补注抑制性标签——并如实标注这是补的。
        rng = np.random.default_rng(args.seed)
        target = int(round(args.inhibitory_fraction * n))
        have = int(inhibitory.sum())
        if have < target:
            pool = np.flatnonzero(~inhibitory & ~np.isin(np.arange(n), np.concatenate([in_new, out_new])))
            pick = rng.choice(pool, size=min(target - have, pool.size), replace=False)
            inhibitory[pick] = True
            supplemented = int(pick.size)
    sign = np.where(inhibitory[src], -1.0, 1.0).astype(np.float32)

    keep = keep_io_reachable(src, dst, weight, inhibitory, in_new, out_new)
    src, dst, weight, sign = src[keep], dst[keep], weight[keep], sign[keep]
    _require(io_reachable(_csr_local(src, dst, sign, weight, n), in_new, out_new),
             "过滤后输入→输出仍不可达；请放宽选组范围，不要降低可达性要求")

    notes = (
        f"来源：{dataset}（{source}）。下行组 {out_new.size} 个、上行组 {in_new.size} 个，"
        f"共 {n} 个节点、{src.size} 条聚合连接边（weight 为 log1p 压缩后的聚合突触数，"
        f"min_synapses={args.min_synapses}）。"
        "神经元动力学、递质符号、权重量纲均为人为建模选择，不是实测生理参数。"
    )
    notes += (f"抑制性标签来自数据自带的递质预测（GABA 判定抑制），"
              f"实测抑制比例 {inhibitory.mean():.1%}；"
              if supplemented == 0 else
              f"抑制性标签来自递质预测，另按 {args.inhibitory_fraction:.0%} 目标"
              f"补注了 {supplemented} 个（已标注为补充，非数据自带）；")
    if module is not None:
        notes += f"模块标签取自数据自带的 class 标注（{len(uniq_cls)} 类）。"
    else:
        notes += "数据未提供可用的模块标签，modularity 记为 0（非真实模块度）。"
    if not args.include_optic_lobe:
        notes += "未包含视叶与复眼感光通路。"

    meta = ConnectomeMeta(
        name=args.name or f"{dataset.replace(':', '-')}-gateway",
        source=source,  # type: ignore[arg-type]
        license=args.license, license_url=args.license_url,
        neuron_count=n, edge_count=int(src.size),
        synapse_count=int(w_raw.sum()) if w_raw.size else None,
        includes_optic_lobe=args.include_optic_lobe, io_reachable=True,
        input_count=int(in_new.size), output_count=int(out_new.size),
        checksum=graph_checksum(src, dst, weight), created_at=now_iso(),
        notes=notes, **_stats(src, dst, inhibitory, module, n),
    )
    graph = NeuronGraph(n, src, dst, weight, sign, inhibitory, in_new, out_new, meta, module)
    out = Path(args.out)
    graph.save(out)
    print(f"[完成] 已写入 {out}")
    print(jdump(meta.model_dump(), indent=2))


def _csr_local(src, dst, sign, weight, n):
    from scipy import sparse
    return sparse.csr_matrix((weight * sign, (src, dst)), shape=(n, n))


# --------------------------------------------------------------------- probe


def probe(args: argparse.Namespace) -> None:
    """只报告能不能连上、有哪些 type，不写任何文件。"""
    token = os.environ.get(args.token_env, "")
    _require(bool(token), f"未设置 {args.token_env}；probe 也需要 token。")
    try:
        from neuprint import Client, fetch_neurons
        from neuprint.queries import SegmentCriteria
    except ImportError as exc:
        raise SystemExit("需要 neuprint-python：pip install -r requirements-connectome.txt") from exc

    client = Client(args.server, dataset=args.dataset, token=token)
    print(f"[探测] 服务器 {args.server} / 数据集 {args.dataset}")
    print(f"[探测] 数据库状态：{client.fetch_dbmeta()}")
    for label, pattern in (("下降神经元", args.descending_regex), ("上行/感觉神经元", args.ascending_regex)):
        df, _ = fetch_neurons(SegmentCriteria(type=pattern, regex=True), client=client)
        print(f"[探测] {label} (regex={pattern!r}) 命中 {len(df)} 个")
        if len(df):
            print("        type 样例：", sorted(df["type"].dropna().unique().tolist())[:15])
            print("        列：", list(df.columns)[:20])


# --------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description="真实果蝇连接组接入工具（在线官方 API / 本地文件双通道）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", default="data/male-cns-gateway.npz")
        sp.add_argument("--name", default=None)
        # neuPrint 的 type 正则是**全串匹配**：`^DN` 匹配不到 `DN1a`，必须写 `^DN.*`。
        sp.add_argument("--descending-regex", default=r"^DN.*",
                        help="下降/输出组。male-cns:v1.0 用 '^DN.*'（484 类）")
        sp.add_argument("--ascending-regex", default=r"^AN.*",
                        help="上行/感觉组。male-cns:v1.0 用 '^AN.*'（556 类）")
        sp.add_argument("--max-input", type=int, default=96)
        sp.add_argument("--max-output", type=int, default=48)
        sp.add_argument("--min-synapses", type=int, default=3)
        sp.add_argument("--inhibitory-fraction", type=float, default=0.12)
        sp.add_argument("--treat-glu-inhibitory", action="store_true")
        sp.add_argument("--include-optic-lobe", action="store_true")
        sp.add_argument("--license", default="CC-BY-4.0")
        sp.add_argument("--license-url", default="https://creativecommons.org/licenses/by/4.0/")
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("online", help="经官方 neuPrint API 抓取")
    common(sp)
    sp.add_argument("--server", default="https://neuprint.janelia.org")
    sp.add_argument("--dataset", default="male-cns:v1.0")
    sp.add_argument("--token-env", default="NEUPRINT_TOKEN")
    sp.add_argument("--intermediate-limit", type=int, default=400,
                    help="一跳中间神经元上限（位于 上行→中间→下降 路径上）")
    sp.add_argument("--expand-hops", type=int, default=1)
    sp.add_argument("--expand-batch", type=int, default=300)
    sp.add_argument("--node-budget", type=int, default=1200)
    sp.add_argument("--edge-batch", type=int, default=120,
                    help="取边时每批多少个预神经元。节点数上千后必须调大，否则查询次数过多")
    sp.set_defaults(func=online)

    sp = sub.add_parser("offline", help="读本地导出的连接表与注释表")
    common(sp)
    sp.add_argument("--connections", required=True, help="CSV/Parquet，支持逗号分隔多个或配合 --glob")
    sp.add_argument("--neurons", default=None)
    sp.add_argument("--glob", action="store_true", help="把 --connections 当通配符展开")
    sp.add_argument("--dataset", default="local-export")
    sp.set_defaults(func=offline)

    sp = sub.add_parser("probe", help="只探测可用性与 type 名称")
    sp.add_argument("--server", default="https://neuprint.janelia.org")
    sp.add_argument("--dataset", default="male-cns:v1.0")
    sp.add_argument("--token-env", default="NEUPRINT_TOKEN")
    sp.add_argument("--descending-regex", default=r"^DN.*")
    sp.add_argument("--ascending-regex", default=r"^AN.*")
    sp.set_defaults(func=probe)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
