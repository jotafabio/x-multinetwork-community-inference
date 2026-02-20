
#
# Author: Jose Fabio Ribeiro Bezerra
# Date: 20-Feb-2026
# Based on: "Identifying Social Networks on X: A Multi-Network Approach for Inference of Community Structures"
#          (draft, to be published)
#

"""
Replicates the paper's multi-network methodology from the *layer-edge list* stage onward:

- Build connectivity union graph G_∪ across layers (treated as undirected)
- DFS connected components on G_∪
- Per-component, per-layer max-normalization after symmetrizing weights
- Fuse layers by weighted summation (default αℓ=1)
- Community detection per component:
    * Preferred: Leiden (leidenalg + igraph), RBConfigurationVertexPartition, n_iterations=20, resolution=1.0
    * Fallback (not Leiden): NetworkX Louvain

Input CSV columns (quoted CSV):
  subgraph (optional int), source_node (str), target_node (str), weight (float), interaction (str)

This script assumes the layer-edge list already exists; it does not parse tweet JSON nor run LDA.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import networkx as nx

Layer = str  # {"p","q","u","t","m","h","r"}

LOGGER = logging.getLogger("infer_multinetwork")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def stage(msg: str) -> None:
    # Stage banner for visual traceability
    LOGGER.info("=== %s ===", msg)


def iter_with_progress(items: List[Tuple[int, Set[str]]], enabled: bool):
    """
    Visual progress over components. Uses tqdm if available, otherwise prints per-component INFO.
    """
    if not enabled:
        for it in items:
            yield it
        return

    try:
        from tqdm import tqdm  # type: ignore

        for it in tqdm(items, desc="Processing components", unit="comp"):
            yield it
    except Exception:
        total = len(items)
        for i, it in enumerate(items, start=1):
            cid, comp = it
            LOGGER.info("Component %d/%d (id=%d, nodes=%d)", i, total, cid, len(comp))
            yield it


def count_lines_fast(path: str) -> int:
    """
    Counts newline characters quickly to estimate the number of lines in a text file.
    Returns the number of lines (>=1 for non-empty files).
    """
    buf_size = 1024 * 1024  # 1MB
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            n += b.count(b"\n")
    return n


def add_layer_column_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df["layer"] based on df["interaction"] using vectorized string operations.
    """
    s = df["interaction"].astype("string").str.strip().str.lower()
    layer = pd.Series(pd.NA, index=df.index, dtype="string")

    m_profile = s.eq("profile similar to")
    m_quotes = s.eq("quotes")
    m_url = s.str.startswith("url", na=False)
    m_retweets = s.eq("retweets")
    m_mentions = s.eq("mentions")
    m_hashtag = s.str.startswith("hashtag", na=False)
    m_replies = s.eq("replies")

    layer.loc[m_profile] = "p"
    layer.loc[m_quotes] = "q"
    layer.loc[m_url] = "u"
    layer.loc[m_retweets] = "t"
    layer.loc[m_mentions] = "m"
    layer.loc[m_hashtag] = "h"
    layer.loc[m_replies] = "r"

    unknown_mask = layer.isna()
    if bool(unknown_mask.any()):
        bad = s.loc[unknown_mask].dropna().unique().tolist()
        sample = bad[:5]
        raise ValueError(f"Unrecognized interaction values (sample up to 5): {sample}")

    df["layer"] = layer.astype("string")
    return df


def read_edges_csv(path: str, *, progress: bool, chunksize: int) -> pd.DataFrame:
    """
    Reads the quoted CSV.
    If progress=True and chunksize>0, reads in chunks and shows progress during read.
    """
    t0 = time.time()
    stage(f"Reading input CSV: {path}")

    required = ("source_node", "target_node", "weight", "interaction")

    if not progress or chunksize <= 0:
        df = pd.read_csv(
            path,
            quotechar='"',
            escapechar="\\",
            skipinitialspace=True,
            dtype={"source_node": "string", "target_node": "string", "interaction": "string"},
            low_memory=False,
        )
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col!r}")

        df["source_node"] = df["source_node"].astype("string").str.strip()
        df["target_node"] = df["target_node"].astype("string").str.strip()
        df["interaction"] = df["interaction"].astype("string").str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="raise").astype(float)
        df = add_layer_column_vectorized(df)

        LOGGER.info("Loaded rows=%d in %.2fs", len(df), time.time() - t0)
        return df

    # Progress-enabled path (chunked)
    stage("Counting lines for CSV progress (fast scan)")
    line_count = count_lines_fast(path)
    # line_count includes header line, so data rows are at most line_count-1 (can be off if embedded newlines)
    total_rows_est = max(0, line_count - 1)
    LOGGER.info("Estimated data rows=%d (may be off if fields contain embedded newlines)", total_rows_est)

    stage(f"Reading CSV in chunks (chunksize={chunksize})")
    frames: List[pd.DataFrame] = []

    try:
        from tqdm import tqdm  # type: ignore

        pbar = tqdm(total=total_rows_est if total_rows_est > 0 else None, desc="Reading CSV", unit="rows")
        use_tqdm = True
    except Exception:
        pbar = None
        use_tqdm = False
        LOGGER.info("tqdm not available; will log progress per chunk.")

    try:
        reader = pd.read_csv(
            path,
            quotechar='"',
            escapechar="\\",
            skipinitialspace=True,
            dtype={"source_node": "string", "target_node": "string", "interaction": "string"},
            low_memory=False,
            chunksize=chunksize,
        )

        rows_seen = 0
        for chunk_idx, chunk in enumerate(reader, start=1):
            for col in required:
                if col not in chunk.columns:
                    raise ValueError(f"Missing required column: {col!r}")

            chunk["source_node"] = chunk["source_node"].astype("string").str.strip()
            chunk["target_node"] = chunk["target_node"].astype("string").str.strip()
            chunk["interaction"] = chunk["interaction"].astype("string").str.strip()
            chunk["weight"] = pd.to_numeric(chunk["weight"], errors="raise").astype(float)
            chunk = add_layer_column_vectorized(chunk)

            frames.append(chunk)
            rows_seen += len(chunk)

            if use_tqdm and pbar is not None:
                pbar.update(len(chunk))
            else:
                LOGGER.info("Read chunk=%d rows=%d total_rows=%d", chunk_idx, len(chunk), rows_seen)

    finally:
        if use_tqdm and pbar is not None:
            pbar.close()

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(required) + ["layer"])
    LOGGER.info("Loaded rows=%d in %.2fs", len(df), time.time() - t0)
    return df


def _unordered(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def dfs_components(nodes: Iterable[str], undirected_edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    """
    DFS connected components on an undirected graph described by (nodes, undirected_edges).
    """
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for u, v in undirected_edges:
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited: Set[str] = set()
    comps: List[Set[str]] = []
    for start in adj.keys():
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: Set[str] = set()
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def build_component_layer_weights(
    df: pd.DataFrame, comp_of_node: Dict[str, int]
) -> Dict[int, Dict[Layer, Dict[Tuple[str, str], float]]]:
    """
    Returns:
      comp_id -> layer -> dict( unordered_pair -> weight )

    Storage is unordered pairs for all layers to match the paper's undirected treatment.
    - URL layer 'u' is forced to binary per pair (1 if any evidence exists).
    - Other layers accumulate weights per pair within a component.
    """
    out: Dict[int, Dict[Layer, Dict[Tuple[str, str], float]]] = {}
    stage("Building per-component, per-layer edge weights")

    for r in df.itertuples(index=False):
        u = str(r.source_node)
        v = str(r.target_node)
        if not u or not v:
            continue

        c = comp_of_node[u]
        if comp_of_node[v] != c:
            # Should not happen if comp_of_node comes from union graph;
            # keep safe behavior.
            continue

        layer: Layer = str(r.layer)
        w = float(r.weight)
        key = _unordered(u, v)

        out.setdefault(c, {}).setdefault(layer, {})
        if layer == "u":
            out[c][layer][key] = 1.0
        else:
            out[c][layer][key] = out[c][layer].get(key, 0.0) + w

    return out


def normalize_layers_in_component(
    comp_nodes: Set[str],
    layer_weights: Dict[Layer, Dict[Tuple[str, str], float]],
) -> Dict[Layer, Dict[Tuple[str, str], float]]:
    """
    Component-wise per-layer max-normalization:

      \bar W(x,y) = (W(x,y) + W(y,x))/2
      μ = max_{x,y∈C_i} \bar W(x,y)
      \hat W(x,y) = \bar W(x,y) / μ     (or 0 if μ=0)

    Because we store unordered pairs, the symmetrization reduces to identity for stored keys.
    """
    norm: Dict[Layer, Dict[Tuple[str, str], float]] = {}
    for layer in ("p", "q", "u", "t", "m", "h", "r"):
        wmap = layer_weights.get(layer, {})
        if not wmap:
            norm[layer] = {}
            continue

        sym = {k: v for k, v in wmap.items() if k[0] in comp_nodes and k[1] in comp_nodes}
        mu = max(sym.values(), default=0.0)
        if mu <= 0.0:
            norm[layer] = {k: 0.0 for k in sym.keys()}
        else:
            norm[layer] = {k: (v / mu) for k, v in sym.items()}
    return norm


def fuse_layers(norm_layers: Dict[Layer, Dict[Tuple[str, str], float]], alphas: Dict[Layer, float]) -> Dict[Tuple[str, str], float]:
    """
    Fused weight:
      W_fused(x,y) = Σ_ℓ α_ℓ * \hat W_ℓ(x,y)
    """
    fused: Dict[Tuple[str, str], float] = {}
    keys: Set[Tuple[str, str]] = set()
    for wmap in norm_layers.values():
        keys |= set(wmap.keys())

    for key in keys:
        w = 0.0
        for layer in ("p", "q", "u", "t", "m", "h", "r"):
            w += float(alphas.get(layer, 1.0)) * float(norm_layers.get(layer, {}).get(key, 0.0))
        if w > 0.0:
            fused[key] = w
    return fused


def _try_leiden(
    nodes: List[str],
    fused: Dict[Tuple[str, str], float],
    resolution: float,
    n_iterations: int,
    seed: Optional[int],
) -> Optional[List[int]]:
    """
    Returns membership list aligned with 'nodes' if Leiden is available, else None.
    """
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except Exception:
        LOGGER.info("Leiden unavailable (missing igraph/leidenalg). Falling back to NetworkX Louvain.")
        return None

    idx = {n: i for i, n in enumerate(nodes)}
    items = [(pair, w) for pair, w in fused.items() if pair[0] != pair[1] and w > 0.0]
    items.sort(key=lambda x: (idx.get(x[0][0], -1), idx.get(x[0][1], -1)))

    edges = [(idx[u], idx[v]) for (u, v), _w in items]
    weights = [float(_w) for _pair, _w in items]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    return list(part.membership)


def detect_communities(
    nodes: Set[str],
    fused: Dict[Tuple[str, str], float],
    *,
    resolution: float,
    n_iterations: int,
    seed: Optional[int],
) -> Dict[str, int]:
    """
    Detect communities in a component using Leiden if available, otherwise Louvain.
    """
    node_list = sorted(nodes)

    # Handle edgeless component explicitly to avoid undefined community behavior across backends.
    if not fused:
        return {n: i for i, n in enumerate(node_list)}  # singleton communities

    membership = _try_leiden(node_list, fused, resolution, n_iterations, seed)
    if membership is not None:
        LOGGER.debug("Community detection: Leiden (nodes=%d, edges=%d)", len(node_list), len(fused))
        return {node_list[i]: int(membership[i]) for i in range(len(node_list))}

    # Fallback (NOT Leiden): NetworkX Louvain
    LOGGER.debug("Community detection: Louvain fallback (nodes=%d, edges=%d)", len(node_list), len(fused))
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for (u, v), w in fused.items():
        if u != v and w > 0.0:
            G.add_edge(u, v, weight=float(w))

    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    out: Dict[str, int] = {}
    for cid, comm in enumerate(comms):
        for n in comm:
            out[n] = cid
    return out


def run(
    df: pd.DataFrame,
    *,
    use_input_subgraph: bool,
    alphas: Dict[Layer, float],
    resolution: float,
    n_iterations: int,
    seed: Optional[int],
    progress: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    stage("Preparing nodes/components")

    nodes = set(df["source_node"].dropna().astype(str)) | set(df["target_node"].dropna().astype(str))
    LOGGER.info("Unique nodes=%d", len(nodes))

    if use_input_subgraph:
        if "subgraph" not in df.columns:
            raise ValueError("Missing 'subgraph' column with --use-input-subgraph.")
        comp_of_node: Dict[str, int] = {}
        for r in df[["subgraph", "source_node", "target_node"]].itertuples(index=False):
            c = int(r.subgraph)
            for n in (str(r.source_node), str(r.target_node)):
                if n in comp_of_node and comp_of_node[n] != c:
                    raise ValueError(f"Node {n!r} appears in multiple subgraphs ({comp_of_node[n]} vs {c}).")
                comp_of_node[n] = c
        comps: Dict[int, Set[str]] = {}
        for n, c in comp_of_node.items():
            comps.setdefault(c, set()).add(n)
    else:
        stage("Computing DFS connected components on union graph")
        comps_list = dfs_components(
            nodes,
            list(zip(df["source_node"].astype(str), df["target_node"].astype(str))),
        )
        comps_list = sorted(comps_list, key=lambda s: (len(s), sorted(s)[0] if s else ""))
        comp_of_node = {n: cid for cid, comp in enumerate(comps_list) for n in comp}
        comps = {cid: comp for cid, comp in enumerate(comps_list)}

    LOGGER.info("Components=%d", len(comps))

    comp_layer = build_component_layer_weights(df, comp_of_node)

    comm_rows: List[Tuple[int, str, int]] = []
    fused_rows: List[Tuple[int, str, str, float]] = []

    stage("Normalizing, fusing, and detecting communities per component")
    comp_items = sorted(comps.items())
    for cid, comp_nodes in iter_with_progress(comp_items, enabled=progress):
        LOGGER.debug("Component id=%d nodes=%d", cid, len(comp_nodes))
        layers = comp_layer.get(cid, {})
        norm = normalize_layers_in_component(comp_nodes, layers)
        fused = fuse_layers(norm, alphas)
        membership = detect_communities(comp_nodes, fused, resolution=resolution, n_iterations=n_iterations, seed=seed)

        for node, comm_id in membership.items():
            comm_rows.append((cid, node, comm_id))
        for (u, v), w in fused.items():
            fused_rows.append((cid, u, v, float(w)))

    LOGGER.info("Done in %.2fs", time.time() - t0)

    return (
        pd.DataFrame(comm_rows, columns=["component_id", "node", "community_id"]),
        pd.DataFrame(fused_rows, columns=["component_id", "source_node", "target_node", "fused_weight"]),
    )


def build_fused_graph(fused_df: pd.DataFrame, all_nodes: Set[str]) -> nx.Graph:
    """
    Builds the fused graph as an undirected weighted NetworkX graph.
    Ensures all nodes appear, including isolates.
    """
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for r in fused_df.itertuples(index=False):
        u = str(r.source_node)
        v = str(r.target_node)
        w = float(r.fused_weight)
        if u == v or w <= 0.0:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] = float(G[u][v].get("weight", 0.0)) + w
        else:
            G.add_edge(u, v, weight=w)
    return G


def partition_from_comm_df(comm_df: pd.DataFrame, all_nodes: Set[str]) -> List[Set[str]]:
    """
    Returns a list of node-sets, one per community. Ensures every node appears exactly once
    (adds singletons if needed).
    """
    comm_map: Dict[Tuple[int, int], Set[str]] = {}
    seen: Set[str] = set()
    for r in comm_df.itertuples(index=False):
        key = (int(r.component_id), int(r.community_id))
        n = str(r.node)
        comm_map.setdefault(key, set()).add(n)
        seen.add(n)

    missing = set(all_nodes) - seen
    if missing:
        for n in missing:
            comm_map[(-1, len(comm_map))] = {n}

    return list(comm_map.values())


def modularity_q(G: nx.Graph, communities: List[Set[str]]) -> float:
    """
    Weighted Newman-Girvan modularity (NetworkX implementation).
    For graphs with 0 edges, returns 0.0.
    """
    if G.number_of_edges() == 0:
        return 0.0
    try:
        return float(nx.algorithms.community.quality.modularity(G, communities, weight="weight"))
    except Exception:
        return 0.0


def _weighted_degree(G: nx.Graph) -> Dict[str, float]:
    return {str(n): float(d) for n, d in G.degree(weight="weight")}


def conductance_stats(G: nx.Graph, communities: List[Set[str]]) -> Dict[str, float]:
    """
    Computes weighted conductance φ for each community *within its connected component*.
      φ(S) = cut(S, C\\S) / min(vol(S), vol(C\\S))
    where vol(.) is sum of weighted degrees within the component.

    Returns summary stats across communities:
      - median, q1, q3, iqr, min, max
      - volume-weighted mean (weights = vol(S))
      - counts: n_eval (included), n_skipped (whole-component or zero-denominator)
    """
    if G.number_of_nodes() == 0:
        return {
            "n_eval": 0,
            "n_skipped": 0,
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "vw_mean": float("nan"),
        }

    deg_w = _weighted_degree(G)

    comp_list = list(nx.connected_components(G))
    comp_id_of: Dict[str, int] = {}
    for i, comp_nodes in enumerate(comp_list):
        for n in comp_nodes:
            comp_id_of[str(n)] = i

    comp_vol: Dict[int, float] = {}
    for i, comp_nodes in enumerate(comp_list):
        comp_vol[i] = float(sum(deg_w.get(str(n), 0.0) for n in comp_nodes))

    phi_vals: List[float] = []
    volS_vals: List[float] = []
    skipped = 0

    for S in communities:
        if not S:
            skipped += 1
            continue

        any_node = next(iter(S))
        cid = comp_id_of.get(str(any_node), None)
        if cid is None:
            skipped += 1
            continue

        C = set(str(n) for n in comp_list[cid])
        S_in = set(str(n) for n in S if str(n) in C)
        if not S_in or S_in == C:
            skipped += 1
            continue

        volC = comp_vol[cid]
        volS = float(sum(deg_w.get(n, 0.0) for n in S_in))
        volT = float(volC - volS)
        denom = min(volS, volT)
        if denom <= 0.0:
            skipped += 1
            continue

        cut = 0.0
        T_set = C - S_in
        for u in S_in:
            for v, attr in G[u].items():
                vs = str(v)
                if vs in T_set:
                    cut += float(attr.get("weight", 1.0))

        phi = float(cut / denom)
        phi_vals.append(phi)
        volS_vals.append(volS)

    if not phi_vals:
        return {
            "n_eval": 0,
            "n_skipped": skipped,
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "vw_mean": float("nan"),
        }

    s = pd.Series(phi_vals, dtype="float64")
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    vw_mean = float(sum(p * w for p, w in zip(phi_vals, volS_vals)) / max(1e-12, sum(volS_vals)))

    return {
        "n_eval": int(len(phi_vals)),
        "n_skipped": int(skipped),
        "median": float(s.median()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "min": float(s.min()),
        "max": float(s.max()),
        "vw_mean": vw_mean,
    }


def basic_graph_stats(G: nx.Graph, communities: List[Set[str]]) -> Dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = float(nx.density(G)) if n > 1 else 0.0
    n_components = int(nx.number_connected_components(G)) if n > 0 else 0

    degs = [int(d) for _n, d in G.degree()]
    if degs:
        deg_min = int(min(degs))
        deg_med = float(statistics.median(degs))
        deg_max = int(max(degs))
    else:
        deg_min = deg_med = deg_max = 0

    sizes = [len(c) for c in communities] if communities else []
    if sizes:
        s = pd.Series(sizes, dtype="int64")
        comm_count = int(len(sizes))
        comm_min = int(s.min())
        comm_med = float(s.median())
        comm_max = int(s.max())
        comm_q1 = float(s.quantile(0.25))
        comm_q3 = float(s.quantile(0.75))
    else:
        comm_count = 0
        comm_min = comm_med = comm_max = 0
        comm_q1 = comm_q3 = float("nan")

    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": density,
        "n_components": n_components,
        "deg_min": int(deg_min),
        "deg_median": float(deg_med),
        "deg_max": int(deg_max),
        "community_count": int(comm_count),
        "community_size_min": int(comm_min),
        "community_size_median": float(comm_med),
        "community_size_max": int(comm_max),
        "community_size_q1": float(comm_q1),
        "community_size_q3": float(comm_q3),
    }


def print_stats_block(title: str, G: nx.Graph, communities: List[Set[str]]) -> None:
    bs = basic_graph_stats(G, communities)
    cq = modularity_q(G, communities)
    cs = conductance_stats(G, communities)

    LOGGER.info("=== %s ===", title)
    LOGGER.info("Graph statistics (fused graph):")
    LOGGER.info(
        "  nodes=%d  edges=%d  density=%.6g  components=%d",
        int(bs["n_nodes"]),
        int(bs["n_edges"]),
        float(bs["density"]),
        int(bs["n_components"]),
    )
    LOGGER.info(
        "  degree(min/median/max)=%d/%.6g/%d",
        int(bs["deg_min"]),
        float(bs["deg_median"]),
        int(bs["deg_max"]),
    )
    LOGGER.info(
        "  communities: count=%d  size(min/median/max)=%d/%.6g/%d  size(Q1/Q3)=%.6g/%.6g",
        int(bs["community_count"]),
        int(bs["community_size_min"]),
        float(bs["community_size_median"]),
        int(bs["community_size_max"]),
        float(bs["community_size_q1"]),
        float(bs["community_size_q3"]),
    )
    LOGGER.info("  modularity Q (weighted)=%.6g", float(cq))
    LOGGER.info(
        "  conductance φ (weighted; within connected component): n_eval=%d n_skipped=%d  "
        "median=%.6g  IQR=%.6g (Q1=%.6g, Q3=%.6g)  min=%.6g max=%.6g  vw_mean(volS)=%.6g",
        int(cs["n_eval"]),
        int(cs["n_skipped"]),
        float(cs["median"]),
        float(cs["iqr"]),
        float(cs["q1"]),
        float(cs["q3"]),
        float(cs["min"]),
        float(cs["max"]),
        float(cs["vw_mean"]),
    )


def evaluate_configuration(
    name: str,
    df: pd.DataFrame,
    all_nodes: Set[str],
    *,
    use_input_subgraph: bool,
    alphas: Dict[Layer, float],
    resolution: float,
    n_iterations: int,
    seed: Optional[int],
    progress: bool,
) -> Dict[str, float]:
    comm_df, fused_df = run(
        df,
        use_input_subgraph=use_input_subgraph,
        alphas=alphas,
        resolution=resolution,
        n_iterations=n_iterations,
        seed=seed,
        progress=progress,
    )
    G = build_fused_graph(fused_df, all_nodes)
    comms = partition_from_comm_df(comm_df, all_nodes)
    bs = basic_graph_stats(G, comms)
    q = modularity_q(G, comms)
    cs = conductance_stats(G, comms)

    row: Dict[str, float] = {"name": name}
    row.update(bs)
    row["modularity_q"] = q
    row["cond_n_eval"] = cs["n_eval"]
    row["cond_n_skipped"] = cs["n_skipped"]
    row["cond_median"] = cs["median"]
    row["cond_q1"] = cs["q1"]
    row["cond_q3"] = cs["q3"]
    row["cond_iqr"] = cs["iqr"]
    row["cond_min"] = cs["min"]
    row["cond_max"] = cs["max"]
    row["cond_vw_mean_volS"] = cs["vw_mean"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-communities", default=None)
    ap.add_argument("--output-fused-edges", default=None)
    ap.add_argument("--use-input-subgraph", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=20)

    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--progress", dest="progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")

    ap.add_argument("--csv-chunksize", type=int, default=250_000)

    ap.add_argument("--report-stats", action="store_true", help="Print descriptive stats for the fused graph of the main run.")
    ap.add_argument("--run-ablation", action="store_true", help="Run minimal ablation (Q + conductance summaries) and exit.")
    ap.add_argument("--ablation-out", default=None, help="Optional CSV output path for ablation summary table.")

    for layer in ("p", "q", "u", "t", "m", "h", "r"):
        ap.add_argument(f"--alpha-{layer}", type=float, default=1.0)
    args = ap.parse_args()

    setup_logging(args.log_level)

    df = read_edges_csv(args.input, progress=args.progress, chunksize=args.csv_chunksize)
    alphas = {layer: float(getattr(args, f"alpha_{layer}")) for layer in ("p", "q", "u", "t", "m", "h", "r")}
    all_nodes = set(df["source_node"].dropna().astype(str)) | set(df["target_node"].dropna().astype(str))

    if args.run_ablation:
        stage("Running ablation")
        base = dict(alphas)

        configs: List[Tuple[str, Dict[Layer, float]]] = []
        configs.append(("all_layers", dict(base)))

        c = dict(base)
        c["h"] = 0.0
        configs.append(("no_hashtags", c))

        c = dict(base)
        c["u"] = 0.0
        configs.append(("no_urls", c))

        c = dict(base)
        c["p"] = 0.0
        configs.append(("interactions_only", c))

        c = {k: 0.0 for k in base.keys()}
        c["p"] = base.get("p", 1.0)
        configs.append(("profile_only", c))

        rows: List[Dict[str, float]] = []
        for name, a in configs:
            stage(f"Ablation config: {name}")
            row = evaluate_configuration(
                name,
                df,
                all_nodes,
                use_input_subgraph=args.use_input_subgraph,
                alphas=a,
                resolution=args.resolution,
                n_iterations=args.iterations,
                seed=args.seed,
                progress=args.progress,
            )
            rows.append(row)
            LOGGER.info(
                "Ablation[%s]: Q=%.6g, cond(median)=%.6g, nodes=%d, edges=%d, comm=%d",
                name,
                row["modularity_q"],
                row["cond_median"],
                int(row["n_nodes"]),
                int(row["n_edges"]),
                int(row["community_count"]),
            )

        out_df = pd.DataFrame(rows)
        stage("Ablation summary table (printed)")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            LOGGER.info("\n%s", out_df.to_string(index=False))

        if args.ablation_out:
            out_df.to_csv(args.ablation_out, index=False, quoting=csv.QUOTE_ALL)
            stage(f"Wrote ablation CSV: {args.ablation_out}")
        return

    if not args.output_communities:
        raise SystemExit("Error: --output-communities is required unless --run-ablation is set.")

    comm_df, fused_df = run(
        df,
        use_input_subgraph=args.use_input_subgraph,
        alphas=alphas,
        resolution=args.resolution,
        n_iterations=args.iterations,
        seed=args.seed,
        progress=args.progress,
    )

    if args.report_stats:
        G = build_fused_graph(fused_df, all_nodes)
        comms = partition_from_comm_df(comm_df, all_nodes)
        print_stats_block("Descriptive statistics (main run)", G, comms)

    stage(
        f"Writing outputs: {args.output_communities}"
        + (f" and {args.output_fused_edges}" if args.output_fused_edges else "")
    )
    comm_df.to_csv(args.output_communities, index=False, quoting=csv.QUOTE_ALL)
    if args.output_fused_edges:
        fused_df.to_csv(args.output_fused_edges, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()