# X Multi-Network Community Inference

Python implementation of the **multi-network (multi-layer) fusion + community detection** pipeline described in:

**“Identifying Social Networks on X: A Multi-Network Approach for Inference of Community Structures”**

This repository focuses on the *layer-edge list → fused graph → per-component community inference* stage. It assumes you have already produced an edge list with interaction labels and weights.

---

## What the script does

Given a CSV of edges across multiple interaction layers, the script:

1. **Loads the layer-edge list** from a quoted CSV (chunked read with progress)
2. Builds a **union graph** \(G_{\cup}\) over all layers (treated as undirected)
3. Computes **connected components** on \(G_{\cup}\) (DFS), unless you provide `subgraph` ids
4. For each component and each layer:
   - aggregates weights per unordered node pair
   - forces the **URL layer** to binary per pair
   - applies **max-normalization** (per component, per layer)
5. **Fuses** layers via \(W_\text{fused}(x,y)=\sum_{\ell}\alpha_\ell \hat{W}_\ell(x,y)\)
6. Runs **community detection per component**
   - **Leiden** if available (`python-igraph` + `leidenalg`)
   - otherwise falls back to **NetworkX Louvain**
7. Optionally prints **descriptive fused-graph statistics** (density, degree summary, modularity, conductance summaries)
8. Optionally runs a **minimal ablation** to recompute Q and conductance summaries for layer subsets

---

## Repository contents

- `infer_multinetwork.py` – main script
- `edges.csv` – your input (not included)
- `communities.csv` – node → community assignment output (generated)
- `fused_edges.csv` – fused graph edges output (generated)

---

## Input format

A **quoted CSV** with a header row and at least these columns:

- `source_node` *(string)*
- `target_node` *(string)*
- `weight` *(float)*
- `interaction` *(string)*

Optional column:

- `subgraph` *(int)* — required only if using `--use-input-subgraph`

### Accepted interaction values

The script maps these strings into layer codes:

- `mentions` → `m`
- `retweets` → `t`
- `quotes` → `q`
- `profile similar to` → `p`
- `url <...>` → `u` (e.g., `url https://example.com/page`)
- `hashtag <...>` → `h` (e.g., `hashtag climatechange`)
- `replies` → `r` *(supported)*

### Example (header + 2 rows)

```csv
"subgraph","source_node","target_node","weight","interaction"
"0","JoanneDCPara","mandamichel","0.83333333333333","mentions"
"0","JoanneDCPara","anotherUser","1.0","url https://example.com/some/article"
