import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
Operation = Tuple[str, ...]  

def build_split_graph(
        start_split: Dict[str, str],
        end_split: Dict[str, str],
        names: List[str],
):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from({start_split[name] for name in names} | {end_split[name] for name in names})
    for name in names:
        u, v = start_split[name], end_split[name]
        if u != v:
            graph.add_edge(u, v, name=name)
    return graph

def decompose(graph: nx.MultiDiGraph):
    adj: Dict[str, List[Tuple[str, str]]] = {n: [] for n in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        adj[u].append((v, data["name"]))

    cycles: List[Tuple[str, ...]] = []
    leftover: List[str] = []

    for start in list(graph.nodes()):
        while adj[start]:
            stack_nodes = [start]
            stack_names: List[str] = []
            pos = {start: 0}

            while True:
                node = stack_nodes[-1]
                if not adj[node]:
                    leftover.extend(stack_names)
                    break
                nxt, name = adj[node].pop()
                stack_names.append(name)

                if nxt in pos:
                    idx = pos[nxt]
                    cycles.append(tuple(stack_names[idx:]))
                    del stack_names[idx:]
                    for removed in stack_nodes[idx + 1:]:
                        del pos[removed]
                    del stack_nodes[idx + 1:]
                else:
                    stack_nodes.append(nxt)
                    pos[nxt] = len(stack_nodes) - 1

    return cycles, leftover

def build_similarity_dataset(datatype: str, data, similarity="ecfp", cache_path=None):
    import pickle
    from pathlib import Path as _Path

    if cache_path is not None and _Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from datasail.reader.read import read_data_type
    from datasail.cluster.clustering import cluster

    dataset = read_data_type(datatype)(
        data=data, weights=None, sim=similarity, dist=None,
        num_clusters=np.inf, detect_duplicates=False,
    )
    dataset.id_map = {n: n for n in dataset.names}   # must stay before cluster() -- see earlier tox21 fix
    dataset = cluster(dataset, threads=os.cpu_count() or 1, logdir=None, linkage="average", output=None)

    if cache_path is not None:
        cache_path = _Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(dataset, f)
        tmp_path.replace(cache_path)  # atomic rename, safe even if another job is mid-write elsewhere

    return dataset

def rank_operations(
        strata_cycles: Dict[Tuple, Tuple[List[Operation], List[str]]],
        start_split: Dict[str, str],
        end_split: Dict[str, str],
        names: List[str],
        metric: "np.ndarray",
        weights: Dict[str, float] = None,):

    id_to_pos = {name: i for i, name in enumerate(names)}
    folds = np.array([start_split[name] for name in names])
    end_folds = np.array([end_split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])

    def delta_for_positions(pos, new_folds_at_pos):
        row = metric[pos] * w[None, :]
        own_w = w[pos][:, None]
        new_term = ((new_folds_at_pos[:, None] != folds) * row * own_w).sum()
        old_term = ((folds[pos][:, None] != folds) * row * own_w).sum()
        return new_term - old_term

    results: List[Tuple[Operation, float]] = []
    for cycles, leftover in strata_cycles.values():
        for cycle in cycles:
            pos = np.array([id_to_pos[n] for n in cycle])
            new_folds_at_pos = np.roll(folds[pos], -1)
            results.append((cycle, delta_for_positions(pos, new_folds_at_pos)))
        for name in leftover:
            pos = np.array([id_to_pos[name]])
            new_folds_at_pos = np.array([end_folds[pos[0]]])
            results.append(((name,), delta_for_positions(pos, new_folds_at_pos)))

    return sorted(results, key=lambda cr: cr[1])

def rank_operations_random(
        strata_cycles: Dict[Tuple, Tuple[List[Operation], List[str]]],
        start_split: Dict[str, str] = None,
        end_split: Dict[str, str] = None,
        names: List[str] = None,
        metric: "np.ndarray" = None,
        weights: Dict[str, float] = None,
        seed: Optional[int] = None,
):
    # Random ranking discards the leakage-delta immediately (shuffle, not sort),
    # so skip computing it -- this used to redo the same O(n^2)-ish work as
    # rank_operations for a value nobody reads.
    results: List[Tuple[Operation, float]] = []
    for cycles, leftover in strata_cycles.values():
        for cycle in cycles:
            results.append((cycle, 0.0))
        for name in leftover:
            results.append(((name,), 0.0))

    rng = np.random.default_rng(seed)
    rng.shuffle(results)
    return results

def apply_cycle(split: Dict[str, str], operation, end_split: Dict[str, str]) :
 
    new_split = dict(split)
    n = len(operation)
    if n == 1:
        name = operation[0]
        new_split[name] = end_split[name]
        return new_split
    for i, name in enumerate(operation):
        next_name = operation[(i + 1) % n]
        new_split[name] = split[next_name]
    return new_split


def apply_k_cycles(
        start_split: Dict[str, str],
        ranked_operations: List[Tuple[Operation, float]],
        end_split: Dict[str, str],
        k: int,):
  
    split = dict(start_split)
    for operation, _ in ranked_operations[:k]:
        split = apply_cycle(split, operation, end_split)
    return split

def leakage(
        split: Dict[str, str],
        names: List[str],
        metric: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
):
    
    folds = np.array([split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])
    mask = np.zeros((len(names), len(names)))
    for f in set(folds.tolist()):
        v = (folds == f).astype(float).reshape(-1, 1)
        mask += v @ v.T
    den = float(w @ metric @ w - w.sum())
    num = float(np.sum((1 - mask) * (w.reshape(-1, 1) @ w.reshape(1, -1)) * metric))
    return 0.0 if den <= 0 else num / den

def leakage_pairwise(
        split: Dict[str, str],
        names: List[str],
        metric: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        fold_pairs=(("train", "test"), ("train", "val"), ("val", "test")),
) -> Dict[str, float]:
    """Per-pair-of-folds leakage, directly comparable to Table 3 of the paper."""
    folds = np.array([split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])
    den = (float(w @ metric @ w) - float(w.sum())) / 2.0
    out = {}
    for s, sp in fold_pairs:
        a, b = (folds == s) * w, (folds == sp) * w
        out[f"{s}-{sp}"] = 0.0 if den <= 0 else float(a @ metric @ b) / den
    return out

@dataclass
class SplitResult:
    split: Dict[str, str]
    k: int
    n_operations: int
    leakage_start: float
    leakage_end: float
    leakage_achieved: float
    percentage_requested: float
    percentage_achieved: float


def generate_split(start_split, end_split, ranked_operations, names, metric, percentage, weights=None):
    if not (0 <= percentage <= 100):
        raise ValueError("percentage must be between 0 and 100")
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)
    target = L_start + percentage / 100 * (L_end - L_start)
    n_ops = len(ranked_operations)
    increasing = L_end >= L_start

    lo, hi = 0, n_ops
    best_k, best_split, best_L = 0, start_split, L_start
    while lo <= hi:
        mid = (lo + hi) // 2
        split_mid = apply_k_cycles(start_split, ranked_operations, end_split, mid)
        L_mid = leakage(split_mid, names, metric, weights)
        if abs(L_mid - target) < abs(best_L - target):
            best_k, best_split, best_L = mid, split_mid, L_mid
        if (L_mid < target) == increasing:
            lo = mid + 1
        else:
            hi = mid - 1

    achieved = 0.0 if L_end == L_start else (best_L - L_start) / (L_end - L_start)
    return SplitResult(split=best_split, k=best_k, n_operations=n_ops,
                        leakage_start=L_start, leakage_end=L_end, leakage_achieved=best_L,
                        percentage_requested=percentage, percentage_achieved=achieved)

def generate_split_linear(start_split, end_split, ranked_operations, names, metric, percentage, weights=None):
    if not (0 <= percentage <= 100):
        raise ValueError("percentage must be between 0 and 100")
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)
    n_ops = len(ranked_operations)
    k = round(n_ops * percentage / 100)

    split = apply_k_cycles(start_split, ranked_operations, end_split, k)
    L_k = leakage(split, names, metric, weights)

    achieved = 0.0 if L_end == L_start else (L_k - L_start) / (L_end - L_start)
    return SplitResult(split=split, k=k, n_operations=n_ops,
                        leakage_start=L_start, leakage_end=L_end, leakage_achieved=L_k,
                        percentage_requested=percentage, percentage_achieved=achieved)


def plot_leakage_curve(start_split, end_split, ranked_operations, names, metric, out_path, weights=None, max_points=300):
    # leakage() rebuilds a full n x n matrix every call. With thousands of
    # operations that dominates runtime for BOTH ranking methods. Sample at
    # most max_points of them -- apply_cycle still runs for every operation,
    # so the final split is unaffected, only curve resolution changes.
    n_ops = len(ranked_operations)
    stride = max(1, n_ops // max_points)

    split = dict(start_split)
    ks = [0]
    leakages = [leakage(split, names, metric, weights)]
    for i, (operation, _) in enumerate(ranked_operations, start=1):
        split = apply_cycle(split, operation, end_split)
        if i % stride == 0 or i == n_ops:
            ks.append(i)
            leakages.append(leakage(split, names, metric, weights))
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, leakages, "-", lw=1.5)
    plt.axhline(L_start, color='#e34948', lw=1.5, ls='--', label='L(start_split)')
    plt.axhline(L_end, color='#1baf7a', lw=1.5, ls='--', label='L(end_split)')
    plt.xlabel("k (number of ranked operations applied)")
    plt.ylabel("Leakage")
    plt.title("Leakage vs. number of applied cycle operations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    return ks, leakages

def cluster_precomputed_matrix(
        names: List[str],
        metric: np.ndarray,
        num_clusters: int,
        linkage: str = "average",
        prefix: str = "",
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    from datasail.reader.utils import DataSet
    from datasail.cluster.clustering import additional_clustering

    dataset = DataSet(
        names=list(names), cluster_names=list(names),
        cluster_map={n: n for n in names},
        cluster_weights={n: 1.0 for n in names},
        num_clusters=num_clusters,
        cluster_similarity=metric,
    )
    reduced = additional_clustering(dataset, n_clusters=num_clusters, linkage=linkage)
    cluster_names = [f"{prefix}{c}" for c in reduced.cluster_names]
    cluster_map = {orig: f"{prefix}{c}" for orig, c in reduced.cluster_map.items()}
    return cluster_names, cluster_map, reduced.cluster_similarity
    
def cluster_within_groups(names, metric, label_start, label_end, total_clusters=50,
                          linkage="average", entity_weights=None):

    name_to_idx = {n: i for i, n in enumerate(names)}
    groups: Dict[Tuple[str, str], List[str]] = {}
    for n in names:
        groups.setdefault((label_start[n], label_end[n]), []).append(n)

    raw = {k: total_clusters * len(v) / len(names) for k, v in groups.items()}
    budget = {k: max(1, round(b)) for k, b in raw.items()}
    drift = total_clusters - sum(budget.values())
    if drift != 0:
        biggest = max(groups, key=lambda k: len(groups[k]))
        budget[biggest] = max(1, budget[biggest] + drift)

    full_cluster_map: Dict[str, str] = {}
    for key, members in groups.items():
        if len(members) <= budget[key]:
            for m in members:
                full_cluster_map[m] = m  # too small to reduce -- each entity is its own cluster
        else:
            sub_idx = [name_to_idx[m] for m in members]
            sub_metric = metric[np.ix_(sub_idx, sub_idx)]
            _, group_map, _ = cluster_precomputed_matrix(
                members, sub_metric, budget[key], linkage=linkage, prefix=f"{key[0]}-{key[1]}::",
            )
            full_cluster_map.update(group_map)

    cluster_names = sorted(set(full_cluster_map.values()))
    members_of = {c: [] for c in cluster_names}
    for n in names:
        members_of[full_cluster_map[n]].append(name_to_idx[n])

    agg_fn = {"average": np.mean, "single": np.max, "complete": np.min}[linkage]
    k = len(cluster_names)
    cluster_metric = np.zeros((k, k))
    for a, ca in enumerate(cluster_names):
        for b, cb in enumerate(cluster_names):
            cluster_metric[a, b] = agg_fn(metric[np.ix_(members_of[ca], members_of[cb])])

    ew = (np.ones(len(names)) if entity_weights is None
          else np.array([entity_weights.get(n, 1.0) for n in names]))
    cluster_weights = {c: float(ew[members_of[c]].sum()) for c in cluster_names}
    cluster_label_start = {full_cluster_map[n]: label_start[n] for n in names}  # pure by construction
    cluster_label_end = {full_cluster_map[n]: label_end[n] for n in names}      # pure by construction

    return cluster_names, full_cluster_map, cluster_metric, cluster_weights, cluster_label_start, cluster_label_end

def _find_fold_file(split_dir, fold):
    for ext, sep in [(".csv", ","), (".tsv", "\t")]:
        path = Path(split_dir) / f"{fold}{ext}"
        if path.exists():
            return path, sep
    return None, None

def continuous_split_1D(
    start_dir: Path,
    end_dir: Path,
    out_dir: Path,
    datatype: str,
    id_col: str,
    value_col: str,
    percentages: List[float],
    similarity: str = "ecfp",
    folds: Optional[List[str]] = None,
    plots: bool = True,
    curve: bool = True,
    weights: Optional[Dict[str, float]] = None,
    num_clusters: Optional[int] = None,
    ranking_method: str = "leakage",
    ranking_seed: Optional[int] = None,
    cache_path: Optional[Path] = None,
    k_method: str = "leakage", 
    dataset=None,
) -> pd.DataFrame:

    from datasail.report import save_t_sne

    folds = folds or ["train", "val", "test"]
    start_dir, end_dir, out_dir = Path(start_dir), Path(end_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_fold_labels(split_dir):
        label, pts = {}, {}
        for fold in folds:
            path, sep = _find_fold_file(split_dir, fold)
            if path is None:
                continue
            df = pd.read_csv(path, sep=sep).drop_duplicates(subset=id_col)
            for _, row in df.iterrows():
                label[row[id_col]] = fold
                pts[row[id_col]] = row[value_col]
        return pts, label
    start_points, label_start = load_fold_labels(start_dir)
    end_points, label_end = load_fold_labels(end_dir)
    assert set(start_points) == set(end_points), "start_dir and end_dir must cover the same entities"
    points = start_points
    def load_all_rows(split_dir):
        parts = []
        for f in folds:
            path, sep = _find_fold_file(split_dir, f)
            if path is not None:
                parts.append(pd.read_csv(path, sep=sep))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    all_rows = pd.concat([load_all_rows(start_dir), load_all_rows(end_dir)], ignore_index=True).drop_duplicates()
    if weights is None:
        weights = all_rows[id_col].value_counts().to_dict()
    if dataset is None:
        dataset = build_similarity_dataset(datatype, points, similarity=similarity, cache_path=cache_path)
    names, metric = dataset.cluster_names, dataset.cluster_similarity

    if num_clusters is not None and num_clusters < len(names):
        (interp_names, cluster_map, interp_metric, interp_weights,
         interp_label_start, interp_label_end) = cluster_within_groups(
             names, metric, label_start, label_end, total_clusters=num_clusters,
             entity_weights=weights,
             )
    else:
        cluster_map = {n: n for n in names}
        interp_names, interp_metric, interp_weights = names, metric, weights
        interp_label_start, interp_label_end = label_start, label_end

    dropped_ids = set(points) - set(names)
    restore_map = {}
    if dropped_ids:
        if datatype == "M":
            from rdkit import Chem
            def canon_key(smiles):
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToInchi(mol) if mol is not None else smiles
        else:
            def canon_key(value):
                return value  # exact-value grouping for non-molecule datatypes

        key_to_survivor = {}
        for n in names:
            key_to_survivor.setdefault(canon_key(points[n]), n)
        for did in dropped_ids:
            key = canon_key(points[did])
            if key in key_to_survivor:
                restore_map[did] = key_to_survivor[key]
        print(f"Restoring {len(restore_map)}/{len(dropped_ids)} dropped entities via structural duplicates "
              f"({len(dropped_ids) - len(restore_map)} had no surviving duplicate; falling back to start label)")
        
    graph = build_split_graph(interp_label_start, interp_label_end, interp_names)
    
    cycles, leftover = decompose(graph)
    if ranking_method == "leakage":
        ranked = rank_operations({(1, 1): (cycles, leftover)}, interp_label_start, interp_label_end,
                                 interp_names, interp_metric, interp_weights)
    elif ranking_method == "random":
        ranked = rank_operations_random({(1, 1): (cycles, leftover)}, interp_label_start, interp_label_end,
                                 interp_names, interp_metric, interp_weights, seed=ranking_seed)
    else:
        raise ValueError(f"Unknown ranking_method: {ranking_method!r}. Use 'leakage' or 'random'.")

    if curve:
        plot_leakage_curve(interp_label_start, interp_label_end, ranked, interp_names, interp_metric,
                            out_dir / "leakage_vs_k.png", weights=interp_weights)

    fold_names = sorted(set(interp_label_start.values()) | set(interp_label_end.values()))
    results = []
    for pct in percentages:
        if k_method == "leakage":
            result = generate_split(interp_label_start, interp_label_end, ranked, interp_names,
                                     interp_metric, percentage=pct, weights=interp_weights)
        elif k_method == "linear":
            result = generate_split_linear(interp_label_start, interp_label_end, ranked, interp_names,
                                            interp_metric, percentage=pct, weights=interp_weights)
        else:
            raise ValueError(f"Unknown k_method: {k_method!r}. Use 'leakage' or 'linear'.")

        entity_split = {n: result.split[cluster_map[n]] for n in names}

        
        for did in dropped_ids:
            if did in restore_map:
                entity_split[did] = entity_split[restore_map[did]]
            else:
                entity_split[did] = label_start[did]  
                
        pct_dir = out_dir / f"pct_{pct}"
        pct_dir.mkdir(parents=True, exist_ok=True)
        row_folds = all_rows[id_col].map(entity_split)
        for fold in fold_names:
            all_rows[row_folds == fold].to_csv(pct_dir / f"{fold}.csv", index=False)

        if plots:
            save_t_sne(pct_dir, dataset, entity_split, entity_split, fold_names)

        pw = leakage_pairwise(result.split, interp_names, interp_metric, interp_weights)
        results.append({"percentage_requested": pct, "percentage_achieved": result.percentage_achieved,
                         "leakage": result.leakage_achieved, "k": result.k,
                         "n_operations": result.n_operations,
                         **{f"leakage_{k}": v for k, v in pw.items()}})
        print(f"pct={pct:>3} | k={result.k:>4}/{result.n_operations} | leakage={result.leakage_achieved:.4f} "
              f"| train-test={pw['train-test']:.4f} train-val={pw['train-val']:.4f} val-test={pw['val-test']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "leakage_summary.csv", index=False)
    return results_df

def assert_entity_pure(split_dir, id_col, value_col, folds, axis_name, strict=True):
    split_dir = Path(split_dir)
    seen: Dict[str, set] = {}
    for fold in folds:
        path, sep = _find_fold_file(split_dir, fold)
        if path is None:
            continue
        for v in pd.read_csv(path, sep=sep)[value_col]:
            seen.setdefault(v, set()).add(fold)
    bad = [v for v, fs in seen.items() if len(fs) > 1]
    profile = {f: sum(1 for fs in seen.values() if len(fs) == 1 and f in fs) for f in folds}
    if bad:
        msg = (f"{split_dir.name} is not a {axis_name} split: {len(bad)} of {len(seen)} "
               f"entities appear in more than one fold. Interpolating towards it would "
               f"require inventing an entity->fold map by majority vote.")
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    print(f"  {split_dir.name:>8} {axis_name:>8}: {len(seen)} entities, "
          + " ".join(f"{f} {profile[f]}" for f in folds)
          + (f", {len(bad)} impure" if bad else ""))
    return profile
 
 
def _fold_pairs(folds):
    return tuple((a, b) for i, a in enumerate(folds) for b in folds[i + 1:])
 
 
def _axis_leakage(dataset, split, kept_ids, weights, folds):
    """Eq. (1) on the interactions that were actually written out, at entity resolution."""
    order = {n: i for i, n in enumerate(dataset.cluster_names)}
    names = [n for n in dataset.cluster_names if n in split and n in kept_ids]
    if len(names) < 2:
        return {f"{a}-{b}": float("nan") for a, b in _fold_pairs(folds)}
    pos = [order[n] for n in names]
    metric = dataset.cluster_similarity[np.ix_(pos, pos)]
    return leakage_pairwise({n: split[n] for n in names}, names, metric,
                            weights, fold_pairs=_fold_pairs(folds))
 

def load_fold_labels_2d(
        split_dir: Path,
        id_col: str,
        value_col: str,
        folds: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    from collections import Counter
    folds = folds or ["train", "val", "test"]
    split_dir = Path(split_dir)
    votes: Dict[str, "Counter"] = {}
    pts: Dict[str, str] = {}
    for fold in folds:
        path, sep = _find_fold_file(split_dir, fold)
        if path is None:
            continue
        df = pd.read_csv(path, sep=sep)
        for _, row in df.iterrows():
            eid = row[id_col]
            votes.setdefault(eid, Counter())[fold] += 1
            pts[eid] = row[value_col]
    label = {eid: counter.most_common(1)[0][0] for eid, counter in votes.items()}
    return pts, label

def continuous_split_2D(
        e_start_dir: Path, e_end_dir: Path,     # protein axis, e.g. I1P -> S1P
        f_start_dir: Path, f_end_dir: Path,     # ligand  axis, e.g. I1L -> S1L
        out_dir: Path,
        e_datatype: str, e_id_col: str, e_value_col: str,
        f_datatype: str, f_id_col: str, f_value_col: str,
        percentages: List,                      # 50 -> both axes at 50 %; (25, 75) -> grid point
        e_similarity: str = "diamond",
        f_similarity: str = "ecfp",
        folds: Optional[List[str]] = None,
        e_num_clusters: Optional[int] = None,
        f_num_clusters: Optional[int] = None,
        ranking_method: str = "leakage",
        ranking_seed: Optional[int] = None,
        cache_path: Optional[Path] = None,
        k_method: str = "leakage",
        strict: bool = True,
) -> pd.DataFrame:
    from datasail.settings import NOT_ASSIGNED
    import tempfile
 
    folds = folds or ["train", "val", "test"]
    e_start_dir, e_end_dir = Path(e_start_dir), Path(e_end_dir)
    f_start_dir, f_end_dir = Path(f_start_dir), Path(f_end_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    pairs = [tuple(p) if isinstance(p, (tuple, list)) else (p, p) for p in percentages]
    axis_pcts = sorted({p for pr in pairs for p in pr})
 
    print("Endpoint checks")
    e_prof_start = assert_entity_pure(e_start_dir, e_id_col, e_value_col, folds, "protein", strict)
    e_prof_end = assert_entity_pure(e_end_dir, e_id_col, e_value_col, folds, "protein", strict)
    f_prof_start = assert_entity_pure(f_start_dir, f_id_col, f_value_col, folds, "ligand", strict)
    f_prof_end = assert_entity_pure(f_end_dir, f_id_col, f_value_col, folds, "ligand", strict)
 
    for axis, a, b in [("protein", e_prof_start, e_prof_end),
                       ("ligand", f_prof_start, f_prof_end)]:
        if a == b:
            print(f"  {axis} fold sizes match {a} -- pure cycles, sizes constant across the sweep")
        else:
            print(f"  WARNING: {axis} fold sizes differ, {a} vs {b}. The decomposition will "
                  f"contain leftover singletons; entity proportions and the leakage baseline "
                  f"will drift along the sweep.")
 
    def _dir_rows(d):
        parts = []
        for fold in folds:
            path, sep = _find_fold_file(Path(d), fold)
            if path is not None:
                parts.append(pd.read_csv(path, sep=sep))
        if not parts:
            raise ValueError(f"no fold files found in {d}")
        return pd.concat(parts, ignore_index=True)
 
    frames = {"e_start": _dir_rows(e_start_dir), "e_end": _dir_rows(e_end_dir),
              "f_start": _dir_rows(f_start_dir), "f_end": _dir_rows(f_end_dir)}
    keys = {k: set(map(tuple, r[[e_id_col, f_id_col]].values)) for k, r in frames.items()}
    ref = keys["f_start"]
    if any(v != ref for v in keys.values()):
        msg = ("the four splits do not cover the same interactions: "
               + ", ".join(f"{k}={len(v)}" for k, v in keys.items()))
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    all_rows = frames["f_start"].drop_duplicates(subset=[e_id_col, f_id_col]).reset_index(drop=True)
    print(f"  {len(all_rows)} interactions, identical across all four splits")
 
    e_weights = all_rows[e_id_col].value_counts().to_dict()
    f_weights = all_rows[f_id_col].value_counts().to_dict()
 
    e_start_points, e_start_label = load_fold_labels_2d(e_start_dir, e_id_col, e_value_col, folds)
    e_end_points, e_end_label = load_fold_labels_2d(e_end_dir, e_id_col, e_value_col, folds)
    f_start_points, f_start_label = load_fold_labels_2d(f_start_dir, f_id_col, f_value_col, folds)
    f_end_points, f_end_label = load_fold_labels_2d(f_end_dir, f_id_col, f_value_col, folds)
 
    e_common = set(e_start_label) & set(e_end_label)
    f_common = set(f_start_label) & set(f_end_label)
    dropped_e = (set(e_start_label) | set(e_end_label)) - e_common
    dropped_f = (set(f_start_label) | set(f_end_label)) - f_common
    if dropped_e or dropped_f:
        print(f"  WARNING: dropping {len(dropped_e)} protein(s) and {len(dropped_f)} molecule(s) "
              f"not present in both endpoints of their axis")
    e_start_points = {k: v for k, v in e_start_points.items() if k in e_common}
    e_end_points = {k: v for k, v in e_end_points.items() if k in e_common}
    e_start_label = {k: v for k, v in e_start_label.items() if k in e_common}
    e_end_label = {k: v for k, v in e_end_label.items() if k in e_common}
    f_start_points = {k: v for k, v in f_start_points.items() if k in f_common}
    f_end_points = {k: v for k, v in f_end_points.items() if k in f_common}
    f_start_label = {k: v for k, v in f_start_label.items() if k in f_common}
    f_end_label = {k: v for k, v in f_end_label.items() if k in f_common}
 
    tmp_root = Path(tempfile.mkdtemp(prefix="axis_projection_"))
    e_start_proj, e_end_proj = tmp_root / "e_start", tmp_root / "e_end"
    f_start_proj, f_end_proj = tmp_root / "f_start", tmp_root / "f_end"
 
    def write_axis_dir(target_dir, points, label, id_col, value_col):
        target_dir.mkdir(parents=True, exist_ok=True)
        rows_by_fold = {f: [] for f in folds}
        for name, fold in label.items():
            rows_by_fold[fold].append({id_col: name, value_col: points[name]})
        for fold, rows in rows_by_fold.items():
            if not rows:
                continue
            pd.DataFrame(rows).to_csv(target_dir / f"{fold}.csv", index=False)
 
    write_axis_dir(e_start_proj, e_start_points, e_start_label, e_id_col, e_value_col)
    write_axis_dir(e_end_proj, e_end_points, e_end_label, e_id_col, e_value_col)
    write_axis_dir(f_start_proj, f_start_points, f_start_label, f_id_col, f_value_col)
    write_axis_dir(f_end_proj, f_end_points, f_end_label, f_id_col, f_value_col)
 
    e_out, f_out = out_dir / "e_axis", out_dir / "f_axis"
 
    def _axis_cache(suffix):
        if cache_path is None:
            return None
        p = Path(cache_path)
        return p.with_name(p.stem + suffix + p.suffix)
 
    e_cache_path, f_cache_path = _axis_cache("_e"), _axis_cache("_f")
 
    e_summary = continuous_split_1D(
        start_dir=e_start_proj, end_dir=e_end_proj, out_dir=e_out,
        datatype=e_datatype, id_col=e_id_col, value_col=e_value_col,
        percentages=axis_pcts, similarity=e_similarity, folds=folds,
        plots=False, curve=True, num_clusters=e_num_clusters, ranking_method=ranking_method,
        ranking_seed=ranking_seed, k_method=k_method, weights=e_weights, cache_path=e_cache_path,
    )
    f_summary = continuous_split_1D(
        start_dir=f_start_proj, end_dir=f_end_proj, out_dir=f_out,
        datatype=f_datatype, id_col=f_id_col, value_col=f_value_col,
        percentages=axis_pcts, similarity=f_similarity, folds=folds,
        plots=False, curve=True, num_clusters=f_num_clusters, ranking_method=ranking_method,
        ranking_seed=ranking_seed, k_method=k_method, weights=f_weights, cache_path=f_cache_path,
    )
 
    e_ds = build_similarity_dataset(e_datatype, e_start_points, similarity=e_similarity,
                                    cache_path=e_cache_path)
    f_ds = build_similarity_dataset(f_datatype, f_start_points, similarity=f_similarity,
                                    cache_path=f_cache_path)
 
    def load_axis_split(axis_out_dir, id_col_here, pct):
        split = {}
        for fold in folds:
            path = axis_out_dir / f"pct_{pct}" / f"{fold}.csv"
            if not path.exists():
                continue
            for v in pd.read_csv(path)[id_col_here]:
                split[v] = fold
        return split
 
    e_splits = {p: load_axis_split(e_out, e_id_col, p) for p in axis_pcts}
    f_splits = {p: load_axis_split(f_out, f_id_col, p) for p in axis_pcts}
 
    results = []
    for e_pct, f_pct in pairs:
        e_split, f_split = e_splits[e_pct], f_splits[f_pct]
        e_col = all_rows[e_id_col].map(e_split)
        f_col = all_rows[f_id_col].map(f_split)
        agree = e_col.notna() & f_col.notna() & (e_col == f_col)
        inter_fold = np.where(agree, e_col, NOT_ASSIGNED)
        coverage = float(agree.mean())
 
        tag = f"pct_{e_pct}" if e_pct == f_pct else f"e{e_pct}_f{f_pct}"
        pct_dir = out_dir / tag
        pct_dir.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            all_rows[inter_fold == fold].to_csv(pct_dir / f"{fold}.csv", index=False)
 
        kept = all_rows[agree]
        psl_pw = _axis_leakage(e_ds, e_split, set(kept[e_id_col]),
                               kept[e_id_col].value_counts().to_dict(), folds)
        msl_pw = _axis_leakage(f_ds, f_split, set(kept[f_id_col]),
                               kept[f_id_col].value_counts().to_dict(), folds)
        psl, msl = float(np.nansum(list(psl_pw.values()))), float(np.nansum(list(msl_pw.values())))
 
        row = {"e_percentage": e_pct, "f_percentage": f_pct, "rows": int(len(kept)),
               "coverage": coverage, "PSL": psl, "MSL": msl, "TSL": psl + msl}
        row.update({f"PSL_{k}": v for k, v in psl_pw.items()})
        row.update({f"MSL_{k}": v for k, v in msl_pw.items()})
        # leakage of the axis projections over the full row set, for reference
        row["PSL_axis"] = float(e_summary.loc[e_summary.percentage_requested == e_pct, "leakage"].iloc[0])
        row["MSL_axis"] = float(f_summary.loc[f_summary.percentage_requested == f_pct, "leakage"].iloc[0])
        for fold in folds:
            row[f"n_{fold}"] = int((inter_fold == fold).sum())
        empty = [f for f in folds if row[f"n_{f}"] == 0]
        if empty:
            print(f"  WARNING: fold(s) {empty} are empty at e={e_pct} f={f_pct}. The agreement "
                  f"rule can starve a small fold when the two axes stop aligning on it.")
        results.append(row)
 
        print(f"e={e_pct:>3} f={f_pct:>3} | PSL={psl:.4f} MSL={msl:.4f} TSL={psl + msl:.4f} "
              f"| rows={len(kept):>6} coverage={coverage:.3f} "
              f"| " + " ".join(f"{f}={row[f'n_{f}']}" for f in folds))
 
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "leakage_summary_2d_via_axes.csv", index=False)
    return results_df
