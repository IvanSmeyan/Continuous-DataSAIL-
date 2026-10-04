## Idea of continuous DataSAIL splitting

Permutations define the difference between two splits. This package writes
those permutations down as a graph, breaks it into **cycles** and **leftovers** (single
entities that move from one fold to another). Apply the first
*k* operations, and an intermediate split is achieved. Apply all of them, and one lands exactly
on the second end split. Choosing *k* to hit a target leakage is what the package does for you.


## Input format

A "split" is a directory with one file per fold (both .csv or .tsv formats):

```
my_split/
  train.csv     
  val.csv
  test.csv
```

## Key functions

### One axis

Sweep the molecule axis from one ligand split to another:

```python
import continuous_datasail as cds

summary = cds.continuous_split_1D(
    start_dir="splits/I1L",
    end_dir="splits/S1L",
    out_dir="out/f_axis",
    datatype="M", id_col="molecule ID", value_col="SMILES",
    percentages=[0, 25, 50, 75, 100],
    similarity="ecfp",
    folds=["train", "val", "test"],
    ranking_method="random", ranking_seed=0,
    plots=False,
)
```

You get `out/f_axis/pct_0/`, `pct_25/`, … each holding `train.csv`, `val.csv`, `test.csv`,
plus `leakage_summary.csv` and a `leakage_vs_k.png` curve. Every input row appears in
exactly one fold of every percentage. 

### Two axes

Interpolate proteins and molecules at the same time, then keep the interactions where the
two axes agree:

```python
summary = cds.continuous_split_2D(
    e_start_dir="splits/I1P", e_end_dir="splits/S1P",     # protein axis
    f_start_dir="splits/I1L", f_end_dir="splits/S1L",     # molecule axis
    out_dir="out/2d",
    e_datatype="P", e_id_col="Uniprot ID", e_value_col="Sequence",
    f_datatype="M", f_id_col="molecule ID", f_value_col="SMILES",
    percentages=[0, 25, 50, 75, 100],
    e_similarity="diamond", f_similarity="ecfp",
    e_num_clusters=1500,
    ranking_method="random", ranking_seed=0,
)
```

An interaction is kept only when its protein and its molecule land in the same fold, so
the result is a true two-dimensional split — but you will lose rows. The function reports
`coverage` (the fraction kept) at every percentage. Expect somewhere between 50 % and 80 %.

# Function reference

## Main functions

### `continuous_split_1D(...)`

Sweeps one axis, proteins or molecules, from one split to another and writes a full set
of fold files for each requested percentage. Every input row lands in exactly one fold at
every step, so there is no data loss anywhere along the sweep. Returns a summary
`DataFrame` and writes it to `leakage_summary.csv`.

| argument | what it does |
|---|---|
| `start_dir`, `end_dir` | the two splits to interpolate between, as directories of fold files |
| `out_dir` | where the results go, one `pct_<n>/` subdirectory per percentage |
| `datatype` | `"P"` for proteins, `"M"` for molecules |
| `id_col`, `value_col` | which columns hold the entity id and the format similarity is computed on|
| `percentages` | the points you want along the sweep, e.g. `[0, 25, 50, 75, 100]` |
| `similarity` | how similarity is measured: `"ecfp"` for molecules, `"diamond"` for proteins |
| `folds` | fold names, defaults to `["train", "val", "test"]`, must match your filenames |
| `plots` | `True` writes a t-SNE picture per percentage. Slow on big datasets |
| `curve` | `True` writes `leakage_vs_k.png`, the leakage against every value of *k* |
| `weights` | how much each entity counts for|
| `num_clusters` | group entities into this many clusters before interpolating |
| `ranking_method` | `"leakage"` puts the highest-impact moves first, so leakage changes fast at the start and slowly later. `"random"` shuffles them, which gives a straighter, more even sweep|
| `ranking_seed` | seed for `"random"`. Set it, or every run gives a different sweep |
| `cache_path` | where to pickle the similarity matrix. When set,  second run skips the most expensive part of similarity matrix computing |
| `k_method` | `"leakage"` searches for the *k* whose leakage matches your target (search on the y-axis). `"linear"` just takes `k = n × pct / 100` (search on the x-axis). The second method is faster.
| `dataset` | a similarity dataset which eas already built, to reuse it across several runs in one process instead of writing it to disk |

### `continuous_split_2D(...)`

Sweeps both axes at once. It runs `continuous_split_1D` on the protein axis and on the
molecule axis separately, then keeps the interactions whose two axes agree on a fold so
every output split is two-dimensional, at the cost of dropping the rows where
they disagree. Before doing any of that, it checks that all four inputs are real
entity-level splits covering the same interactions, and refuses to run if they aren't.

| argument | what it does |
|---|---|
| `e_start_dir`, `e_end_dir` | the two endpoints for the protein axis |
| `f_start_dir`, `f_end_dir` | the two endpoints for the molecule axis |
| `out_dir` | resulting directory, plus an `e_axis/` and `f_axis/` subdirectory holding each axis's own interpolation |
| `e_datatype`, `e_id_col`, `e_value_col` | the protein side: `"P"`, and the two column names |
| `f_datatype`, `f_id_col`, `f_value_col` | the molecule side: `"M"`, and the two column names |
| `percentages` |the points you want along the sweep, e.g. `[0, 25, 50, 75, 100]` |
| `e_similarity`, `f_similarity` | usually `"diamond"` and `"ecfp"` |
| `folds` | fold names, defaults to `["train", "val", "test"]` |
| `e_num_clusters`, `f_num_clusters` | cluster each axis before interpolating |
| `ranking_method`, `ranking_seed`, `k_method`, `cache_path` | as in `continuous_split_1D`, passed straight through to both axes |

## Measuring leakage

### `leakage(split, names, metric, weights=None)`

Returns one number: the share of all similarity in the dataset that sits between
different folds rather than inside one, following the structure of datasail's own leakage function.

### `leakage_pairwise(split, names, metric, weights=None, fold_pairs=...)`

The same quantity, but reported separately for each pair of folds — train–test,
train–val, val–test.

### `plot_leakage_curve(start_split, end_split, ranked_operations, names, metric, out_path, weights=None, max_points=300)`

Draws leakage against the number of operations applied and saves it to `out_path`, with
dashed lines marking the two endpoints. `max_points` caps how many points get
computed, since each one costs a full pass over the similarity matrix.

### `assert_entity_pure(split_dir, id_col, value_col, folds, axis_name, strict=True)`

Checks that every entity in a split lives in exactly one fold, and returns how many are in
each. It compares on `value_col` rather than the id on purpose. With `strict=False`
it warns instead of raising.

## Building the list of operations (cycles, swaps, leftovers)

### `build_split_graph(start_split, end_split, names)`

Turns the difference between two splits into a directed graph, with one edge per entity
that changes fold. Raises an error if either split is missing an entity from `names`.

### `decompose(graph)`

Breaks that graph into cycles and leftovers, and returns both. Cycles rotate folds among a
group of entities and leave the fold sizes alone, whereas leftovers are single moves that change
them. If `leftover` comes back empty, two endpoints have matching fold sizes. This decomposition is based on 
Hierholzer's algorithm for Eulerian circuits, which is deterministic yet finds only one decomposition among many.

### `rank_operations(strata_cycles, start_split, end_split, names, metric, weights=None)`

Orders the moves by how much each one changes leakage, smallest first. The deltas are all
measured against the starting split.

### `rank_operations_random(strata_cycles, ..., seed=None)`

Shuffles the moves instead of ranking them. Set `seed` for a reproducible sweep — without
it, every run walks a different path from start to end.

### `apply_cycle(split, operation, end_split)`

Applies a single move to a split and returns the new one — rotating folds around a cycle,
or sending one leftover entity to its destination fold.

### `apply_k_cycles(start_split, ranked_operations, end_split, k)`

Applies the first `k` moves in order and returns the split you end up with. Each entity is
touched by at most one move, so the order doesn't affect where you finish.

### `generate_split(start_split, end_split, ranked_operations, names, metric, percentage, weights=None)`

Finds the `k` whose leakage sits closest to `percentage` of the way from the first split's
leakage to the second's, and returns a `SplitResult` with the split, the `k`, and both the
requested and achieved percentages. It searches by halving the range, which assumes leakage
moves steadily with `k`.

### `generate_split_linear(...)`

Same signature, but skips the search and takes `k = n × percentage / 100` directly. Much
cheaper, and roughly as good when the moves were ordered randomly.

### `SplitResult`

A small record holding the resulting `split`, the `k` used, the starting, ending, and
achieved leakage, and the requested versus achieved percentage.

## Similarity and clustering

### `build_similarity_dataset(datatype, data, similarity="ecfp", cache_path=None)`

Asks DataSAIL for the all-against-all similarity matrix over your entities. Give it a
`cache_path` and it pickles the result, so the next run loads in seconds instead of
recomputing.

### `cluster_within_groups(names, metric, label_start, label_end, total_clusters=50, linkage="average", entity_weights=None)`

Reduces a large set of entities to a smaller set of clusters so the interpolation stays
fast. It clusters *within* groups of entities that share the same start fold and end fold,
which guarantees no cluster straddles a fold boundary. Pass `entity_weights` so a cluster's
weight is the sum of its members' interaction counts rather than a plain head count.

### `cluster_precomputed_matrix(names, metric, num_clusters, linkage="average", prefix="")`

Runs DataSAIL's agglomerative clustering on a similarity matrix you already have, and
returns the cluster names, the entity→cluster mapping, and the cluster-level matrix.

## Reading files

### `_find_fold_file(split_dir, fold)`

Looks for `<fold>.csv` then `<fold>.tsv` in a directory and returns the path and the right
separator, or `(None, None)` if neither exists. This is the single place file format is
decided; `.csv` wins if both are present.

### `load_fold_labels_2d(split_dir, id_col, value_col, folds=None)`

Reads a split directory and returns two dictionaries: each entity's value (its sequence or
SMILES) and its fold. When an entity appears in more than one fold, it takes the most
common one. Running `assert_entity_pure` first is recommended. 
