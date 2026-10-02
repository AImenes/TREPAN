# From the papers to the code

This document maps every component of TREPAN, as described in Craven & Shavlik
(1995) and in Chapter 3 of Craven's thesis (1996), to its implementation here, and
records the places where the sources leave room for interpretation.

## 1. The main loop — `trepan/core.py`, `Trepan.fit`

Thesis Figure 9 (`Trepan`) is followed line by line:

| Thesis | Code |
|---|---|
| `for each x in S: class label for x := Oracle(x)` | `y_oracle = self.oracle_.predict(X_arr)` |
| initialise root as a leaf; build instance model; `DrawSample({}, min_sample − |S|)`; label root | `_new_node`, `FeatureDistributions.fit`, `_fill_pool`, `_decide_leaf`, `_refresh_estimates` |
| `while Queue not empty and global stopping criteria not satisfied` | `while queue and len(self.expansion_order_) < self.max_internal_nodes` |
| `remove N from head of Queue` (best first) | `heapq` ordered by `-node.priority`, `priority = reach · (1 − fidelity)` |
| `T := ConstructTest(F, S_N ∪ query_instances_N)` | `_select_split` → `search.best_binary_split` + `search.search_m_of_n` |
| `for each outcome t of T: make child C; constraints_C; S_C; model; DrawSample; label; queue if not leaf` | the `for outcome in (True, False)` block |

The query instances drawn when a child is created are stored on the node and reused
when the node is later expanded, exactly as the tuple `<N, S_N, query_instances_N,
constraints_N>` in the thesis.

**reach / fidelity** (thesis 3.2.3). For each internal node the fraction of its
instances (training examples *and* query instances) sent down each branch is
recorded; `reach(N)` is the product along the path. `fidelity(N)` is the fraction of
the node's instances whose oracle label equals the node's label.

## 2. The oracle — `trepan/oracle.py`

Membership queries only. Any object with `predict(X)` or any callable works; the
wrapper counts queries and, when the tree was fitted on a DataFrame, presents query
instances to the model as a DataFrame with the same column names.

## 3. Drawing query instances — `trepan/sampling.py`

* **Marginal models** (thesis 3.2.2): `NominalDistribution` (empirical frequencies) and
  `KernelDensityDistribution` (Gaussian kernels, Silverman 1986). The thesis sets the
  kernel width to `1/√m`; its inputs were scaled to `[0, 1]`, so here the width is
  `range/√m` with the range taken over the training set (`kde_bandwidth="craven"`).
  Silverman's and Scott's rules and fixed widths are available.
* **Local models** (thesis 3.2.2, Figure 12): `FeatureDistributions.fit_local`. For
  each feature *not constrained by a test on the path* the node's examples are
  compared with the examples behind the nearest ancestor model — χ² for discrete,
  two-sample Kolmogorov–Smirnov for continuous — at level `alpha / k` (Bonferroni
  over the `k` tested features; thesis `alpha = 0.10`). One rejection makes the node
  fit a complete local model; otherwise the ancestor's model object is reused.
  `min_local_examples` (default 5) is an addition: nodes with fewer examples always
  inherit.
* **DrawInstance** (thesis Figure 13): `draw_instances`. Non-disjunctive constraints
  — binary splits, satisfied n-of-n tests, failed 1-of-n tests — become per-feature
  bounds (`derive_bounds`, `Constraint.forced_literals`) and each feature is sampled
  from its conditional distribution (`sample(..., bounds)`). For each remaining m-of-n
  constraint, `_select_forced_literals` repeatedly computes `Pr_g(l_ij)` for the
  literals still available, selects one with probability proportional to it, adds it to
  the hard bounds of its feature and updates the conditional distribution, until `m`
  literals are forced true (satisfied outcome) or `n − m + 1` forced false. Finally
  every instance is verified against the constraints; the rare violations (integer
  rounding at a threshold) are redrawn.
* Additions: samples of integer-valued continuous features are rounded to integers,
  and continuous samples are kept within the training range
  (`truncate_to_training_range`). Both keep queries realistic and can be switched off.

## 4. Splitting tests — `trepan/splits.py`, `trepan/search.py`

* **Candidate tests** (thesis 3.2.4): one `== value` test for a two-valued feature,
  one per value for larger nominal features, and threshold tests at midpoints between
  adjacent observed values for continuous features — only at *boundary* midpoints
  (Fayyad & Irani, 1992), i.e. where the neighbouring value groups are not all of one
  shared class. Candidates are built from all instances at the node (training examples
  plus query instances), the set `ConstructTest` receives.
* **Criterion**: information gain (thesis; `criterion="gain"`). The NeurIPS paper
  mentions the gain ratio; it is available as `criterion="gain_ratio"`.
* **ConstructMofNTest** (thesis Figure 16): `search_m_of_n`. Beam search of width 2
  with the operators *m-of-n+1* and *m+1-of-n+1*; an application is admissible when the
  new test is significantly different from the test it extends (χ², `split_significance
  = 0.05`) and scores better than the worst test in the beam; the search stops when an
  iteration leaves the beam unchanged. Rules from the thesis text:
  adding the negation of a present literal is allowed and simplified
  (`2-of-{a, b, c, ¬c} → 1-of-{a, b}`, `MofNTest.with_literal`); a literal implied by,
  or implying, a present literal is not eligible (`implies`, `LiteralBank.eligible_mask`);
  a feature already used by an m-of-n test on the path is not available
  (`Node.features_in_compound_splits`); a literal-pruning pass tries dropping each
  literal, in insertion order, with and without decrementing `m`, keeping modifications
  that do not reduce the score (`_prune_literals`).
* **Beam initialisation.** Figure 16 initialises the beam with the best test only, but
  the accompanying text motivates the beam width of two by the need to pursue both
  `x1` and `¬x1` as seeds, and later descriptions of the algorithm initialise it with
  the best test and its complement. The complement is included here for
  `beam_width ≥ 2`.
* **The χ² admissibility test.** The thesis says the test checks whether the new test
  "results in a significantly different partitioning of the instances". This is
  implemented as the thesis' two-sample χ² statistic comparing the class distribution
  of the instances on the satisfied side before and after the change (and likewise for
  the unsatisfied side); the application is admissible if either side differs
  significantly. Small changes that move a few instances fail the test, which is the
  stated purpose.

## 5. Stopping criteria — `trepan/stopping.py`, `Trepan._decide_leaf`

* **Local** (thesis 3.2.5): with `p̂_c = 1`, the one-sided Wilson interval gives the
  number of instances needed for `Pr(prop_c < 1 − ε) < δ`:
  `m_L = z²_δ (1 − ε) / ε` (`required_sample_size`; 52 for ε = δ = 0.05). A node with
  at least `m_L` unanimous instances is a leaf; otherwise TREPAN keeps querying until a
  disagreeing instance appears (expand) or `m_L` unanimous instances have been seen
  (leaf). This is `stopping_rule="strict"`, the default. `stopping_rule="interval"`
  implements the paper's wording literally — the lower confidence bound on `p_c` must
  reach `1 − ε` — which also accepts large nearly-pure nodes; Wilson, Clopper–Pearson
  and Wald intervals are available.
* **Global**: `max_internal_nodes`. With `validation_fraction` (thesis: 10 %) or an
  explicit `X_validation`, the fidelity of every tree in the nested best-first sequence
  is measured and the best (smallest on ties) is returned
  (`_select_by_validation`, `validation_fidelity_curve_`).
* **Pruning** (thesis 3.2.6): `prune_redundant_subtrees` collapses, in post-order,
  subtrees whose leaves all predict the same class. Predictions are unchanged.

## 6. Parameter values used in the sources

| | NeurIPS 1995 | Thesis 1996 |
|---|---|---|
| `min_sample` | 1000 | 10 000 |
| `max_internal_nodes` | 15 | 31 |
| ε, δ | 0.05, 0.05 | –, 0.01 |
| χ² test for m-of-n operators | – | 0.05 |
| local-model test | – | 0.10 (Bonferroni) |
| beam width | hill climbing | 2 |
| validation set | – | 10 % of the training set |

The defaults here (`min_sample=1000`, 15 internal nodes, ε = δ = 0.05) follow the
paper because they run in seconds on small data sets; the thesis settings are a
parameter change away.
