"""
╔══════════════════════════════════════════════════════════════════╗
║   LDA Hyperparameter Tuning — Chunked Bayesian Search (v3)       ║
║                                                                   ║
║   KEY FIXES vs v2:                                                ║
║     1. T_THRESH is NOT a Bayesian parameter anymore.              ║
║        It is swept INSIDE each trial (free, uses same Z_link)     ║
║        to find the best T achieving precision ≥ 0.95.             ║
║        → Objective = multi-fund clusters at that T                ║
║                                                                   ║
║     2. Bayesian tunes only 3 params: K, alpha, c_conf             ║
║        Fewer dims = faster convergence in same # of trials.       ║
║                                                                   ║
║     3. Resume across sessions via SQLite (Optuna) + pickle.       ║
║        Run 20 trials today, 20 more tomorrow — study accumulates. ║
║                                                                   ║
║   Usage:                                                           ║
║     First run  : python lda_hyperopt_tuning_v3.py --trials 20     ║
║     Resume     : python lda_hyperopt_tuning_v3.py --trials 20     ║
║     Reset all  : python lda_hyperopt_tuning_v3.py --reset         ║
║     Show best  : python lda_hyperopt_tuning_v3.py --trials 0      ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, sys, json, time, pickle, argparse, warnings, logging

import numpy as np
import pandas as pd
import lda as lda_pkg
import optuna
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")
logging.getLogger("lda").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LDA Hyperparameter Tuning (chunked)")
parser.add_argument("--trials",  type=int, default=20,
                    help="Number of NEW trials to run this session (default: 20)")
parser.add_argument("--reset",   action="store_true",
                    help="Delete existing study and start fresh")
parser.add_argument("--precision-target", type=float, default=0.90,
                    help="Minimum precision to accept (default: 0.90)")
args = parser.parse_args()

PRECISION_TARGET = args.precision_target

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_DATA_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "ne04j_with_labels.csv"),
    "/mnt/user-data/uploads/ne04j_with_labels.csv",
    "ne04j_with_labels.csv",
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    print("ERROR: ne04j_with_labels.csv not found next to this script.")
    sys.exit(1)

OUT_DIR    = os.path.join(_SCRIPT_DIR, "output", "hypertuned_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# Persistence files
DB_PATH      = os.path.join(OUT_DIR, "optuna_study.db")    # SQLite study store
PICKLE_PATH  = os.path.join(OUT_DIR, "study_snapshot.pkl") # redundant backup
ITER_CSV     = os.path.join(OUT_DIR, "trial_log.csv")        # one row per trial
T_SWEEP_CSV  = os.path.join(OUT_DIR, "trial_t_sweep.csv")   # 15 rows per trial
BEST_JSON    = os.path.join(OUT_DIR, "best_params.json")

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
BETA       = 0.01       # Low β → sharp topic-asset distributions
N_PSEUDO   = 1000       # Resolves 0.1% weight differences
LINKAGE    = "complete" # Most conservative for high-precision clustering
BURN_IN    = 200        # Fast setting for tuning (use 500 in production)
N_SAMPLES  = 15         # Posterior draws per trial (use 100 in production)

# T_THRESH sweep range — evaluated inside every trial at zero extra LDA cost.
# Upper bound extended to 0.50: at precision ≥ 0.90 (vs 0.95 before) there is
# more headroom to merge aggressively and still stay above the target.
T_SWEEP = np.round(np.arange(0.02, 0.52, 0.02), 3).tolist()

# ── Handle reset ──────────────────────────────────────────────────────────────
if args.reset:
    for p in [DB_PATH, PICKLE_PATH, ITER_CSV, T_SWEEP_CSV, BEST_JSON]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted: {p}")
    print("Study reset complete. Run without --reset to start fresh.")
    sys.exit(0)

# ── Data loading ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  LDA HYPERPARAMETER TUNING — CHUNKED (v3)")
print("=" * 65)
print(f"\nData: {DATA_PATH}")

raw = pd.read_csv(DATA_PATH)
raw = raw[~raw["asset"].isin(["Num_Assets", "Risky_Proportion"])].copy()

wm = raw.pivot_table(
    index="fund_name", columns="asset", values="weight",
    aggfunc="sum", fill_value=0.0,
)
wm = wm.div(wm.sum(axis=1), axis=0)

funds  = wm.index.tolist()
assets = wm.columns.tolist()
F, A   = len(funds), len(assets)
W      = wm.values

print(f"Dataset: {F} funds × {A} assets")

# ── Ground-truth labels ───────────────────────────────────────────────────────
NOISE_LABELS = {"Num_Assets", "Risky_Proportion", "Risk_Free"}
lw_map = {}
for _, row in raw.iterrows():
    if row["label"] in NOISE_LABELS:
        continue
    fn, lbl = row["fund_name"], row["label"]
    lw_map.setdefault(fn, {})[lbl] = lw_map.get(fn, {}).get(lbl, 0.0) + row["weight"]

gt_labels = [
    max(lw_map.get(fn, {"Unknown": 1.0}),
        key=lw_map.get(fn, {"Unknown": 1.0}).get)
    for fn in funds
]
true_str   = np.array(gt_labels)
valid_secs = sorted(set(true_str))
sec_to_idx = {s: i for i, s in enumerate(valid_secs)}
true_idx   = np.array([sec_to_idx[lbl] for lbl in true_str])

# Precompute ground-truth comparison matrix (used in every trial)
same_truth = true_idx[:, None] == true_idx[None, :]
upper_tri  = np.triu(np.ones((F, F), dtype=bool), k=1)

n_true_positive_pairs = int((same_truth & upper_tri).sum())
print(f"Ground-truth: {len(valid_secs)} labels | "
      f"{n_true_positive_pairs:,} same-class pairs")

# Pseudo-count matrix (fixed for all trials)
C = np.round(W * N_PSEUDO).astype(np.int32)

# ── JSD helper ────────────────────────────────────────────────────────────────
def jsd_vectorised(Pi3, Qall):
    M    = 0.5 * (Pi3 + Qall)
    eps  = 1e-12
    kl_p = np.where(Pi3  > eps, Pi3  * np.log2(np.maximum(Pi3,  eps) / np.maximum(M, eps)), 0.0).sum(-1)
    kl_q = np.where(Qall > eps, Qall * np.log2(np.maximum(Qall, eps) / np.maximum(M, eps)), 0.0).sum(-1)
    jv   = np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)
    return jv.mean(axis=0), jv.std(axis=0)


# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate(K, alpha, c_conf):
    """
    Runs the LDA pipeline for given (K, alpha, c_conf).
    Sweeps T_THRESH internally to find the best T achieving
    precision >= PRECISION_TARGET with maximum multi-fund clusters.

    Returns
    -------
    dict with:
      best_T          : optimal threshold found
      best_multi_fund : multi-fund clusters at best_T
      best_precision  : precision at best_T
      best_recall     : recall at best_T
      all_T_results   : full sweep table (list of dicts)
      ari, nmi        : cluster quality at best_T
      n_clusters, singletons at best_T
    """
    # --- Gibbs sampling ---
    theta_samples = []
    for s in range(N_SAMPLES):
        model = lda_pkg.LDA(
            n_topics=K, n_iter=BURN_IN,
            alpha=alpha, eta=BETA,
            random_state=42 + s * 17,
            refresh=BURN_IN,
        )
        model.fit(C)
        n_dk    = model.ndz_.astype(np.float64)
        theta_s = n_dk + alpha
        theta_s /= theta_s.sum(axis=1, keepdims=True)
        theta_samples.append(theta_s)

    theta_arr = np.stack(theta_samples, axis=0)   # (S, F, K)

    # --- Pairwise JSD ---
    dist_m = np.zeros((F, F))
    dist_s = np.zeros((F, F))
    for i in range(F):
        Qall = theta_arr[:, i + 1:, :]
        if Qall.shape[1] > 0:
            mij, sij = jsd_vectorised(theta_arr[:, i, :][:, None, :], Qall)
            dist_m[i, i + 1:] = dist_m[i + 1:, i] = mij
            dist_s[i, i + 1:] = dist_s[i + 1:, i] = sij

    dist_c = dist_m + c_conf * dist_s
    np.fill_diagonal(dist_c, 0.0)

    # --- Build dendrogram ONCE, sweep T for free ---
    condensed = squareform(dist_c, checks=False)
    Z_link    = linkage(condensed, method=LINKAGE)

    best = dict(T=None, multi_fund=-1, precision=0.0, recall=0.0,
                ari=0.0, nmi=0.0, n_clusters=0, singletons=F)
    all_T_results = []

    for T in T_SWEEP:
        lbl_t  = fcluster(Z_link, t=T, criterion="distance")
        sc_t   = lbl_t[:, None] == lbl_t[None, :]
        tp     = int(( sc_t &  same_truth & upper_tri).sum())
        fp     = int(( sc_t & ~same_truth & upper_tri).sum())
        fn     = int((~sc_t &  same_truth & upper_tri).sum())
        pr     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        nc     = len(np.unique(lbl_t))
        sg     = int((pd.Series(lbl_t).value_counts() == 1).sum())
        mf     = nc - sg

        all_T_results.append(dict(
            T=T, precision=pr, recall=rc,
            multi_fund=mf, singletons=sg, n_clusters=nc,
        ))

        # Accept if precision >= target AND more multi-fund clusters than current best
        if pr >= PRECISION_TARGET and mf > best["multi_fund"]:
            ari = adjusted_rand_score(true_idx, lbl_t)
            nmi = normalized_mutual_info_score(true_idx, lbl_t)
            best = dict(
                T=T, multi_fund=mf, precision=pr, recall=rc,
                ari=ari, nmi=nmi, n_clusters=nc, singletons=sg,
            )

    # If no T hit the target, take the T with highest precision as fallback
    if best["T"] is None:
        best_pr_row = max(all_T_results, key=lambda r: (r["precision"], r["multi_fund"]))
        T   = best_pr_row["T"]
        lbl_t = fcluster(Z_link, t=T, criterion="distance")
        best = dict(
            T=T,
            multi_fund=best_pr_row["multi_fund"],
            precision=best_pr_row["precision"],
            recall=best_pr_row["recall"],
            ari=adjusted_rand_score(true_idx, lbl_t),
            nmi=normalized_mutual_info_score(true_idx, lbl_t),
            n_clusters=best_pr_row["n_clusters"],
            singletons=best_pr_row["singletons"],
        )

    best["all_T_results"] = all_T_results
    return best


# ── Composite scoring ─────────────────────────────────────────────────────────
def composite_score(m):
    """
    PRIMARY GOAL  : minimise singletons
    HARD CONSTRAINT: precision >= PRECISION_TARGET (default 0.90)

    Score design
    ────────────
    If precision >= target (constraint satisfied):
        score = -singletons
        → Optuna maximises this, so it minimises singletons directly.
        → No multi_fund term needed: fewer singletons ≡ more merged clusters.
        → A 1-singleton improvement is always worth the same regardless of
          where you are in the search space (no scaling ambiguity).

    If precision < target (constraint NOT satisfied):
        score = precision * 10 - 1000
        → Hard floor well below any valid trial (valid scores are 0 .. -500,
          invalid scores are -991 .. -1000).
        → Still guides the optimizer toward higher precision rather than
          returning a flat constant.

    Why not use multi_fund instead of -singletons?
        multi_fund = n_clusters - singletons.  n_clusters varies across trials
        (different K, T, c_conf produce different total cluster counts), so
        maximising multi_fund conflates "fewer singletons" with "more total
        clusters", which are not the same objective.  -singletons is unambiguous.
    """
    pr = m["precision"]
    sg = m["singletons"]

    if pr >= PRECISION_TARGET:
        return -sg                          # maximise → minimise singletons
    else:
        return pr * 10 - 1000              # hard floor; guides toward precision


# ── Module-level cache: passes T-sweep detail from objective → callback ───────
# objective() stores all_T_results here; callback reads and clears it.
_t_sweep_cache: dict = {}


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    # Search space narrowed based on v2 results:
    #   K=30 was best → [20, 60] covers ±2 domain-width
    #   alpha=0.03 was best → [0.01, 0.15] log-scale
    #   c_conf=2.0 was best → [0.25, 3.0] (avoid 0.0: no uncertainty penalty)
    K      = trial.suggest_int(  "K",      20,   60, step=5)
    alpha  = trial.suggest_float("alpha",  0.01, 0.15, log=True)
    c_conf = trial.suggest_float("c_conf", 0.25, 3.0,  step=0.25)

    m     = evaluate(K, alpha, c_conf)
    score = composite_score(m)

    # Store metrics as user attrs (persisted in SQLite — survives resume)
    trial.set_user_attr("best_T",       m["T"])
    trial.set_user_attr("precision",    m["precision"])
    trial.set_user_attr("recall",       m["recall"])
    trial.set_user_attr("ari",          m["ari"])
    trial.set_user_attr("nmi",          m["nmi"])
    trial.set_user_attr("n_clusters",   m["n_clusters"])
    trial.set_user_attr("singletons",   m["singletons"])
    trial.set_user_attr("multi_fund",   m["multi_fund"])
    trial.set_user_attr("score",        score)
    trial.set_user_attr("precision_target_met",
                        m["precision"] >= PRECISION_TARGET)

    # Stash T-sweep detail for the callback (cleared after callback writes it)
    _t_sweep_cache[trial.number] = {
        "K": K, "alpha": alpha, "c_conf": c_conf,
        "all_T_results": m["all_T_results"],
    }

    return score


# ── CSV-writing callback (fires after EVERY trial, even on exception) ─────────
# Using a callback instead of writing inside objective() guarantees the row
# is always flushed to disk — objective exceptions / KeyboardInterrupt won't
# leave partial or missing rows.
def csv_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Write one row to trial_log.csv and 15 rows to trial_t_sweep.csv."""

    # ── 1. Summary row → trial_log.csv ───────────────────────────────────────
    ua = trial.user_attrs
    summary_row = {
        "trial":                trial.number,
        "state":                trial.state.name,
        "score":                trial.value if trial.value is not None else float("nan"),
        # Tuned params
        "K":                    trial.params.get("K"),
        "alpha":                trial.params.get("alpha"),
        "c_conf":               trial.params.get("c_conf"),
        # Fixed params (recorded for completeness)
        "beta":                 BETA,
        "n_pseudo":             N_PSEUDO,
        "burn_in":              BURN_IN,
        "n_samples":            N_SAMPLES,
        "linkage":              LINKAGE,
        # Best-T metrics
        "best_T":               ua.get("best_T"),
        "precision":            ua.get("precision"),
        "recall":               ua.get("recall"),
        "ari":                  ua.get("ari"),
        "nmi":                  ua.get("nmi"),
        "n_clusters":           ua.get("n_clusters"),
        "singletons":           ua.get("singletons"),
        "multi_fund":           ua.get("multi_fund"),
        "precision_target_met": ua.get("precision_target_met"),
        # Running best at this point in the study
        "cumulative_best_score": study.best_value
            if study.best_trial is not None else float("nan"),
        "wall_time_s":          trial.duration.total_seconds()
            if trial.duration is not None else float("nan"),
    }
    pd.DataFrame([summary_row]).to_csv(
        ITER_CSV, mode="a", header=not os.path.exists(ITER_CSV), index=False,
    )

    # ── 2. T-sweep detail → trial_t_sweep.csv ────────────────────────────────
    cache = _t_sweep_cache.pop(trial.number, None)   # clear after use
    if cache is not None:
        sweep_rows = []
        for row in cache["all_T_results"]:
            sweep_rows.append({
                "trial":     trial.number,
                "K":         cache["K"],
                "alpha":     cache["alpha"],
                "c_conf":    cache["c_conf"],
                "T":         row["T"],
                "precision": row["precision"],
                "recall":    row["recall"],
                "multi_fund":row["multi_fund"],
                "singletons":row["singletons"],
                "n_clusters":row["n_clusters"],
                "target_met":row["precision"] >= PRECISION_TARGET,
            })
        pd.DataFrame(sweep_rows).to_csv(
            T_SWEEP_CSV, mode="a",
            header=not os.path.exists(T_SWEEP_CSV),
            index=False,
        )

    # ── 3. Pickle every 5 trials (belt-and-suspenders alongside SQLite) ───────
    if trial.number % 5 == 0:
        try:
            with open(PICKLE_PATH, "wb") as f:
                pickle.dump(study, f)
        except Exception:
            pass   # never let pickle failure abort the study


# ── Optuna study — load or create ─────────────────────────────────────────────
storage_url = f"sqlite:///{DB_PATH}"

is_resuming = os.path.exists(DB_PATH)

study = optuna.create_study(
    study_name      = "lda-chunked-v3",
    direction       = "maximize",
    sampler         = optuna.samplers.TPESampler(seed=42),
    storage         = storage_url,
    load_if_exists  = True,   # ← resumes automatically if DB exists
)

completed = len([t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])

print(f"\nPrecision target : {PRECISION_TARGET}")
print(f"Session trials   : {args.trials}")
print(f"Already completed: {completed} (resuming = {is_resuming})")
print(f"Study DB         : {DB_PATH}")
print(f"\nSearch space:")
print(f"  K        : int  [20 … 60, step 5]")
print(f"  alpha    : float [0.01 … 0.15, log-scale]")
print(f"  c_conf   : float [0.25 … 3.00, step 0.25]")
print(f"  T_THRESH : swept internally [0.02 … 0.30] — NOT a Bayesian param")


# ── Run ───────────────────────────────────────────────────────────────────────
if args.trials > 0:
    print(f"\nRunning {args.trials} new trial(s)...\n")
    t_start = time.time()

    try:
        study.optimize(objective, n_trials=args.trials,
                       callbacks=[csv_callback],
                       show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nInterrupted — saving progress.")

    elapsed = time.time() - t_start
    new_completed = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nSession complete: {new_completed - completed} trials in "
          f"{elapsed / 60:.1f} min  |  Total so far: {new_completed}")

    # Save pickle snapshot alongside SQLite (belt-and-suspenders)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(study, f)
    print(f"Pickle snapshot saved → {PICKLE_PATH}")


# ── Report ────────────────────────────────────────────────────────────────────
all_done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not all_done:
    print("\nNo completed trials yet.")
    sys.exit(0)

best = study.best_trial

print("\n" + "=" * 65)
print(f"  BEST TRIAL  (out of {len(all_done)} total)")
print("=" * 65)
print(f"  Score (= -singletons)   : {best.value:.0f}  "
      f"→ singletons = {-best.value:.0f}")
print(f"  Precision           : {best.user_attrs['precision']:.4f}  "
      f"(target ≥ {PRECISION_TARGET})")
print(f"  Recall              : {best.user_attrs['recall']:.4f}")
print(f"  ARI                 : {best.user_attrs['ari']:.4f}")
print(f"  NMI                 : {best.user_attrs['nmi']:.4f}")
print(f"  Total Clusters      : {best.user_attrs['n_clusters']:.0f}")
print(f"  Multi-Fund Clusters : {best.user_attrs['multi_fund']:.0f}")
print(f"  Singletons          : {best.user_attrs['singletons']:.0f}")
print(f"  Optimal T_THRESH    : {best.user_attrs['best_T']:.2f}")

p = best.params
print("\n" + "=" * 65)
print("  COPY THESE INTO Lda_pipeline_multiple.py")
print("=" * 65)
print(f"  K         = {p['K']}")
print(f"  ALPHA     = {p['alpha']:.6f}")
print(f"  BETA      = {BETA}")
print(f"  N_PSEUDO  = {N_PSEUDO}")
print(f"  BURN_IN   = 500        # use full burn-in for production")
print(f"  N_SAMPLES = 100        # use full samples for production")
print(f"  C_CONF    = {p['c_conf']:.2f}")
print(f"  T_THRESH  = {best.user_attrs['best_T']:.2f}")
print("=" * 65)

# Save best params JSON
best_out = {
    "K":        p["K"],
    "ALPHA":    round(p["alpha"], 6),
    "BETA":     BETA,
    "N_PSEUDO": N_PSEUDO,
    "C_CONF":   p["c_conf"],
    "T_THRESH": best.user_attrs["best_T"],
    "LINKAGE":  LINKAGE,
    "metrics": {k: float(v) for k, v in best.user_attrs.items()
                if k != "all_T_results"},
    "total_trials_completed": len(all_done),
}
with open(BEST_JSON, "w") as f:
    json.dump(best_out, f, indent=4)

# Save full trials table
study.trials_dataframe().to_csv(
    os.path.join(OUT_DIR, "optuna_all_trials.csv"), index=False
)

# ── Summary table: all trials meeting precision target ────────────────────────
if os.path.exists(ITER_CSV):
    log = pd.read_csv(ITER_CSV)
    met = log[log["precision_target_met"] == True].copy()

    print(f"\n  Trials meeting precision ≥ {PRECISION_TARGET}: {len(met)} / {len(log)}")
    if not met.empty:
        top = met.nsmallest(10, "singletons")[
            ["trial", "K", "alpha", "c_conf", "best_T",
             "precision", "recall", "multi_fund", "singletons"]
        ].reset_index(drop=True)
        print("\n  Top 10 by fewest singletons (precision target met):")
        print(top.to_string(index=False))
    else:
        # No trial met target yet — show top by precision
        print("\n  No trial met the target yet. Top 10 by precision so far:")
        top_pr = log.nlargest(10, "precision")[
            ["trial", "K", "alpha", "c_conf", "best_T",
             "precision", "recall", "multi_fund", "singletons"]
        ].reset_index(drop=True)
        print(top_pr.to_string(index=False))
        print(f"\n  Tip: lower --precision-target to e.g. 0.92 if 0.95 is never met,")
        print( "  then tighten once you find a promising region.")

print(f"\n  Saved: {BEST_JSON}")
print(f"  Saved: {ITER_CSV}  ← one row per trial")
print(f"  Saved: {T_SWEEP_CSV}  ← all 15 T values per trial")
print(f"  DB   : {DB_PATH}  (auto-resumes next run)")

# ── Progress across sessions ──────────────────────────────────────────────────
if os.path.exists(ITER_CSV):
    log = pd.read_csv(ITER_CSV)
    sessions_msg = (
        f"\n  Total trials so far: {len(log)}"
        f"  |  Next chunk: python lda_hyperopt_tuning_v3.py --trials {args.trials}"
    )
    print(sessions_msg)
