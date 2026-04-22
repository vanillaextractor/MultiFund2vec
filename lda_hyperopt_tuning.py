"""
Fast Hyperparameter Tuning for LDA Mutual Fund Similarity Pipeline
Uses Optuna (Bayesian Optimization / TPE) to efficiently search the parameter space.

Optimizes for:
1. High Precision (>= 0.95)
2. Maximum number of clusters (avoiding singletons)

Usage:
  .venv/bin/python lda_hyperopt_tuning.py
"""

import numpy as np
import pandas as pd
import lda as lda_pkg
import time
import warnings
import logging
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import optuna
import os

warnings.filterwarnings("ignore")
logging.getLogger("lda").setLevel(logging.ERROR)

DATA_PATH = "ne04j_with_labels.csv"

print(f"Loading data from {DATA_PATH}...")
try:
    raw = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Please ensure the data file exists.")
    exit(1)

NOISE_COLS = ["Num_Assets", "Risky_Proportion"]
raw_filtered = raw[~raw["asset"].isin(NOISE_COLS)].copy()

wm = raw_filtered.pivot_table(
    index="fund_name", columns="asset", values="weight",
    aggfunc="sum", fill_value=0.0
)
wm = wm.div(wm.sum(axis=1), axis=0)

funds = wm.index.tolist()
assets = wm.columns.tolist()
F, A = len(funds), len(assets)
W = wm.values

print(f"Loaded {F} funds and {A} assets.")

# Ground truth derivation
NOISE_LABELS = {'Num_Assets', 'Risky_Proportion', 'Risk_Free'}
label_weight_map = {}
for _, row in raw.iterrows():
    if row['label'] in NOISE_LABELS:
        continue
    fund = row['fund_name']
    lbl = row['label']
    if fund not in label_weight_map:
        label_weight_map[fund] = {}
    label_weight_map[fund][lbl] = label_weight_map[fund].get(lbl, 0.0) + row['weight']

ground_truth_labels = []
for fn in funds:
    lmap = label_weight_map.get(fn, {})
    dominant = max(lmap, key=lmap.get) if lmap else 'Unknown'
    ground_truth_labels.append(dominant)

true_str = np.array(ground_truth_labels)
valid_secs = sorted(set(true_str))
sec_to_idx = {s: i for i, s in enumerate(valid_secs)}
true_idx = np.array([sec_to_idx[lbl] for lbl in true_str])

same_truth = true_idx[:, None] == true_idx[None, :]
upper_tri = np.triu(np.ones((F, F), dtype=bool), k=1)


def jsd_vectorised(Pi3, Qall):
    """
    Vectorised Jensen-Shannon Divergence (base-2 log, result in [0, 1]).
    Pi3  : (S, 1,     K)  — samples for one fund, broadcast-ready
    Qall : (S, F-i-1, K)  — samples for all remaining funds
    """
    M = 0.5 * (Pi3 + Qall)
    eps = 1e-12
    kl_p = np.where(Pi3 > eps, Pi3 * np.log2(np.maximum(Pi3, eps) / np.maximum(M, eps)), 0.0).sum(-1)
    kl_q = np.where(Qall > eps, Qall * np.log2(np.maximum(Qall, eps) / np.maximum(M, eps)), 0.0).sum(-1)
    jv = np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)
    return jv.mean(axis=0), jv.std(axis=0)


def evaluate_pipeline(K, alpha, beta, n_pseudo, c_conf, t_thresh, linkage_method="complete", burn_in=200, n_samples=15):
    """
    Runs the full LDA pipeline given the hyperparameters.
    burn_in and n_samples are kept low during tuning for speed.
    """
    # Step 2: Pseudo counts
    C = np.round(W * n_pseudo).astype(np.int32)
    
    # Step 4 & 5: Gibbs Sampling
    theta_samples = []
    
    for s in range(n_samples):
        model = lda_pkg.LDA(
            n_topics=K,
            n_iter=burn_in,
            alpha=alpha,
            eta=beta,
            random_state=42 + s * 17,
            refresh=burn_in
        )
        model.fit(C)
        
        n_dk = model.ndz_.astype(np.float64)
        theta_s = n_dk + alpha
        theta_s /= theta_s.sum(axis=1, keepdims=True)
        theta_samples.append(theta_s)
        
    theta_arr = np.stack(theta_samples, axis=0)
    
    # Step 6: Distances
    dist_m = np.zeros((F, F))
    dist_s = np.zeros((F, F))
    
    for i in range(F):
        Pi = theta_arr[:, i, :]
        Pi3 = Pi[:, None, :]
        Qall = theta_arr[:, i+1:, :]
        if Qall.shape[1] > 0:
            mij, sij = jsd_vectorised(Pi3, Qall)
            dist_m[i, i+1:] = dist_m[i+1:, i] = mij
            dist_s[i, i+1:] = dist_s[i+1:, i] = sij
            
    dist_c = dist_m + c_conf * dist_s
    np.fill_diagonal(dist_c, 0.0)
    
    # Step 7: Clustering
    condensed = squareform(dist_c, checks=False)
    Z_link = linkage(condensed, method=linkage_method)
    labels_pred = fcluster(Z_link, t=t_thresh, criterion="distance")
    
    # Evaluation
    same_cluster = labels_pred[:, None] == labels_pred[None, :]
    
    TP = int((same_cluster & same_truth & upper_tri).sum())
    FP = int((same_cluster & ~same_truth & upper_tri).sum())
    FN = int((~same_cluster & same_truth & upper_tri).sum())
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    ari = adjusted_rand_score(true_idx, labels_pred)
    nmi = normalized_mutual_info_score(true_idx, labels_pred)
    
    n_clusters = len(np.unique(labels_pred))
    size_series = pd.Series(labels_pred).value_counts()
    singletons = int((size_series == 1).sum())
    multi_fund_clusters = n_clusters - singletons
    
    # Threshold sweep to find best precision threshold
    sweep_best_t = t_thresh
    sweep_best_prec = precision
    for T_try in np.round(np.arange(0.02, 0.32, 0.02), 3):
        lbl_t   = fcluster(Z_link, t=T_try, criterion="distance")
        sc_t    = lbl_t[:, None] == lbl_t[None, :]
        tp  = int(( sc_t &  same_truth & upper_tri).sum())
        fp  = int(( sc_t & ~same_truth & upper_tri).sum())
        pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if pr > sweep_best_prec:
            sweep_best_prec = pr
            sweep_best_t = T_try
            
    return precision, recall, ari, nmi, multi_fund_clusters, singletons, n_clusters, sweep_best_t, sweep_best_prec


def objective(trial):
    # Suggest hyperparameters
    K = trial.suggest_int('K', 20, 100, step=10)
    alpha = trial.suggest_float('alpha', 0.01, 1.0, log=True)
    beta = trial.suggest_float('beta', 0.005, 0.2, log=True)
    n_pseudo = trial.suggest_int('n_pseudo', 200, 1500, step=100)
    c_conf = trial.suggest_float('c_conf', 0.0, 3.0, step=0.25)
    t_thresh = trial.suggest_float('t_thresh', 0.05, 0.40, step=0.02)
    linkage_method = trial.suggest_categorical('linkage_method', ['complete', 'average', 'single'])
    
    # Fast settings for tuning to keep it tractable
    burn_in = 200
    n_samples = 15
    
    precision, recall, ari, nmi, multi_fund_clusters, singletons, n_clusters, sweep_best_t, sweep_best_prec = evaluate_pipeline(
        K, alpha, beta, n_pseudo, c_conf, t_thresh, linkage_method, burn_in, n_samples
    )
    
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("ari", ari)
    trial.set_user_attr("nmi", nmi)
    trial.set_user_attr("multi_fund_clusters", multi_fund_clusters)
    trial.set_user_attr("singletons", singletons)
    trial.set_user_attr("n_clusters", n_clusters)
    
    # Composite Score Design:
    # 1. We strongly desire precision >= 0.95. If lower, heavily penalize it.
    # 2. Once precision is acceptable, we want to maximize the number of multi-fund clusters
    #    and minimize singletons.
    
    if precision < 0.90:
        # If precision is bad, just guide the optimizer to improve precision
        score = precision * 10  # Max score here is < 9.0
    else:
        # If precision is good (>=0.90), we boost the score and add the clustering metrics
        # precision adds up to ~10
        # multi_fund_clusters adds up to ~number of clusters
        # singletons acts as a penalty
        penalty_for_singletons = singletons / max(1, F) * 10  # Scale 0 to 10
        
        # We give a bonus for being over 0.95 precision
        precision_bonus = 20 if precision >= 0.95 else 0
        
        score = 50 + precision_bonus + (precision * 10) + multi_fund_clusters - penalty_for_singletons
        
    # Write custom row to CSV for this iteration
    os.makedirs("output/hypertuned_best", exist_ok=True)
    custom_csv = "output/hypertuned_best/custom_iteration_results.csv"
    row = {
        "Trial": trial.number,
        "K": K,
        "Alpha": alpha,
        "Beta": beta,
        "N_Pseudo": n_pseudo,
        "Burn_In": burn_in,
        "N_Samples": n_samples,
        "C_Conf": c_conf,
        "T_Thresh": t_thresh,
        "Total_Clusters": n_clusters,
        "Singletons": singletons,
        "Multi_Fund_Clusters": multi_fund_clusters,
        "Precision": precision,
        "Recall": recall,
        "ARI": ari,
        "NMI": nmi,
        "Best_Precision_T": sweep_best_t,
        "Best_Precision": sweep_best_prec,
        "Score": score
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(custom_csv, mode='a', header=not os.path.exists(custom_csv), index=False)
        
    return score


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LDA HYPERPARAMETER TUNING (OPTUNA BAYESIAN SEARCH)  ")
    print("="*60)
    print("This script uses Bayesian Optimization to find the optimal")
    print("set of hyperparameters for the LDA pipeline.")
    print("Objective: Maximize multi-fund clusters while keeping precision >= 0.95")
    print("\nUsing fast settings for evaluation: burn_in=200, n_samples=15.")
    
    # We use a TPE (Tree-structured Parzen Estimator) sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lda-tune")
    
    # Run optimization
    n_trials = 50  # Adjust as needed
    print(f"\nStarting optimization for {n_trials} trials...")
    
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Showing best results so far...")
    
    print("\n" + "="*60)
    print("  OPTIMIZATION FINISHED  ")
    print("="*60)
    
    if len(study.trials) > 0:
        best = study.best_trial
        
        print("\nBest Trial Overview:")
        print(f"  Composite Score     : {best.value:.4f}")
        print(f"  Precision           : {best.user_attrs.get('precision', 0):.4f}")
        print(f"  Total Clusters      : {best.user_attrs.get('n_clusters', 0)}")
        print(f"  Multi-Fund Clusters : {best.user_attrs.get('multi_fund_clusters', 0)}")
        print(f"  Singletons          : {best.user_attrs.get('singletons', 0)}")
        
        print("\nBest Hyperparameters to update in Lda_pipeline_multiple.py:")
        print("-" * 50)
        params = best.params
        print(f"K         = {params.get('K')}")
        print(f"ALPHA     = {params.get('alpha'):.6f}")
        print(f"BETA      = {params.get('beta'):.6f}")
        print(f"N_PSEUDO  = {params.get('n_pseudo')}")
        print(f"C_CONF    = {params.get('c_conf'):.2f}")
        print(f"T_THRESH  = {params.get('t_thresh'):.2f}")
        print(f"linkage   = '{params.get('linkage_method')}' (update linkage(..., method=...))")
        print("-" * 50)
        
        # Save results to CSV for analysis
        os.makedirs("output/hypertuned_best", exist_ok=True)
        
        # Save best parameters to a JSON file
        import json
        with open("output/hypertuned_best/best_params.json", "w") as f:
            json.dump(best.params, f, indent=4)
            
        df_trials = study.trials_dataframe()
        out_file = "output/hypertuned_best/optuna_trials_results.csv"
        df_trials.to_csv(out_file, index=False)
        print(f"\nFull trial results saved to {out_file}")
        print("Best parameters saved to output/hypertuned_best/best_params.json")
    else:
        print("No completed trials found.")
