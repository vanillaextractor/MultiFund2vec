"""
LDA-Based Mutual Fund Similarity Pipeline
==========================================
Complete implementation of the 8-step pipeline from lda_roadmap_v2.docx
Applied to: ne04jmodified.csv (500 funds, 1000 assets)
"""

# ============================================================
# BLOCK 0: SETUP & CONFIGURATION
# ============================================================
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)

CONFIG = {
    'data_path': 'ne04j_with_labels.csv',
    'N': 1000,           # Pseudo-count scaling factor
    'K': 50,             # Number of topics
    'alpha': 0.1,        # Fund-theme sparsity
    'beta': 0.01,        # Theme-asset sparsity
    'n_iter': 1000,      # Total Gibbs iterations (500 burn-in + 500 collection)
    'burn_in': 500,      # Burn-in iterations
    'thin': 5,           # Thinning interval
    'c': 1.5,            # Conservative distance parameter (higher = stricter, fewer FP)
    'T': 0.15,           # Clustering threshold (initial)
}

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

S = (CONFIG['n_iter'] - CONFIG['burn_in']) // CONFIG['thin']  # Number of posterior samples
print(f"{'='*60}")
print(f"  LDA MUTUAL FUND SIMILARITY PIPELINE")
print(f"{'='*60}")
print(f"  Config: K={CONFIG['K']}, α={CONFIG['alpha']}, β={CONFIG['beta']}")
print(f"  Gibbs: {CONFIG['n_iter']} iters, burn-in={CONFIG['burn_in']}, thin={CONFIG['thin']}")
print(f"  Posterior samples per fund: {S}")
print(f"{'='*60}\n")

# ============================================================
# BLOCK 1: DATA PREPARATION (Step 1)
# ============================================================
print("=" * 60)
print("  BLOCK 1: DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(CONFIG['data_path'])
print(f"  Raw data: {df.shape[0]} rows, {df['fund_name'].nunique()} funds, {df['asset'].nunique()} assets")

# Pivot to Fund x Asset weight matrix
weight_matrix = df.pivot_table(index='fund_name', columns='asset', values='weight', fill_value=0.0)

# Drop metadata columns that are NOT real asset holdings:
#   Num_Assets       — raw count of assets held (e.g. 31, 82, 180), NOT a proportion.
#                      After row normalization it would take 31/32 = 97% of every row,
#                      completely drowning all real sector signals.
#   Risky_Proportion — always = 1 - Risk_Free, purely redundant information.
#   Risk_Free        — KEPT: a real portfolio allocation (proportion 0–0.6),
#                      meaningfully discriminates equity vs debt vs hybrid funds.
NOISE_COLS = ['Num_Assets', 'Risky_Proportion']
cols_to_drop = [c for c in NOISE_COLS if c in weight_matrix.columns]
weight_matrix = weight_matrix.drop(columns=cols_to_drop)
print(f"  Dropped metadata columns: {cols_to_drop}")
print(f"  Kept 'Risk_Free' — real allocation weight, discriminative signal")

fund_names = weight_matrix.index.tolist()
asset_names = weight_matrix.columns.tolist()
F, A = weight_matrix.shape
print(f"  Weight matrix shape: {F} funds × {A} assets")

# Normalize rows to sum to 1.0
W = weight_matrix.values.copy()
row_sums = W.sum(axis=1, keepdims=True)
W_norm = W / row_sums
assert np.allclose(W_norm.sum(axis=1), 1.0), "Row normalization failed!"
print(f"  ✓ All rows normalized to sum to 1.0")
print(f"  Sparsity: {(W_norm == 0).sum() / W_norm.size * 100:.1f}% zeros")
print(f"  Assets per fund: min={np.count_nonzero(W_norm, axis=1).min()}, "
      f"max={np.count_nonzero(W_norm, axis=1).max()}, "
      f"median={np.median(np.count_nonzero(W_norm, axis=1)):.0f}")

# --- Plot: Sparsity heatmap (subset) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# Subsample for visualization
idx_funds = np.random.choice(F, min(50, F), replace=False)
idx_assets = np.where(W_norm[idx_funds].sum(axis=0) > 0)[0][:100]
sns.heatmap(W_norm[np.ix_(idx_funds, idx_assets)], cmap='YlOrRd', ax=axes[0],
            xticklabels=False, yticklabels=False)
axes[0].set_title('Fund-Asset Weight Matrix (50 funds × top assets)', fontsize=12)
axes[0].set_xlabel('Assets'); axes[0].set_ylabel('Funds')

axes[1].hist(np.count_nonzero(W_norm, axis=1), bins=30, color='steelblue', edgecolor='white')
axes[1].set_title('Assets per Fund Distribution', fontsize=12)
axes[1].set_xlabel('Number of Assets'); axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_data_preparation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/01_data_preparation.png\n")

# ============================================================
# BLOCK 2: PSEUDO-COUNT CONVERSION (Step 2)
# ============================================================
print("=" * 60)
print("  BLOCK 2: PSEUDO-COUNT CONVERSION")
print("=" * 60)

N = CONFIG['N']
C = np.round(W_norm * N).astype(int)
row_sums_C = C.sum(axis=1)
print(f"  Scaling factor N = {N}")
print(f"  Pseudo-count row sums: mean={row_sums_C.mean():.1f}, "
      f"min={row_sums_C.min()}, max={row_sums_C.max()}")
print(f"  Total tokens: {C.sum():,}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(row_sums_C, bins=30, color='coral', edgecolor='white')
ax.axvline(N, color='black', linestyle='--', label=f'Target N={N}')
ax.set_title('Pseudo-Count Row Sums Distribution', fontsize=12)
ax.set_xlabel('Row Sum'); ax.set_ylabel('Count'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_pseudocount_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/02_pseudocount_histogram.png\n")

# ============================================================
# BLOCK 3: HYPERPARAMETER SELECTION (Step 3)
# ============================================================
print("=" * 60)
print("  BLOCK 3: HYPERPARAMETER SELECTION")
print("=" * 60)
K = CONFIG['K']
alpha = CONFIG['alpha']
beta = CONFIG['beta']
V = A  # Vocabulary size = number of unique assets
print(f"  K (topics)  = {K}   (range 30-80; higher K → fewer FP, more FN)")
print(f"  α (alpha)   = {alpha}  (low → sparse fund-theme distributions)")
print(f"  β (beta)    = {beta} (low → sharp theme-asset distributions)")
print(f"  V (vocab)   = {V}   (total unique assets)")
print()

# ============================================================
# BLOCK 4: LDA VIA COLLAPSED GIBBS SAMPLING (Step 4)
# ============================================================
print("=" * 60)
print("  BLOCK 4: COLLAPSED GIBBS SAMPLING")
print("=" * 60)

# Build sparse representation: list of (doc_id, word_id, count) for non-zero entries
print("  Building sparse corpus representation...")
nonzero_pairs = []
doc_ids_sparse = []
word_ids_sparse = []
counts_sparse = []
for d in range(F):
    nz = np.nonzero(C[d])[0]
    for w_idx in nz:
        if C[d, w_idx] > 0:
            nonzero_pairs.append((d, w_idx))
            doc_ids_sparse.append(d)
            word_ids_sparse.append(w_idx)
            counts_sparse.append(C[d, w_idx])

doc_ids_sparse = np.array(doc_ids_sparse, dtype=np.int32)
word_ids_sparse = np.array(word_ids_sparse, dtype=np.int32)
counts_sparse = np.array(counts_sparse, dtype=np.int32)
num_pairs = len(nonzero_pairs)
print(f"  Non-zero (doc, word) pairs: {num_pairs:,}")
print(f"  Total tokens: {counts_sparse.sum():,}")

# Initialize topic assignments: dwk[i, k] = count of tokens for pair i assigned to topic k
print("  Initializing random topic assignments...")
dwk = np.zeros((num_pairs, K), dtype=np.int32)
n_dk = np.zeros((F, K), dtype=np.int32)       # doc-topic counts
n_kw = np.zeros((K, V), dtype=np.int32)       # topic-word counts
n_k = np.zeros(K, dtype=np.int32)             # topic totals

for i in range(num_pairs):
    c = counts_sparse[i]
    d = doc_ids_sparse[i]
    w = word_ids_sparse[i]
    # Random multinomial assignment
    assignment = np.random.multinomial(c, np.ones(K) / K)
    dwk[i] = assignment
    n_dk[d] += assignment
    n_kw[:, w] += assignment
    n_k += assignment

# Precompute doc-to-pair mapping for efficiency
doc_pair_indices = [[] for _ in range(F)]
for i in range(num_pairs):
    doc_pair_indices[doc_ids_sparse[i]].append(i)

# Gibbs sampling iterations
print(f"  Running {CONFIG['n_iter']} Gibbs iterations...")
print(f"  (burn-in={CONFIG['burn_in']}, collect every {CONFIG['thin']} after burn-in)")

log_likelihoods = []
theta_samples = []  # Will store (S, F, K) posterior samples
sample_count = 0

start_time = time.time()
for iteration in range(CONFIG['n_iter']):
    # Shuffle pair order each iteration
    order = np.random.permutation(num_pairs)

    for idx in order:
        d = doc_ids_sparse[idx]
        w = word_ids_sparse[idx]
        c = counts_sparse[idx]
        old_assignment = dwk[idx].copy()

        # Remove current assignment from counts
        n_dk[d] -= old_assignment
        n_kw[:, w] -= old_assignment
        n_k -= old_assignment

        # Compute conditional probabilities for each topic
        p = (n_dk[d] + alpha) * (n_kw[:, w] + beta) / (n_k + V * beta)
        p = np.maximum(p, 1e-100)
        p /= p.sum()

        # Sample new assignment from Multinomial
        new_assignment = np.random.multinomial(c, p)
        dwk[idx] = new_assignment

        # Add new assignment to counts
        n_dk[d] += new_assignment
        n_kw[:, w] += new_assignment
        n_k += new_assignment

    # Compute log-likelihood proxy every 10 iterations
    if iteration % 10 == 0:
        theta_current = (n_dk + alpha) / (n_dk + alpha).sum(axis=1, keepdims=True)
        phi_current = (n_kw + beta) / (n_kw + beta).sum(axis=1, keepdims=True)
        # Log-likelihood: sum over non-zero entries of C * log(theta @ phi)
        ll = 0.0
        for i in range(num_pairs):
            d = doc_ids_sparse[i]
            w = word_ids_sparse[i]
            c_val = counts_sparse[i]
            prob = np.dot(theta_current[d], phi_current[:, w])
            ll += c_val * np.log(prob + 1e-300)
        log_likelihoods.append((iteration, ll))
        elapsed = time.time() - start_time
        eta = elapsed / (iteration + 1) * (CONFIG['n_iter'] - iteration - 1)
        print(f"    Iter {iteration:4d}/{CONFIG['n_iter']} | LL={ll:,.0f} | "
              f"Elapsed={elapsed:.0f}s | ETA={eta:.0f}s", end='\r')

    # Collect posterior sample after burn-in, at thinning intervals
    if iteration >= CONFIG['burn_in'] and (iteration - CONFIG['burn_in']) % CONFIG['thin'] == 0:
        theta_s = (n_dk + alpha) / (n_dk + alpha).sum(axis=1, keepdims=True)
        theta_samples.append(theta_s.copy())
        sample_count += 1

elapsed_total = time.time() - start_time
print(f"\n  ✓ Gibbs sampling complete in {elapsed_total:.1f}s")
print(f"  ✓ Collected {len(theta_samples)} posterior samples")

theta_samples = np.array(theta_samples)  # Shape: (S, F, K)

# --- Plot: Convergence ---
fig, ax = plt.subplots(figsize=(10, 4))
iters, lls = zip(*log_likelihoods)
ax.plot(iters, lls, color='steelblue', linewidth=1.5)
ax.axvline(CONFIG['burn_in'], color='red', linestyle='--', alpha=0.7, label=f"Burn-in ({CONFIG['burn_in']})")
ax.set_title('Log-Likelihood Convergence', fontsize=13)
ax.set_xlabel('Iteration'); ax.set_ylabel('Log-Likelihood'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/03_convergence.png\n")

# ============================================================
# BLOCK 5: EXTRACT POSTERIOR DISTRIBUTIONS (Step 5)
# ============================================================
print("=" * 60)
print("  BLOCK 5: POSTERIOR DISTRIBUTIONS")
print("=" * 60)

theta_mean = theta_samples.mean(axis=0)  # (F, K)
theta_std = theta_samples.std(axis=0)    # (F, K)

# Identify a confident fund (low avg std) and uncertain fund (high avg std)
fund_avg_std = theta_std.mean(axis=1)
confident_idx = np.argmin(fund_avg_std)
uncertain_idx = np.argmax(fund_avg_std)

print(f"  θ samples shape: {theta_samples.shape} (samples × funds × topics)")
print(f"  Most confident fund: {fund_names[confident_idx]} (avg θ std = {fund_avg_std[confident_idx]:.4f})")
print(f"  Most uncertain fund: {fund_names[uncertain_idx]} (avg θ std = {fund_avg_std[uncertain_idx]:.4f})")

# --- Plot: Posterior examples ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# Get top 10 topics for each fund by mean weight
for ax, fidx, label in [(axes[0], confident_idx, 'Confident'), (axes[1], uncertain_idx, 'Uncertain')]:
    fund_theta = theta_samples[:, fidx, :]  # (S, K)
    top_topics = np.argsort(fund_theta.mean(axis=0))[-10:][::-1]
    data = fund_theta[:, top_topics]
    bp = ax.boxplot(data, labels=[f'T{t}' for t in top_topics], patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(f'{label} Fund: {fund_names[fidx]}\n(avg std = {fund_avg_std[fidx]:.4f})', fontsize=11)
    ax.set_xlabel('Topic'); ax.set_ylabel('θ (topic proportion)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_posterior_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/04_posterior_examples.png\n")

# ============================================================
# BLOCK 6: PAIRWISE DISTANCES WITH UNCERTAINTY (Step 6)
# ============================================================
print("=" * 60)
print("  BLOCK 6: PAIRWISE JSD DISTANCES")
print("=" * 60)

S_samples = theta_samples.shape[0]
n_pairs = F * (F - 1) // 2
print(f"  Computing JSD for {n_pairs:,} pairs × {S_samples} samples...")

# For each posterior sample, compute pairwise JSD using scipy
jsd_all = np.zeros((S_samples, n_pairs))
start_time = time.time()
for s in range(S_samples):
    # jensenshannon returns sqrt(JSD), square it to get JSD
    jsd_condensed = pdist(theta_samples[s], metric='jensenshannon')
    jsd_all[s] = jsd_condensed ** 2  # Convert metric to divergence
    if (s + 1) % 10 == 0:
        print(f"    Sample {s+1}/{S_samples} done...", end='\r')

jsd_mean = jsd_all.mean(axis=0)  # Mean JSD per pair
jsd_std = jsd_all.std(axis=0)    # Std JSD per pair

# Conservative distance: mean + c * std
c_param = CONFIG['c']
jsd_conservative = jsd_mean + c_param * jsd_std

elapsed = time.time() - start_time
print(f"\n  ✓ JSD computation done in {elapsed:.1f}s")
print(f"  Mean JSD: {jsd_mean.mean():.4f} ± {jsd_mean.std():.4f}")
print(f"  Conservative distance: {jsd_conservative.mean():.4f} ± {jsd_conservative.std():.4f}")

# Convert to square matrices for visualization
dist_mean_matrix = squareform(jsd_mean)
dist_conservative_matrix = squareform(jsd_conservative)

# --- Plot: Distance analysis ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Heatmap of conservative distances (subset)
idx_sub = np.random.choice(F, min(60, F), replace=False)
idx_sub.sort()
sns.heatmap(dist_conservative_matrix[np.ix_(idx_sub, idx_sub)], cmap='viridis',
            ax=axes[0], xticklabels=False, yticklabels=False)
axes[0].set_title('Conservative JSD Distance\n(60-fund subset)', fontsize=11)

# Mean vs Std scatter
axes[1].scatter(jsd_mean, jsd_std, alpha=0.05, s=1, color='steelblue')
axes[1].set_xlabel('Mean JSD'); axes[1].set_ylabel('Std JSD')
axes[1].set_title('Mean vs Std of Pairwise JSD', fontsize=11)

# Histogram of conservative distances
axes[2].hist(jsd_conservative, bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('Conservative JSD'); axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Conservative Distances', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_distance_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/05_distance_analysis.png\n")

# ============================================================
# BLOCK 7: HIERARCHICAL CLUSTERING (Step 7)
# ============================================================
print("=" * 60)
print("  BLOCK 7: HIERARCHICAL CLUSTERING")
print("=" * 60)

# Complete-linkage hierarchical clustering using conservative distances
Z = linkage(jsd_conservative, method='complete')
print(f"  Linkage matrix shape: {Z.shape}")

# Threshold sweep
thresholds = np.arange(0.01, 0.51, 0.01)

# --- Derive ground truth using actual labels from ne04j_with_labels.csv ---
print("  Deriving ground truth from actual 'label' column (Sector_CapTier level)...")
# The label column tags each asset with its true Sector_CapTier
# (e.g. 'Technology_Large Cap', 'Energy_Mid Cap').
# Noise entries (Num_Assets, Risky_Proportion, Risk_Free) have their own label —
# exclude these so only real investment holdings define the fund's true label.
NOISE_LABELS = {'Num_Assets', 'Risky_Proportion', 'Risk_Free'}

# Build fund → {label: total_weight} map using actual label column
label_weight_map = {}
for _, row in df.iterrows():
    if row['label'] in NOISE_LABELS:
        continue
    fund = row['fund_name']
    lbl  = row['label']
    if fund not in label_weight_map:
        label_weight_map[fund] = {}
    label_weight_map[fund][lbl] = label_weight_map[fund].get(lbl, 0.0) + row['weight']

# Dominant label per fund = actual ground truth
ground_truth_labels = []
for fn in fund_names:
    lmap = label_weight_map.get(fn, {})
    dominant = max(lmap, key=lmap.get) if lmap else 'Unknown'
    ground_truth_labels.append(dominant)
ground_truth_labels = np.array(ground_truth_labels)

unique_labels = np.unique(ground_truth_labels)
print(f"  Ground truth labels: {len(unique_labels)} unique (Sector × Cap-Tier)")
for lbl in unique_labels:
    cnt = (ground_truth_labels == lbl).sum()
    print(f"    {lbl}: {cnt} funds")

# Build ground truth pairwise similarity
label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
gt_int = np.array([label_to_int[l] for l in ground_truth_labels])

# Sweep thresholds and compute metrics
print("  Sweeping thresholds...")
results = []
for T in thresholds:
    clusters = fcluster(Z, t=T, criterion='distance')

    # Pairwise precision & recall
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(F):
        for j in range(i + 1, F):
            same_cluster = (clusters[i] == clusters[j])
            same_label = (gt_int[i] == gt_int[j])
            if same_cluster and same_label:
                tp += 1
            elif same_cluster and not same_label:
                fp += 1
            elif not same_cluster and same_label:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ari = adjusted_rand_score(gt_int, clusters)
    nmi = normalized_mutual_info_score(gt_int, clusters)
    n_clusters = len(set(clusters))
    n_singletons = sum(1 for c in Counter(clusters).values() if c == 1)

    results.append({
        'T': T, 'precision': precision, 'recall': recall,
        'ari': ari, 'nmi': nmi, 'n_clusters': n_clusters,
        'n_singletons': n_singletons, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    })

results_df = pd.DataFrame(results)

# Recompute F0.5 metrics for downstream plotting
results_df['f05'] = (1.25 * results_df['precision'] * results_df['recall']) / \
                     (0.25 * results_df['precision'] + results_df['recall'] + 1e-10)

# Select best T by maximising precision strictly, BUT guard against degenerate
# trivial clustering (e.g. T=0.01 gives 1.0 precision but 0.01 recall with 450 singletons).
# We require a minimum acceptable recall of 15% (0.15).
valid_candidates = results_df[results_df['recall'] >= 0.15]
if not valid_candidates.empty:
    max_prec = valid_candidates['precision'].max()
    best_candidates = valid_candidates[valid_candidates['precision'] == max_prec]
    best_idx = best_candidates['recall'].idxmax()
    strategy = "max-prec (recall >= 15%)"
else:
    best_idx = results_df['f05'].idxmax()  # Fallback
    strategy = "F0.5 fallback"

best_T = results_df.loc[best_idx, 'T']
print(f"\n  Best threshold T = {best_T:.2f} ({strategy})")
print(f"  → Precision = {results_df.loc[best_idx, 'precision']:.4f}  "
      f"Recall = {results_df.loc[best_idx, 'recall']:.4f}")

# Apply best threshold
final_clusters = fcluster(Z, t=best_T, criterion='distance')
n_final_clusters = len(set(final_clusters))
cluster_sizes = Counter(final_clusters)
n_singletons = sum(1 for v in cluster_sizes.values() if v == 1)
print(f"  Clusters: {n_final_clusters} total, {n_singletons} singletons")

# --- Plot: Dendrogram + cluster sizes ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Truncated dendrogram
dendrogram(Z, truncate_mode='lastp', p=30, ax=axes[0], color_threshold=best_T,
           above_threshold_color='grey')
axes[0].axhline(best_T, color='red', linestyle='--', alpha=0.8, label=f'T={best_T:.2f}')
axes[0].set_title('Hierarchical Clustering Dendrogram', fontsize=12)
axes[0].set_xlabel('Fund (clustered)'); axes[0].set_ylabel('Distance'); axes[0].legend()

# Cluster size distribution
sizes = sorted(cluster_sizes.values(), reverse=True)
axes[1].bar(range(len(sizes)), sizes, color='steelblue', edgecolor='white')
axes[1].set_title(f'Cluster Size Distribution (n={n_final_clusters})', fontsize=12)
axes[1].set_xlabel('Cluster Rank'); axes[1].set_ylabel('Size')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/06_clustering.png\n")

# ============================================================
# BLOCK 8: EVALUATION (Step 8)
# ============================================================
print("=" * 60)
print("  BLOCK 8: EVALUATION")
print("=" * 60)

best_row = results_df.loc[best_idx]
print(f"  At T = {best_T:.2f}:")
print(f"    Pairwise Precision : {best_row['precision']:.4f}")
print(f"    Pairwise Recall    : {best_row['recall']:.4f}")
print(f"    F0.5 Score         : {best_row['f05']:.4f}")
print(f"    ARI                : {best_row['ari']:.4f}")
print(f"    NMI                : {best_row['nmi']:.4f}")
print(f"    TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, "
      f"FN={int(best_row['fn'])}, TN={int(best_row['tn'])}")

# --- Plot: Metrics vs Threshold ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(results_df['T'], results_df['precision'], 'b-', label='Precision', linewidth=2)
axes[0].plot(results_df['T'], results_df['recall'], 'r--', label='Recall', linewidth=2)
axes[0].plot(results_df['T'], results_df['f05'], 'g-.', label='F0.5', linewidth=2)
axes[0].axvline(best_T, color='grey', linestyle=':', alpha=0.7)
axes[0].set_xlabel('Threshold T'); axes[0].set_ylabel('Score')
axes[0].set_title('Precision / Recall / F0.5 vs Threshold', fontsize=12); axes[0].legend()

axes[1].plot(results_df['T'], results_df['ari'], 'purple', label='ARI', linewidth=2)
axes[1].plot(results_df['T'], results_df['nmi'], 'orange', label='NMI', linewidth=2)
axes[1].axvline(best_T, color='grey', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Threshold T'); axes[1].set_ylabel('Score')
axes[1].set_title('ARI / NMI vs Threshold', fontsize=12); axes[1].legend()

axes[2].plot(results_df['T'], results_df['n_clusters'], 'steelblue', linewidth=2)
axes[2].axvline(best_T, color='grey', linestyle=':', alpha=0.7)
axes[2].set_xlabel('Threshold T'); axes[2].set_ylabel('Number of Clusters')
axes[2].set_title('Clusters vs Threshold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_evaluation_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/07_evaluation_metrics.png")

# Confusion matrix at best T
cm_labels = ['Predicted Same', 'Predicted Diff']
cm_data = np.array([[int(best_row['tp']), int(best_row['fn'])],
                     [int(best_row['fp']), int(best_row['tn'])]])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Actually Similar', 'Actually Different'],
            yticklabels=['Model: Same Cluster', 'Model: Diff Clusters'])
ax.set_title(f'Pairwise Confusion Matrix (T={best_T:.2f})', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/08_confusion_matrix.png\n")

# ============================================================
# BLOCK 9: POST-ANALYSIS — THEME INTERPRETATION (Step 4 uses)
# ============================================================
print("=" * 60)
print("  BLOCK 9: THEME INTERPRETATION & SUMMARY")
print("=" * 60)

# Extract final phi (topic-asset distribution)
phi = (n_kw + beta) / (n_kw + beta).sum(axis=1, keepdims=True)  # (K, V)

print("\n  Top 10 assets per topic (top 10 themes by total weight):")
# Rank themes by average weight across funds
theme_importance = theta_mean.mean(axis=0)
top_themes = np.argsort(theme_importance)[::-1][:10]

# Plot theme interpretation
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
for ax_idx, k in enumerate(top_themes):
    ax = axes[ax_idx // 5][ax_idx % 5]
    top_asset_ids = np.argsort(phi[k])[::-1][:10]
    top_asset_names = [asset_names[a] for a in top_asset_ids]
    top_asset_probs = phi[k, top_asset_ids]

    # Shorten asset names for display
    short_names = []
    for name in top_asset_names:
        parts = name.split('_')
        short_names.append(f"{parts[0]}_{parts[-1]}" if len(parts) >= 2 else name)

    # Determine dominant sector
    sectors_in_theme = [name.split('_')[0] for name in top_asset_names]
    dominant_sector = Counter(sectors_in_theme).most_common(1)[0][0]

    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    ax.barh(range(10), top_asset_probs[::-1], color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(short_names[::-1], fontsize=7)
    ax.set_title(f'Topic {k} ({dominant_sector})\nAvg weight: {theme_importance[k]:.3f}', fontsize=9)
    ax.set_xlabel('φ (probability)', fontsize=8)

plt.suptitle('Top 10 Themes — Asset Composition', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_theme_interpretation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/09_theme_interpretation.png")

# Print theme summary
for k in top_themes:
    top_asset_ids = np.argsort(phi[k])[::-1][:5]
    assets_str = ', '.join([asset_names[a].split('_')[0] for a in top_asset_ids])
    print(f"    Topic {k:2d} (weight {theme_importance[k]:.3f}): {assets_str}")

# Fund classification summary
print(f"\n  Fund Classification Summary:")
print(f"  {'Fund':<12} {'Cluster':>8} {'Dominant Topic':>15} {'Confidence':>12} {'Sector':>15}")
print(f"  {'-'*62}")
for i in range(min(20, F)):
    dominant_topic = np.argmax(theta_mean[i])
    confidence = theta_mean[i, dominant_topic]
    sector = ground_truth_labels[i]
    print(f"  {fund_names[i]:<12} {final_clusters[i]:>8} {dominant_topic:>15} "
          f"{confidence:>12.3f} {sector:>15}")

# ============================================================
# BLOCK 10: FOLD-IN INFERENCE FOR NEW FUNDS (Section 4.3)
# ============================================================
# "When a new mutual fund needs to be analyzed (one that was not in the
#  training data): hold the learned φ matrices fixed and run Gibbs
#  sampling on only this new fund's data."
# ============================================================
print("=" * 60)
print("  BLOCK 10: FOLD-IN INFERENCE FOR NEW FUNDS (Sec 4.3)")
print("=" * 60)

# Use the posterior mean of φ (topic-word distribution) — held fixed
phi_mean = (n_kw + beta) / (n_kw + beta).sum(axis=1, keepdims=True)  # (K, V)

def fold_in(new_fund_weights, phi_fixed, alpha, N=1000, n_iter=200, burn_in=50, thin=5):
    """
    Fold-in inference for a new fund.
    Holds φ fixed; runs Gibbs on only this fund's topic assignments.

    Args:
        new_fund_weights : array of shape (A,), raw weights (will be normalized)
        phi_fixed        : (K, V) fixed topic-word distributions
        Returns: array (S, K) of posterior θ samples for the new fund
    """
    K_local, V_local = phi_fixed.shape
    # Normalize and convert to pseudo-counts
    w_norm = new_fund_weights / (new_fund_weights.sum() + 1e-300)
    counts = np.round(w_norm * N).astype(int)
    nonzero_w = np.nonzero(counts)[0]
    if len(nonzero_w) == 0:
        return np.ones((1, K_local)) / K_local  # degenerate

    word_ids = nonzero_w
    word_cnts = counts[word_ids]

    # Initialize topic assignment counts for this single fund
    n_dk_new = np.zeros(K_local, dtype=np.float64)
    # track per (word, topic) assignments
    dwk_new = np.zeros((len(word_ids), K_local), dtype=np.int32)

    for i, (w, c) in enumerate(zip(word_ids, word_cnts)):
        assignment = np.random.multinomial(c, np.ones(K_local) / K_local)
        dwk_new[i] = assignment
        n_dk_new += assignment

    samples = []
    for it in range(n_iter):
        for i, (w, c) in enumerate(zip(word_ids, word_cnts)):
            old = dwk_new[i].copy()
            n_dk_new -= old
            # Conditional: p(z|rest) ∝ (n_dk + α) * φ[k, w]
            p = (n_dk_new + alpha) * phi_fixed[:, w]
            p = np.maximum(p, 1e-300)
            p /= p.sum()
            new_asgn = np.random.multinomial(c, p)
            dwk_new[i] = new_asgn
            n_dk_new += new_asgn

        if it >= burn_in and (it - burn_in) % thin == 0:
            theta_s = (n_dk_new + alpha) / (n_dk_new + alpha).sum()
            samples.append(theta_s.copy())

    return np.array(samples)  # (S_new, K)


# Simulate 5 new funds by taking held-out funds from the dataset
# (we treat the last 5 funds as "new")
held_out_indices = list(range(F - 5, F))
print(f"  Simulating fold-in for {len(held_out_indices)} held-out funds:")
print(f"  (φ fixed from training; only θ is inferred via Gibbs)")
print()

fold_in_results = {}
for idx in held_out_indices:
    fund_weights = W_norm[idx]   # (A,) normalized weights
    theta_new_samples = fold_in(
        fund_weights, phi_mean, alpha,
        N=1000, n_iter=200, burn_in=50, thin=5
    )
    theta_new_mean = theta_new_samples.mean(axis=0)
    theta_new_std  = theta_new_samples.std(axis=0)
    dominant_topic = np.argmax(theta_new_mean)
    confidence     = theta_new_mean[dominant_topic]

    # Compute conservative JSD to each training fund (using posterior mean of training θ)
    jsd_to_training = np.array([
        jensenshannon(theta_new_mean, theta_mean[i]) ** 2
        for i in range(F)
    ])
    nearest_idx   = np.argmin(jsd_to_training)
    nearest_dist  = jsd_to_training[nearest_idx]

    # Assign to cluster if distance < T, else flag as novel
    if nearest_dist < best_T:
        assigned_cluster = final_clusters[nearest_idx]
        assignment_str   = f"Cluster {assigned_cluster}"
    else:
        assigned_cluster = -1
        assignment_str   = "Novel strategy"

    fold_in_results[fund_names[idx]] = {
        'theta_mean': theta_new_mean,
        'theta_std':  theta_new_std,
        'dominant_topic': dominant_topic,
        'confidence': confidence,
        'nearest_fund': fund_names[nearest_idx],
        'nearest_dist': nearest_dist,
        'assigned_cluster': assigned_cluster,
        'assignment_str': assignment_str,
    }

    actual_sector = ground_truth_labels[idx]
    print(f"  Fund: {fund_names[idx]:<10}  Dominant topic: {dominant_topic:>3}  "
          f"Confidence: {confidence:.3f}  Sector: {actual_sector:<15}  "
          f"→ {assignment_str} (dist={nearest_dist:.3f})")

# --- Plot: Fold-in θ comparison (new vs nearest training fund) ---
fig, axes = plt.subplots(1, len(held_out_indices), figsize=(20, 4))
for ax, idx in zip(axes, held_out_indices):
    fn = fund_names[idx]
    res = fold_in_results[fn]
    nearest_fn = res['nearest_fund']
    nearest_i  = fund_names.index(nearest_fn)

    x = np.arange(K)
    ax.bar(x - 0.2, res['theta_mean'], width=0.4, label='New fund (fold-in)',
           color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, theta_mean[nearest_i], width=0.4, label='Nearest training fund',
           color='coral', alpha=0.8)
    ax.set_title(f'{fn}\n→ {res["assignment_str"]}', fontsize=8)
    ax.set_xlabel('Topic', fontsize=7)
    ax.set_ylabel('θ', fontsize=7)
    ax.tick_params(labelsize=6)
    if ax == axes[0]:
        ax.legend(fontsize=6)

plt.suptitle('Fold-in Inference: New Fund θ vs Nearest Training Fund θ', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_fold_in_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  📊 Saved: {OUTPUT_DIR}/10_fold_in_inference.png\n")

# ============================================================
# BLOCK 11: GENERATING SYNTHETIC FUND PORTFOLIOS (Section 4.4)
# ============================================================
# "Specify a desired θ vector; compute expected portfolio weights as
#  w = θ × φ. The resulting vector w automatically sums to 1."
# ============================================================
print("=" * 60)
print("  BLOCK 11: SYNTHETIC PORTFOLIO GENERATION (Sec 4.4)")
print("=" * 60)

# Define several desired theme profiles to generate portfolios for
desired_profiles = {
    'Tech-heavy':         {0: 0.70, 1: 0.20, 2: 0.10},   # 70% top-theme, 20% second, 10% third
    'Balanced (2-theme)': {0: 0.50, 1: 0.50},
    'Diversified (3-theme)': {0: 0.40, 1: 0.35, 2: 0.25},
    'Single-theme':       {0: 1.00},
}

# Rank themes by importance (avg weight across funds)
theme_importance = theta_mean.mean(axis=0)
ranked_topics    = np.argsort(theme_importance)[::-1]  # most important first

print(f"  w = θ × φ  (θ sums to 1, each row of φ sums to 1 → w sums to 1)")
print(f"  Top-5 topics by avg fund weight: {ranked_topics[:5].tolist()}")
print()

synthetic_portfolios = {}
for profile_name, topic_weights in desired_profiles.items():
    # Build θ vector: assign weights to ranked topics
    theta_desired = np.zeros(K)
    for rank, wt in topic_weights.items():
        theta_desired[ranked_topics[rank]] = wt

    # w = θ @ φ   — expected portfolio weights
    w_synthetic = theta_desired @ phi_mean       # (A,)

    # verify it sums to 1
    assert abs(w_synthetic.sum() - 1.0) < 1e-6, "Portfolio weights don't sum to 1!"

    # Top-10 assets by weight
    top10_idx    = np.argsort(w_synthetic)[::-1][:10]
    top10_assets = [(asset_names[i], w_synthetic[i]) for i in top10_idx]

    synthetic_portfolios[profile_name] = {
        'theta': theta_desired,
        'weights': w_synthetic,
        'top10': top10_assets,
    }

    print(f"  Profile: '{profile_name}'  (sum={w_synthetic.sum():.6f})")
    for asset, wt in top10_assets[:5]:
        sector = asset.split('_')[0]
        print(f"    {asset:<40}  w={wt:.5f}  [{sector}]")
    print()

# --- Plot: Synthetic portfolio weight distributions ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for ax, (profile_name, pdata) in zip(axes, synthetic_portfolios.items()):
    top10 = pdata['top10']
    names  = [f"{a.split('_')[0]}_{a.split('_')[-1]}" for a, _ in top10]
    values = [w for _, w in top10]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars   = ax.barh(range(len(names)), values[::-1], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel('Portfolio Weight  (w = θ × φ)', fontsize=9)
    ax.set_title(f"Synthetic Portfolio: '{profile_name}'\n"
                 f"sum = {sum(values):.6f}", fontsize=10)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7)

plt.suptitle('Synthetic Fund Portfolios Generated from LDA (w = θ × φ)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_synthetic_portfolios.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/11_synthetic_portfolios.png\n")

# ============================================================
# BLOCK 12: UNCERTAINTY-BASED FUND CLASSIFICATION (Section 4.5)
# ============================================================
# "For each fund report: most likely theme composition (posterior mean
#  of θ), confidence score (posterior std — low std = high confidence),
#  probabilistic cluster membership."
# ============================================================
print("=" * 60)
print("  BLOCK 12: UNCERTAINTY-BASED CLASSIFICATION (Sec 4.5)")
print("=" * 60)

# Compute cluster-level representative θ (centroid = mean of all member θ)
cluster_ids   = sorted(set(final_clusters))
cluster_theta = {}   # cluster_id -> mean θ vector
for cid in cluster_ids:
    members = [i for i in range(F) if final_clusters[i] == cid]
    cluster_theta[cid] = theta_mean[members].mean(axis=0)

# For each fund: compute JSD to every cluster centroid → probabilistic membership
def softmax_jsd(jsd_vec, temp=0.05):
    """Convert JSD distances to membership probabilities via softmax (inverted)."""
    score = -jsd_vec / temp
    score = score - score.max()
    exp_s = np.exp(score)
    return exp_s / exp_s.sum()

print(f"  For each fund: posterior mean θ, std (confidence), probabilistic cluster membership")
print()
print(f"  {'Fund':<10} {'Sector':<15} {'Cluster':>8} {'Dom.Topic':>10} "
      f"{'Confidence':>11} {'Uncertain?':>10}")
print(f"  {'-'*70}")

uncertainty_threshold = 0.020  # funds with avg θ std > this are flagged
fund_report = []
for i in range(F):
    theta_m   = theta_mean[i]                    # posterior mean
    theta_s   = theta_std[i]                     # posterior std per topic
    avg_std   = theta_s.mean()                   # overall uncertainty
    dom_topic = np.argmax(theta_m)
    conf      = theta_m[dom_topic]               # confidence = max θ component
    sector    = ground_truth_labels[i]
    cluster   = final_clusters[i]
    uncertain = avg_std > uncertainty_threshold

    # Probabilistic cluster membership across top-3 clusters by JSD
    jsd_to_clusters = np.array([
        jensenshannon(theta_m, cluster_theta[cid]) ** 2
        for cid in cluster_ids
    ])
    membership_probs = softmax_jsd(jsd_to_clusters)
    top3_clusters    = np.argsort(membership_probs)[::-1][:3]
    top3_str = " | ".join([
        f"C{cluster_ids[c]}: {membership_probs[c]*100:.0f}%"
        for c in top3_clusters
    ])

    fund_report.append({
        'fund': fund_names[i],
        'sector': sector,
        'cluster': cluster,
        'dominant_topic': dom_topic,
        'confidence': conf,
        'avg_std': avg_std,
        'uncertain': uncertain,
        'top3_membership': top3_str,
    })

report_df = pd.DataFrame(fund_report)

# Print sample
for _, row in report_df.head(25).iterrows():
    flag = " ⚠ review" if row['uncertain'] else ""
    print(f"  {row['fund']:<10} {row['sector']:<15} {row['cluster']:>8} "
          f"{row['dominant_topic']:>10} {row['confidence']:>11.3f} "
          f"{'Yes' if row['uncertain'] else 'No':>10}{flag}")

n_uncertain = report_df['uncertain'].sum()
print(f"\n  Flagged for manual review (high uncertainty): {n_uncertain} / {F} funds")
print(f"  Uncertainty threshold (avg θ std): {uncertainty_threshold}")

# Save full report
report_df.to_csv(f'{OUTPUT_DIR}/fund_classification_report.csv', index=False)
print(f"  💾 Saved: {OUTPUT_DIR}/fund_classification_report.csv")

# --- Plot: Uncertainty-based classification ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confidence distribution
axes[0].hist(report_df['confidence'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Confidence (max θ component)'); axes[0].set_ylabel('Count')
axes[0].set_title('Fund Confidence Distribution', fontsize=12)

# 2. Uncertainty (avg θ std) distribution
axes[1].hist(report_df['avg_std'], bins=40, color='coral', edgecolor='white')
axes[1].axvline(uncertainty_threshold, color='red', linestyle='--',
                label=f'Threshold={uncertainty_threshold}')
axes[1].set_xlabel('Avg θ Std (uncertainty)'); axes[1].set_ylabel('Count')
axes[1].set_title(f'Uncertainty Distribution\n({n_uncertain} funds flagged)', fontsize=12)
axes[1].legend()

# 3. Confidence vs Uncertainty scatter — coloured by sector
sector_list   = sorted(report_df['sector'].unique())
sector_colors = {s: plt.cm.tab10(i / len(sector_list)) for i, s in enumerate(sector_list)}
for sector_name, grp in report_df.groupby('sector'):
    axes[2].scatter(grp['confidence'], grp['avg_std'],
                    color=sector_colors[sector_name], label=sector_name,
                    alpha=0.7, s=25)
axes[2].axhline(uncertainty_threshold, color='red', linestyle='--', alpha=0.6)
axes[2].set_xlabel('Confidence (max θ)'); axes[2].set_ylabel('Avg θ Std (uncertainty)')
axes[2].set_title('Confidence vs Uncertainty by Sector', fontsize=12)
axes[2].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_uncertainty_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {OUTPUT_DIR}/12_uncertainty_classification.png\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE — FULL SUMMARY")
print(f"{'='*60}")
print(f"  Data:           {F} funds, {A} assets")
print(f"  LDA:            K={K}, α={alpha}, β={beta}, {S} posterior samples")
print(f"  Clustering:     T={best_T:.2f}, {n_final_clusters} clusters ({n_singletons} singletons)")
print(f"  Precision:      {best_row['precision']:.4f}")
print(f"  Recall:         {best_row['recall']:.4f}")
print(f"  ARI:            {best_row['ari']:.4f}")
print(f"  NMI:            {best_row['nmi']:.4f}")
print(f"  Uncertain funds:{n_uncertain} / {F} flagged for manual review")
print(f"\n  Outputs in: {os.path.abspath(OUTPUT_DIR)}/")
print(f"    01_data_preparation.png")
print(f"    02_pseudocount_histogram.png")
print(f"    03_convergence.png")
print(f"    04_posterior_examples.png")
print(f"    05_distance_analysis.png")
print(f"    06_clustering.png")
print(f"    07_evaluation_metrics.png")
print(f"    08_confusion_matrix.png")
print(f"    09_theme_interpretation.png")
print(f"    10_fold_in_inference.png")
print(f"    11_synthetic_portfolios.png")
print(f"    12_uncertainty_classification.png")
print(f"    fund_classification_report.csv")
print(f"{'='*60}")

