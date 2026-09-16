import pandas as pd 
import numpy as np
import scanpy as sc 
import scipy.sparse as sps
import liana as li
import threading, time as _time
import psutil, os
from sklearn.neighbors import NearestNeighbors
import numba
from anndata import AnnData 
import warnings



def format_key(k):
    correspondances = {
        # empty 
        'matrix':'','clustering':'','fun':'','preprocessing':'','hvg_layer':'','function':'','target_sum':'','nbr_weight_decay':'',
        #methods 
        'BANKSY':'BANK','SEDR':'SEDR',
        #params
        'k_geom':'kg','scaled_gaussian':'rgb','param_lambda':'lam','n_comps':'nPC','n_components':'nPC',
        'spatial_n_neighbors':'snn','random_seed':'seed','partition_seed':'seed','sigma_e':'se','alpha':'a',
        'n_top_genes':'g','spatial_coef':'sc',
        }
    if k in correspondances: 
        return(correspondances[k])
    else:
        return(k)
    
def format_value(v):
    if v in ['True','False']:
        v = True if v =='True' else False
        return(v)
    if isinstance(v,int):
        return(int(v))
    elif isinstance(v,float):
        if abs(v) >= 0.01:
            return f"{v:.2f}"
        else:
            return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    elif isinstance(v,str):
        return(format_key(v))
    elif v is None:
        return('Na')
    else:
        print(v,type(v),' unkown')

def format_size(size_in_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"

def format_time(t):
    if t<60:
        return(f'{t:.2f}s')
    elif t<60*60:
        return(f'{t//60}min-{t%60:.2f}s')
    elif t<60*60*24:
        return(f'{t//(60*60)}h-{t%(60*60)//60}min-{t%(60*60)%60:.2f}s')
    else:
        return(f'{t//(60*60*24)}d-{(t%(60*60*24))//(60*60)}h-{(t%(60*60*24))%(60*60)//60}min-{(t%(60*60*24))%(60*60)%60:.2f}s')

def memory_of(m):
    """Memory in bytes of a scipy sparse matrix (CSR, CSC, COO, BSR, DOK, LIL, DIA)."""

    if isinstance(m, (sps.csr_matrix, sps.csr_array)):
        mem = m.indptr.nbytes + m.indices.nbytes + m.data.nbytes
    elif isinstance(m, (sps.csc_matrix, sps.csc_array)):
        mem = m.indptr.nbytes + m.indices.nbytes + m.data.nbytes
    elif isinstance(m, (sps.coo_matrix, sps.coo_array)):
        mem = m.row.nbytes +m.col.nbytes +m.data.nbytes
    elif isinstance(m, (sps.bsr_matrix, sps.bsr_array)):
        mem = m.indptr.nbytes +m.indices.nbytes +m.data.nbytes
    elif isinstance(m, (sps.dia_matrix, sps.dia_array)):
        mem = m.offsets.nbytes + m.data.nbytes
    # elif isinstance(m,torch.Tensor) and m.is_sparse:
    #     m = m.coalesce()
    #     idx_bytes = m.indices().element_size() * m.indices().nelement()
    #     val_bytes = m.values().element_size() * m.values().nelement()
    #     mem = idx_bytes + val_bytes
    elif isinstance(m,np.ndarray):
        mem = m.nbytes
    else:
        raise TypeError(f"Unsupported sparse format: {type(m)}")
    return(format_size(mem))    
        

class Profiler:
    """Per-block wall time + peak process RSS (+children) + peak GPU memory."""
    def __init__(self, poll=0.02, gpu=False):
        self.poll = poll
        self.gpu = gpu
        self.proc = psutil.Process(os.getpid())

    def _rss_tree(self):
        rss = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss

    def __enter__(self):
        self._baseline = self._rss_tree()
        self._peak = self._baseline
        self._stop = False
        self._t0 = _time.perf_counter()
        if self.gpu:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop:
            self._peak = max(self._peak, self._rss_tree())
            _time.sleep(self.poll)

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join()
        self.time = _time.perf_counter() - self._t0
        self.peak_rss = self._peak                     # absolute peak RSS
        self.delta_rss = self._peak - self._baseline   # memory attributable to the block
        self.gpu_peak = 0
        if self.gpu:
            import torch
            torch.cuda.synchronize()
            self.gpu_peak = torch.cuda.max_memory_allocated()
        return False



# Kernels 
def self_tuned_gauss_kernel(mat, n_neighbors, sparse=True):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(mat)
    dists, ids = nbrs.kneighbors(mat)
    dists, ids = dists[:, 1:], ids[:, 1:]
    N, k = ids.shape

    sigma = dists[:, -1]
    sig_i = np.repeat(sigma, k)
    sig_j = sigma[ids.ravel()]

    vals = np.exp(-(dists.ravel() ** 2) / (sig_i * sig_j + 1e-15))
    rows = np.repeat(np.arange(N), k).astype(np.int32)
    cols = ids.ravel().astype(np.int32)

    W = sps.coo_matrix((vals, (rows, cols)), shape=(N, N))
    return W.tocsr() if sparse else np.asarray(W.todense())

def spatial_kernel(coords, n_neighbors, sparse=True):
    return self_tuned_gauss_kernel(coords, n_neighbors, sparse=sparse)

@numba.jit(nopython=True, parallel=True, fastmath=True)
def _sparse_pairwise_dists(data, indices, indptr, rows, cols, sq, out):
    """
    Parallel pairwise Euclidean distance for CSR rows.
    rows[i], cols[i] index into the CSR matrix.
    """
    n_pairs = rows.shape[0]
    for idx in numba.prange(n_pairs):
        r = rows[idx]
        c = cols[idx]

        # Pointers to the two CSR rows we need to intersect
        i   = indptr[r]
        i_end = indptr[r + 1]
        j   = indptr[c]
        j_end = indptr[c + 1]

        dot = 0.0
        while i < i_end and j < j_end:
            col_i = indices[i]
            col_j = indices[j]
            if col_i == col_j:
                dot += data[i] * data[j]
                i += 1
                j += 1
            elif col_i < col_j:
                i += 1
            else:
                j += 1

        # Guard against tiny negatives from rounding drift
        d2 = sq[r] + sq[c] - 2.0 * dot
        if d2 < 0.0:
            d2 = 0.0
        out[idx] = np.sqrt(d2)

def _dists_sparse_numba(X_csr, rows, cols):
    """Numba path: zero Python-loop chunked overhead."""
    # Numba is strict about types — ensure contiguous native arrays
    data    = np.asarray(X_csr.data,    dtype=np.float64)
    indices = np.asarray(X_csr.indices, dtype=np.int32)
    indptr  = np.asarray(X_csr.indptr,  dtype=np.int32)
    rows    = np.asarray(rows,          dtype=np.int32)
    cols    = np.asarray(cols,          dtype=np.int32)

    # Pre-compute squared row norms (same as before, but float64)
    sq = np.asarray(X_csr.power(2).sum(axis=1)).ravel().astype(np.float64)

    nnz = rows.shape[0]
    dists = np.empty(nnz, dtype=np.float64)

    # Single compiled, parallel call
    _sparse_pairwise_dists(data, indices, indptr, rows, cols, sq, dists)
    return dists

def expression_kernel_at_indices(X, ws, coef, chunk=500_000,
                                  mem_budget_gb=15.0):
    """
    Unchanged API, but sparse path now uses Numba instead of chunked Python.
    ( chunk is kept for backward compatibility; it is no longer used
      in the sparse branch because Numba handles the full nnz vector. )
    """
    is_sparse = sps.issparse(X)
    coo = sps.coo_matrix(ws)
    rows, cols = coo.row, coo.col
    nnz, n = len(rows), X.shape[0]

    if is_sparse:
        dists = _dists_sparse_numba(X.tocsr(), rows, cols)
    else:
        X = np.asarray(X, dtype=np.float64)
        dists = _dists_dense(X, rows, cols, nnz, mem_budget_gb)

    bw = np.median(dists) * coef
    vals = np.exp(-dists ** 2 / (2 * bw ** 2))
    return sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

def _dists_dense(X, rows, cols, nnz, mem_budget_gb):
    """Distances for dense X — strategy picked by matrix size."""
    n, G = X.shape
    sq = (X ** 2).sum(1)

    full_bytes = n * n * 8
    if full_bytes < mem_budget_gb * 1e9:
        print('compute full matrix then masks')
        gram = X @ X.T
        D2 = sq[:, None] + sq[None, :] - 2 * gram
        del gram
        np.maximum(D2, 0, out=D2)
        np.sqrt(D2, out=D2)
        dists = D2[rows, cols]
        del D2
        return dists

    block = max(1, int(mem_budget_gb * 1e9 / (n * 8)))
    dists = np.empty(nnz, dtype=np.float64)
    print(f'by blocks n={n} block={block}')
    for b_start in range(0, n, block):
        b_end = min(b_start + block, n)
        print('b start end', b_start, b_end)
        mask = (rows >= b_start) & (rows < b_end)
        if not mask.any():
            continue

        gram_block = X[b_start:b_end] @ X.T
        r_local = rows[mask] - b_start
        c_local = cols[mask]
        d2 = sq[rows[mask]] + sq[c_local] - 2 * gram_block[r_local, c_local]
        dists[mask] = np.sqrt(np.maximum(d2, 0.0))
        del gram_block

    return dists

def add_diagonal(w, coef):
    """Add coef · row_sums to the diagonal."""
    w = sps.csr_matrix(w)
    sums = np.asarray(w.sum(1)).ravel()
    return w + sps.diags(coef * sums)

def row_normalize(w):
    """Row-normalize a sparse matrix."""
    w = sps.csr_matrix(w)
    sums = np.asarray(w.sum(1)).ravel()
    sums[sums == 0] = 1.0
    return sps.diags(1.0 / sums) @ w

def neighborhood_product(X, w):
    """Row-normalised weighted average: norm(w) @ X."""
    return row_normalize(w) @ X

def pca_on_adata(adata: AnnData, n_comps: int = 50, random_state: int = 0):
    """
    Fast, sparse-aware PCA. Keeps the matrix sparse when possible and uses
    the randomized SVD solver, which is much faster than ARPACK for a small
    fixed number of components.
    """
    X = adata.X

    # --- NaN handling, sparse-aware -----------------------------------------
    if sps.issparse(X):
        # NaNs live only in .data for CSR/CSC
        if np.isnan(X.data).any():
            X = X.copy()
            X.data[np.isnan(X.data)] = 0
            adata.X = X
    else:
        if np.isnan(X).any():
            print("pca_on_adata: applied on a matrix with NaN values")
            adata.X = np.nan_to_num(X, nan=0.0)

    # --- PCA ---------------------------------------------------------------
    sc.pp.pca(
        adata,
        n_comps=n_comps,
        svd_solver="randomized",   # faster than "arpack" for 50 comps
        zero_center=None,          # auto: sparse -> TruncatedSVD, dense -> PCA
        random_state=random_state,
    )
    return adata

def pca_of_mat(adata, mat, n_comps=50, key="X"):
    """PCA on *mat*, store in adata.obsm['{key}_pca']."""
    mat = mat.tocsr() if hasattr(mat, 'tocsr') else np.asarray(mat)
    tmp = AnnData(mat, obs=adata.obs)
    pca_on_adata(tmp, n_comps=n_comps)          # your existing helper
    adata.obsm[f"{key}_pca"] = tmp.obsm["X_pca"]

def within_weights(w, labels, normalize=True):
    """Poids intra-type. Si normalize, w est row-normalisé AVANT masquage,
    de sorte que within + cross = rownorm(w) exactement."""
    w = row_normalize(w) if normalize else _as_sparse(w)
    cats = labels.cat.categories if hasattr(labels, "cat") else np.unique(labels)
    return sum(
        sps.diags((labels == c).to_numpy().astype(np.float64)) @ w
        @ sps.diags((labels == c).to_numpy().astype(np.float64))
        for c in cats
    )

def cross_weights(w, labels, normalize=True):
    w = row_normalize(w) if normalize else _as_sparse(w)
    return w - within_weights(w, labels, normalize=False)

def decompose_weights(w, labels, normalize=True):
    w = row_normalize(w) if normalize else _as_sparse(w)
    print("w",w.shape)
    print("labels",len((labels == labels.cat.categories.tolist()[0])))
    return {c: w @ sps.diags((labels == c).to_numpy().astype(float))
            for c in labels.cat.categories}



















def _as_sparse(w):
    return w if sps.issparse(w) else sps.csr_matrix(w)

def to_array(x):
    """Any matrix → dense numpy array."""
    if sps.issparse(x):
        return np.asarray(x.todense())
    if hasattr(x, "numpy"):          # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_dense(X):
    return X.toarray() if sps.issparse(X) else np.asarray(X, dtype=np.float64)

def lognorm(X, target=1e4):
    """Library-size normalise + log1p."""
    X = to_dense(X).astype(np.float64)
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1
    return np.log1p(X / s * target)

def zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd

def prep(X):
    """Raw counts → log-normalised, gene-wise z-scored."""
    # return (lognorm(X))
    return zscore(lognorm(X))

def safe_nz(matrix, axis):
    counts = (matrix != 0).sum(axis)
    counts[counts == 0] = 1
    return counts





def get_joint_knn_graph(latents_1, k,knn_metric = 'manhattan', latents_2=None):
    r"""
    Computes k nearest neighbors of latents_1 among latents_2
    :param latents_1: array of points
    :param latents_2: array of points or None if base and query are the same
    :param k: number of neighbors to keep
    :param knn_metric: metric used to measure distances (must be a sklearn metric keyword like manhattan)
    :return:
    """
    exclude_same = False
    if latents_2 is None:
        latents_2 = latents_1
        exclude_same = True
    k = min(k, len(latents_2))
    if exclude_same:
        k += 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=knn_metric).fit(latents_2)
    _, indices = nbrs.kneighbors(latents_1)
    if exclude_same:
        indices = indices[:, 1:]
    return indices


def knn_indices(mat, k, metric="euclidean", approx=True):
    """Euclidean brute force uses BLAS (fast, multithreaded); pynndescent if available."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if approx:
        try:
            from pynndescent import NNDescent
            idx = NNDescent(mat, n_neighbors=k + 1, metric=metric,
                            n_jobs=-1, random_state=0).neighbor_graph[0]
            return idx[:, 1:k + 1]
        except ImportError:
            pass
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                          metric=metric, n_jobs=-1).fit(mat)
    return nn.kneighbors(mat, return_distance=False)[:, 1:]

def get_purity_vector_per_domain(mat, cell_type, domain_labels,
                                  ks=(10, 20, 30, 50, 70, 100), **kw):
    _, ct = np.unique(np.asarray(cell_type), return_inverse=True)
    cl = np.asarray(domain_labels)
    idx = knn_indices(mat, max(ks), **kw)
    match = (ct[idx] == ct[:, None])                       # (n, k_max), computed ONCE
    csum = np.cumsum(match, axis=1, dtype=np.int32)        # all ks in one pass
    cols = np.array(ks) - 1
    per_cell = csum[:, cols] / np.array(ks)                # (n, len(ks))
    return list(ks), {c: per_cell[cl == c].mean(0).tolist() for c in np.unique(cl)}









LR_MIN_SUPPORT = 100        # min. number of positive cells for a conditional mean
LR_SEP = "__"               # cell-type names contain "_", so use "__" as separator


def _nnz_per_column(S):
    """Number of non-zero entries per column, sparse- and dense-safe."""
    if sps.issparse(S):
        return S.getnnz(axis=0).astype(np.int64)
    return np.count_nonzero(S, axis=0).astype(np.int64)


def _nnz_per_row(S):
    if sps.issparse(S):
        return S.getnnz(axis=1).astype(np.int64)
    return np.count_nonzero(S, axis=1).astype(np.int64)




def lr_code_of(communication_type, groups):
    if communication_type == "all":
        return "all"
    if communication_type in ("exocrine", "endocrine"):
        return f"{communication_type}_{groups}"
    return f"{groups}"

def get_LR_matrices(adata, communication_type="exocrine",
                          groups="cell_type", w_key="wc", groups_of_interest=None):
    """
    Build gene-subsetted expression (zX_lr) and context (zC_lr_*) matrices,
    materialising only ligand/receptor columns to keep memory low.
    Results are stored in adata.obsm.
    """
    pairs = adata.uns["context_LR"]["pairs"]
    var   = pd.Index(adata.var_names)
    code  = lr_code_of(communication_type, groups)

    if f"zC_lr_{code}" in adata.obsm:
        return code

    # --- subset to ligand/receptor genes only ---------------------------
    lr_genes = sorted(set(pairs["ligand"]) | set(pairs["receptor"]))
    adata.uns["context_LR"]["lr_idx"] = pd.Index(lr_genes)
    lr_cols  = var.get_indexer(lr_genes)

    X_lr = adata.layers['counts']
    tmp = AnnData(adata.layers['counts'].copy(), var=adata.var)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_lr = tmp.X[:, lr_cols]
    mX = np.asarray(X_lr.mean(0)).ravel() + 1e-12
    adata.obsm['zX_lr'] = to_array(X_lr @ sps.diags(1.0/mX))
    
    # --- context matrices ------------------------------------------------
    if communication_type == "all":
        # context = (adata.layers["C"][:, lr_cols] if "C" in adata.layers
        #            else adata.obsm[w_key] @ X_lr)
        C_lr = row_normalize(adata.obsm[w_key]) @ X_lr
        mC = np.asarray(C_lr.mean(0)).ravel() + 1e-12
        adata.obsm[f"zC_lr_{code}"] = to_array(C_lr @ sps.diags(1.0/mC))


    elif communication_type in ("exocrine", "endocrine"):
        weights = (cross_weights if communication_type == "exocrine"
                else within_weights)(adata.obsm[w_key], adata.obs[groups])
        C_lr = to_array(weights @ X_lr) 
        mC = np.asarray(C_lr.mean(0)).ravel() + 1e-12
        adata.obsm[f"zC_lr_{code}"] =to_array(C_lr @ sps.diags(1.0/mC)) 

    elif communication_type == "groups":
        groups_of_interest = (adata.obs[groups].unique() if groups_of_interest is None else groups_of_interest)
        for label, w_label in decompose_weights(adata.obsm[w_key], adata.obs[groups]).items():
            if label in groups_of_interest:
                C_lr = to_array(w_label @ X_lr)
                mC = np.asarray(C_lr.mean(0)).ravel() + 1e-12
                adata.obsm[f"zC_lr_{code}_c{label}"] = to_array(C_lr @ sps.diags(1.0/mC))
    else:
        raise ValueError(communication_type)

    return code

def compute_LR_score(adata,context_key,score_key):
    
    pairs = adata.uns["context_LR"]["pairs"]
    lr_genes = sorted(set(pairs["ligand"]) | set(pairs["receptor"]))
    adata.uns["context_LR"]["lr_idx"] = pd.Index(lr_genes)
    lr_idx       = adata.uns["context_LR"]["lr_idx"]
    ligand_idx   = lr_idx.get_indexer(pairs["ligand"])
    receptor_idx = lr_idx.get_indexer(pairs["receptor"])
    zX           = adata.obsm["zX_lr"]
    matrices_ready = True
    S = zX[:, receptor_idx] * adata.obsm[context_key][:, ligand_idx]
    adata.obsm[score_key] = S
    return(S)



def _pair_summary(S, min_support=LR_MIN_SUPPORT):
    """
    Per-pair (column-wise) summaries of a score block S (n_cells x n_pairs).

    Returns raw accumulators (sum, n_cells, n_pos) so that section- or
    region-level statistics can be pooled exactly afterwards, plus the three
    derived quantities of the decomposition  mean = frac * mean_pos.
    """
    n_cells = S.shape[0]
    total = np.asarray(S.sum(0), dtype=np.float64).ravel()
    n_pos = _nnz_per_column(S)
    mean = total / n_cells if n_cells else np.full(S.shape[1], np.nan)
    frac = n_pos / n_cells if n_cells else np.full(S.shape[1], np.nan)
    mean_pos = np.divide(total, n_pos, out=np.full(S.shape[1], np.nan),
                         where=n_pos >= min_support)          # NaN, not 0
    return {"sum": total, "n_cells": np.full(S.shape[1], n_cells, dtype=np.int64),
            "n_pos": n_pos, "mean": mean, "frac": frac, "mean_pos": mean_pos}
            
def accumulate_lr_stats( S, score_key, stat_code, receiver_masks,
                         obs_stats, obs_names, pair_stats, pair_names,
                         min_support=LR_MIN_SUPPORT, col_slice=None):
    """
    Accumulate per-cell and per-pair summaries of one score matrix.

    S               : (n_cells x n_pairs) non-negative score block. May be a
                      column chunk of the full matrix; pass `col_slice` then.
    score_key       : name of the score matrix, e.g. 'S_heterotypic_main_cell_type'.
    stat_code       : short code identifying the communication type / sender.
    receiver_masks  : {group_label: boolean mask over cells} -- receiver groups
                      for which per-pair statistics are also reported.

    Per-cell statistics are only accumulated when the full pair axis is present
    (col_slice is None), since a mean over a chunk of pairs is meaningless.
    """
    # ---- per-cell (row-wise): aggregate over ligand-receptor pairs --------
    if col_slice is None:
        n_pairs = S.shape[1]
        row_total = np.asarray(S.sum(1), dtype=np.float64).ravel()
        obs_stats.extend([row_total / n_pairs])
        obs_names.extend([f"{score_key}_mean"])

    # ---- per-pair (column-wise), all cells --------------------------------
    for stat, values in _pair_summary(S, min_support).items():
        pair_stats.append(values)
        pair_names.append(f"{stat}{LR_SEP}{stat_code}")

    # ---- per-pair, restricted to each receiver group ----------------------
    for group, mask in receiver_masks.items():
        if mask.sum() == 0:
            continue
        for stat, values in _pair_summary(S[mask], min_support).items():
            pair_stats.append(values)
            pair_names.append(f"{stat}{LR_SEP}{stat_code}{LR_SEP}{group}")


# def accumulate_lr_stats( S, score_key, stat_code, er, cell_type_masks,
#                         obs_stats, obs_names, pair_stats, pair_names):
#     # per-observation summaries
#     nz_obs = safe_nz(S, 1)
#     obs_stats.extend([S.mean(1), S.sum(1), S.sum(1) / nz_obs])
#     obs_names.extend([f"{score_key}_m", f"{score_key}_s", f"{score_key}_snz"])

#     # per-pair summaries (all cells)
#     nz_pair = safe_nz(S, 0)
#     pair_stats.extend([S.mean(0), S.sum(0), S.sum(0) / nz_pair])
#     pair_names.extend([f"m{er}_{stat_code}", f"s{er}_{stat_code}", f"m{er}nz_{stat_code}"])

#     # per-pair summaries (per cell type)
#     for ct, mask in cell_type_masks.items():
#         Sc = S[mask]
#         nz_c = safe_nz(Sc, 0)
#         pair_stats.extend([Sc.mean(0), Sc.sum(0), Sc.sum(0) / nz_c])
#         pair_names.extend([f"m{er}_{stat_code}_{ct}",
#                         f"s{er}_{stat_code}_{ct}",
#                         f"m{er}nz_{stat_code}_{ct}"])







def lr_progression_df(adata, sender, receiver, regions,
                      annotation='annotation', region='region',
                      k=20, score='R', zscore=False, nonzero_only=False):
    """
    Mean LR score for a sender→receiver pair.
    Rows = LR pairs (top-k by total), columns = ordered `regions`.

    interaction_class : 'signaling', 'ecm_adhesion', or None (all).
    nonzero_only      : if True, per-region per-pair mean is taken over cells
                        with non-zero score only (matches the 'snz' statistic),
                        so silent cells don't dilute the average.
    zscore            : z-score each row across the selected regions.
    """
    keys = [key for key in adata.obsm
            if key.startswith(f'LR_{score}') and key.endswith(f'_c{sender}')]
    if not keys:
        raise KeyError(f'No LR matrix for sender {sender}')

    pairs = adata.uns['context_LR']['pairs']
    mask  = adata.obs[annotation] == receiver
    mat   = np.asarray(adata.obsm[keys[0]][mask])
    reg   = adata.obs.loc[mask, region].values

    def region_mean(sub):
        if not nonzero_only:
            return sub.mean(axis=0)
        nz = (sub != 0).sum(axis=0)
        mean_nz = sub.sum(axis=0) / np.where(nz == 0, 1, nz)
        mean_nz[nz < 100] = 0        
        return mean_nz
    df = pd.DataFrame({r: region_mean(mat[reg == r]) for r in regions},
                      index=pairs.index)

    top = df.max(axis=1).nlargest(k).index
    df  = df.loc[top]
    if zscore:
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)
    return df









#########
#   LR pairs preprocessing 
########



import re

# Organism of each dataset, used to pick the right LR resource.
DATASET_ORGANISM = {
    'MOSTA':   'mouse',      # mouse embryo, Stereo-seq
    'MBA':     'mouse',      # mouse brain atlas, MERFISH
    'CARDIO':  'mouse',      # mouse heart, Xenium
    'LYMPH':   'human',      # HNSCC lymph node, Open-ST / Xenium
    'ZESTA':   'zebrafish',  # zebrafish embryo, Stereo-seq
    'ARTISTA': 'axolotl',    # axolotl telencephalon, Stereo-seq
}


def get_organism(adata, default='human'):
    """Organism of `adata`, inferred from adata.uns['data']['dataset']."""
    dataset = adata.uns.get('data', {}).get('dataset')
    if dataset not in DATASET_ORGANISM:
        print(f"get_organism: unknown dataset {dataset!r}, falling back to {default!r}")
    return DATASET_ORGANISM.get(dataset, default)


def normalized_symbol(gene_name):
    """
    Canonical form of a gene symbol for cross-species matching.
    'Col1a1|AMEX60DD001' -> 'COL1A1' ; 'col1a1a' -> 'COL1A1' ; 'FN1.1' -> 'FN1'
    """
    symbol = str(gene_name).split('|')[0].split('.')[0]
    symbol = re.sub(r'[-_](?:1|2|like)$', '', symbol, flags=re.IGNORECASE)
    return symbol.upper()


def register_LR_pairs(adata, pairs, organism, verbose=True):
    """
    Keep the LR pairs whose ligand *and* receptor are both in adata.var_names,
    store them in adata.uns['context_LR'] and return them.
    Receptor complexes (subunits joined by '_') are dropped, since no single
    panel gene matches them.
    """
    pairs = pairs[['ligand', 'receptor']].dropna().drop_duplicates()
    panel_genes = set(adata.var_names)
    is_measured = pairs['ligand'].isin(panel_genes) & pairs['receptor'].isin(panel_genes)
    pairs = pairs.loc[is_measured].copy()

    pairs['LR'] = pairs['ligand'] + '_' + pairs['receptor']
    pairs = pairs.set_index('LR', drop=False)

    if len(pairs) == 0:
        print(f"register_LR_pairs: WARNING no LR pair recovered for organism={organism!r} "
              f"(panel example: {list(adata.var_names[:5])})")
    elif verbose:
        print(f"{organism}: {len(pairs)} pairs, {pairs['ligand'].nunique()} ligands, "
              f"{pairs['receptor'].nunique()} receptors")

    adata.uns['context_LR'] = {'pairs': pairs, 'organism': organism}
    return pairs


def get_zebrafish_LR_pairs(adata, min_evidence=3, verbose=True):
    """
    Human consensus resource translated to zebrafish symbols through HCOP orthologs,
    keeping only one-to-one orthologs. Unmapped symbols are matched case-insensitively
    as a fallback (ZESTA symbols are lowercase zebrafish symbols).
    """
    human_pairs = li.resource.select_resource('consensus')
    orthologs = li.rs.get_hcop_orthologs(
        columns=['human_symbol', 'zebrafish_symbol'], min_evidence=min_evidence
    ).rename(columns={'human_symbol': 'source', 'zebrafish_symbol': 'target'})

    zebrafish_pairs = li.rs.translate_resource(
        human_pairs, map_df=orthologs, columns=['ligand', 'receptor'],
        replace=True, one_to_many=1,
    )
    zebrafish_pairs = match_pairs_to_panel(adata, zebrafish_pairs)
    return register_LR_pairs(adata, zebrafish_pairs, organism='zebrafish', verbose=verbose)


def get_axolotl_LR_pairs(adata, verbose=True):
    """
    The AmexT_v47 annotation used by ARTISTA labels genes with their human orthologue
    symbol, sometimes suffixed ('SYMBOL|AMEX60DD...', 'SYMBOL_1'). We therefore map the
    human consensus resource onto the panel by normalized symbol.
    """
    human_pairs = li.resource.select_resource('consensus')
    axolotl_pairs = match_pairs_to_panel(adata, human_pairs)
    return register_LR_pairs(adata, axolotl_pairs, organism='axolotl', verbose=verbose)


def match_pairs_to_panel(adata, pairs):
    """Rewrite ligand/receptor symbols with the panel gene they normalize to."""
    panel_by_symbol = {normalized_symbol(gene): gene for gene in adata.var_names}
    pairs = pairs[['ligand', 'receptor']].dropna().drop_duplicates().copy()
    for role in ('ligand', 'receptor'):
        pairs[role] = pairs[role].map(lambda s: panel_by_symbol.get(normalized_symbol(s)))
    return pairs.dropna()


def get_LR_pairs(adata, organism=None, verbose=True):
    """Select LR pairs measured in `adata` from the resource matching its organism."""
    organism = organism or get_organism(adata)

    if organism in ('mouse', 'human'):
        resource = 'mouseconsensus' if organism == 'mouse' else 'consensus'
        pairs = li.resource.select_resource(resource)
        return register_LR_pairs(adata, pairs, organism=organism, verbose=verbose)
    if organism == 'zebrafish':
        return get_zebrafish_LR_pairs(adata, verbose=verbose)
    if organism == 'axolotl':
        return get_axolotl_LR_pairs(adata, verbose=verbose)
    raise ValueError(f"get_LR_pairs: no LR resource defined for organism {organism!r}")



#########
# Niche #
#########

def assert_cell_type(adata,cell_type_key):
    if cell_type_key not in adata.obs:
        raise KeyError(
            f"Column '{cell_type_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )

def compute_niche_context(adata,w,cell_type_key = "cell_type",verbose=False):
    """
    Compute the cellular niche context for each observation from a spatial
    weight matrix and categorical cell-type annotations, using **PyTorch
    sparse tensors** (with optional GPU acceleration).

    The niche context of cell *i* is defined as the weighted cell-type
    composition of its spatial neighbourhood:

        N = W @ C            (raw)
        N = D⁻¹ W @ C       (row-normalised, i.e. proportions)

    where C is the (n_obs × n_celltypes) sparse one-hot encoding matrix
    and D = diag(W 1).

    
    """
    key_added = f'niche_{cell_type_key}'
    assert_cell_type(adata,cell_type_key)
    # ---- one-hot encoding C as torch sparse CSR --------------------------
    categories,Q = one_hot_encoding_category(adata,cell_type_key)
    # ---- niche context: N = W @ C  (sparse CSR × sparse CSR) ------------
    niche = w @ Q
    if verbose:
        print(f'memory of niche {key_added}:',memory_of(niche))
    adata.obsm[key_added] = niche
    adata.uns[key_added] = dict(
        cell_type_key=cell_type_key,
        categories=list(categories),
    )
    niche_cols = {f'{key_added}_{cat}': niche[:, i].toarray().ravel() for i, cat in enumerate(categories)}
    adata.obs = pd.concat([adata.obs, pd.DataFrame(niche_cols, index=adata.obs.index)], axis=1)
    return adata 


def one_hot_encoding_category(adata, column):

    labels = adata.obs[column]
    if not hasattr(labels, "cat"):
        labels = labels.astype("category")

    categories = labels.cat.categories
    codes = labels.cat.codes.to_numpy()
    n_obs = len(adata)
    n_cats = len(categories)

    valid = codes >= 0
    n_invalid = n_obs - valid.sum()
    if n_invalid:
        warnings.warn(
            f"{n_invalid} cells have missing/NaN labels in '{column}'; "
            "they contribute zero rows to the one-hot matrix.",
            stacklevel=2,
        )

    row = np.flatnonzero(valid)
    col = codes[valid]
    data = np.ones(row.size, dtype=np.float64)
    Q = sps.coo_matrix(
        (data, (row, col)),
        shape=(n_obs, n_cats),
        dtype=np.float64,
    ).tocsr()

    adata.obsm[f"one_hot_{column}"] = Q
    return categories, Q


def niche_enrichment(adata, group,categories_of_interest,
                     key="niche_annotation", categories=None):
    """Log2 enrichment of niche context in clusters vs global baseline."""
    N = adata.obsm[key].toarray() if sps.issparse(adata.obsm[key]) else np.asarray(adata.obsm[key])
    categories = categories or adata.uns[key]["categories"]
    global_mean = N.mean(axis=0)  # expected
    print(global_mean)
    idx = np.where(adata.obs[group].isin(categories_of_interest))[0]
    cluster_mean = N[idx].mean(axis=0)  # observed
    print(cluster_mean)
    enrichment = np.log2((cluster_mean + 1e-9) / (global_mean + 1e-9))
    return pd.Series(enrichment, index=categories)




# Communication



_D = lambda M: np.asarray(M.todense() if sps.issparse(M) else M, dtype=np.float32)

def _pairs_idx(adata):
    g, p = pd.Index(adata.uns['lr']['genes']), adata.uns['lr']['pairs']
    return g.get_indexer(p['ligand']), g.get_indexer(p['receptor']), p.index


def lr_normalize(adata, pairs, target_sum=1e4, key='X_lr'):
    from sklearn.preprocessing import normalize
    keep = pd.Index(sorted(set(pairs.ligand) | set(pairs.receptor))).intersection(adata.var_names)
    C = sps.csr_matrix(adata.layers['counts'], dtype=np.float32)
    C = sps.diags(target_sum / np.maximum(np.asarray(C.sum(1)).ravel(), 1)) @ C 
    adata.obsm[key] = C[:, adata.var_names.get_indexer(keep)].log1p().tocsr()
    if 'lr' not in adata.uns:
        adata.uns['lr'] = {'genes': list(keep), 'pairs': pairs[pairs.ligand.isin(keep) & pairs.receptor.isin(keep)]}



def lr_availability(adata, sender=None, groups='main_cell_type', w_key='wc', key='X_lr'):
    W = sps.csr_matrix(adata.obsm[w_key], dtype=np.float32)
    W = sps.diags(1 / np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)) @ W
    t = np.ones(adata.n_obs, np.float32) if sender is None else (adata.obs[groups] == sender).values.astype(np.float32)
    m, k2, s = W @ t, W.multiply(W) @ t, sender or 'all'
    adata.obs[f'm_{s}'] = m 
    adata.obs[f'neff_{s}'] = np.where(k2 > 0, m**2 / np.maximum(k2, 1e-24), 0.)
    adata.obsm[f'A_{s}'] = (W @ adata.obsm[key].multiply(t[:, None])).tocsr()
    adata.obsm[f'P_{s}'] = (sps.diags(1 / np.maximum(m, 1e-12)) @ adata.obsm[f'A_{s}']).tocsr()


def lr_moments(adata, src, om=None, q=0.05,r=None):
    M = adata.obsm[src] if om is None else adata.obsm[src][om]
    if r is not None:
        M = M[(adata.obs[r['col']]==r['value']).values] if om is None else M[(adata[om].obs[r['col']]==r['value']).values]
    mu = np.asarray(M.mean(0)).ravel().astype(np.float32)
    ex2 = np.asarray(M.multiply(M).mean(0)).ravel() if sps.issparse(M) else (_D(M)**2).mean(0)
    sd = np.sqrt(np.maximum(ex2 - mu**2, 0)).astype(np.float32)
    if r is not None:
        adata.uns['lr'][f"mom_{src}_{r['col']}_{r['value']}"] = {'mu': mu, 'sd': sd + 1e-9, 'den': mu + np.quantile(mu[mu > 0], q)}
    else:
        adata.uns['lr'][f'mom_{src}'] = {'mu': mu, 'sd': sd + 1e-9, 'den': mu + np.quantile(mu[mu > 0], q)}


def lr_support(adata, sender='all', receiver=None, groups='main_cell_type'):
    r = np.ones(adata.n_obs, bool) if receiver is None else (adata.obs[groups] == receiver).values
    return r & (adata.obs[f'm_{sender}'].values > 0)

# def _norm(adata, src, om, cols, mode):                     # slice columns BEFORE densifying
#     M, s = _D(adata.obsm[src][om][:, cols]), adata.uns['lr'][f'mom_{src}']
#     if mode == 'raw':
#         return(M)
#     if mode == 'mean': 
#         return M / s['mu'][cols]
#         # return M / s['den'][cols]
#     Z = (M - s['mu'][cols]) / s['sd'][cols]
#     return np.clip(Z, 0, None) if mode == 'zplus' else Z

def _norm(adata, src, om, cols, mode,r=None):                     # slice columns BEFORE densifying
    if r is None:
        M = _D(adata.obsm[src][om][:, cols])
        s = adata.uns['lr'][f'mom_{src}']
    else:
        M = _D(adata.obsm[src][om][:, cols])
        M = M[(adata[om].obs[r['col']]==r['value']).values]
        s = adata.uns['lr'][f"mom_{src}_{r['col']}_{r['value']}"]

    if mode == 'raw':
        return(M)
    if mode == 'mean': 
        # return M / s['mu'][cols]
        return M / s['den'][cols]
    Z = (M - s['mu'][cols]) / s['sd'][cols]
    return np.clip(Z, 0, None) if mode == 'zplus' else Z

# def lr_score(adata, sender, receiver, groups='main_cell_type', mode='mean'):
#     om, (li, ri, names) = lr_support(adata, sender, receiver, groups), _pairs_idx(adata)
#     R = _norm(adata, 'X_lr', om, ri, mode)
#     L = _norm(adata, f'A_{sender}', om, li, mode)
#     return pd.DataFrame(R * L, index=adata.obs_names[om], columns=names)


def lr_score(adata, sender, receiver, groups='main_cell_type', mode='mean', r=None, src='A'):
    om, (li, ri, names) = lr_support(adata, sender, receiver, groups), _pairs_idx(adata)
    R = _norm(adata, 'X_lr', om, ri, mode, r=r)
    L = _norm(adata, f'{src}_{sender}', om, li, mode, r=r)
    return pd.DataFrame(R * L, index=adata.obs_names[om], columns=names)


def lr_aggregate(adata, S, sender, by=None):

    # w = adata.obs.loc[S.index, f'neff_{sender}'].values 
    w = adata.obs.loc[S.index, f'm_{sender}'].values
    g = np.array(['all'] * len(S)) if by is None else adata.obs.loc[S.index, by].astype(str).values
    num = S.mul(w, axis=0).groupby(g).sum()
    den = pd.Series(w, index=S.index).groupby(g).sum()
    pos = (S > 0).mul(w, axis=0).groupby(g).sum()
    return {'all': num.div(den, axis=0), 'prev': pos.div(den, axis=0), 'pos': num / pos.replace(0, np.nan)}

def lr_log(T, floor=1e-2): return np.log2(T.clip(lower=floor))

# def lr_cell_score(adata, sender='all', receiver=None, groups='main_cell_type', mode='mean', col=None, chunk=64):
#     om = lr_support(adata, sender, receiver, groups)
#     li, ri, names = _pairs_idx(adata) 
#     v = 0.
#     for k in range(0, len(names), chunk):
#         s = slice(k, k + chunk)
#         v = v + (_norm(adata, 'X_lr', om, ri[s], mode) * _norm(adata, f'A_{sender}', om, li[s], mode)).sum(1)
#     adata.obs[col or f'LR_{sender}_{mode}'] = pd.Series(v / len(names), index=adata.obs_names[om])

def lr_cell_score(adata, sender='all', receiver=None, groups='main_cell_type', mode='mean', col=None, chunk=64,r=None,src='A'):
    om = lr_support(adata, sender, receiver, groups)
    li, ri, names = _pairs_idx(adata) 
    v = 0.
    for k in range(0, len(names), chunk):
        s = slice(k, k + chunk)
        v = v + (_norm(adata, 'X_lr', om, ri[s], mode,r=r) * _norm(adata, f'{src}_{sender}', om, li[s], mode, r=r)).sum(1)
    adata.obs[col or f'LR_{src}_{sender}_{mode}'] = pd.Series(v / len(names), index=adata.obs_names[om])


