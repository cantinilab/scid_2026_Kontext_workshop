# import torch
# import numpy as np
import json
import scipy.sparse as sps 
import pandas as pd
import numpy as np
from pathlib import Path
from kontext.clustering import gmm_clustering, kmeans_clustering
from kontext.utils import format_key,format_value,spatial_kernel,expression_kernel_at_indices, add_diagonal, memory_of, neighborhood_product, pca_of_mat, cross_weights, compute_niche_context, get_LR_matrices,accumulate_lr_stats,compute_LR_score
from kontext.utils import lr_normalize,lr_moments,lr_availability,lr_moments,lr_score,lr_aggregate,lr_cell_score,lr_log
import logging 

log = logging.getLogger(__name__)


CLUSTERERS = {
    "gmm": gmm_clustering,
    "kmeans": kmeans_clustering,
}


def signature(params):
    se, snn = params.get('sigma_e'), params.get('snn')
    if isinstance(se, list) or isinstance(snn, list):
        se_l = se if isinstance(se, list) else [se] * len(snn)
        snn_l = snn if isinstance(snn, list) else [snn] * len(se)
        rest = {k: v for k, v in params.items() if k not in ('sigma_e', 'snn')}
        return "+".join(signature(dict(snn=s, sigma_e=e, **rest))
                        for s, e in zip(snn_l, se_l))
    return "_".join(f"{format_key(k)}{format_value(v)}"
                    for k, v in params.items() if v is not None)


class Kontext:

    _ARTIFACTS = ("weights", "E_pca", "C_pca", "E_umap", "C_umap", "spatial_domains", "scores",'LR_scores')

    def __init__(self, sigma_e=1, snn=20, alpha=1/2, n_comps=50,
                save=None, load=None, verbose=False,chunk=50_000):

        self.sigma_e = sigma_e
        self.snn = snn
        self.alpha = alpha
        self.n_comps = n_comps
        self.verbose = verbose
        self.chunk = chunk
        
        self.save = {k: False for k in self._ARTIFACTS}
        self.load = {k: False for k in self._ARTIFACTS}
        if save is not None:
            self.save.update(save)
        if load is not None:
            self.load.update(load)

        # --- list support ---
        self._is_list = isinstance(sigma_e, list) or isinstance(snn, list)
        if self._is_list:
            se_l = sigma_e if isinstance(sigma_e, list) else [sigma_e] * len(snn)
            snn_l = snn if isinstance(snn, list) else [snn] * len(sigma_e)
            self._pairs = list(zip(snn_l, se_l))
            self._pair_sigs = [signature(dict(snn=s, sigma_e=e, n_comps=n_comps))
                            for s, e in self._pairs]

        self.params_C = dict(snn=snn, sigma_e=sigma_e, n_comps=n_comps)
        self.params_E = {**self.params_C, "alpha": alpha}
        self.sig_C = signature(self.params_C)
        self.sig_E = signature(self.params_E)
        self.spatial_domains_sig = None



    def fit(self, adata,add_layer=False):
        if self.verbose:
            print('fit')
        self.ensure_weights(adata)
        self.smooth_and_pca(adata, "w", "E",add_layer=add_layer)
        self.maybe_save(adata, "E_pca")
        return adata

    def spatial_domains(self, adata, n_domains="ground_truth",
                clustering="gmm", seed=0):
        
        self.init_spatial_domains(adata, n_domains, clustering, seed)
        if not self.maybe_load(adata, "spatial_domains"):
            if "E_pca" not in adata.obsm:
                if not self.maybe_load(adata, "E_pca"):
                    # self.fit(adata)
                    self.spatial_domains_embedding(adata,add_layer=True)
                    pca_of_mat(adata,adata.layers['E'],n_comps=self.n_comps,key='E')
            self.compute_spatial_domains(adata)
        return adata

    def weights(self, adata):
        self.ensure_weights(adata)
        # return adata

    def niche(self, adata, cell_type_key="global_cell_type",mask_column=None,exocrine=True,verbose=False,weights_key='wc'):
        verbose = self.verbose if verbose is None else verbose
        self.ensure_weights(adata)
    
        w = adata.obsm[weights_key]
        if exocrine:
            if mask_column is None:
                mask_column = cell_type_key
            w = cross_weights(w, adata.obs[mask_column])
        compute_niche_context(adata, w, cell_type_key=cell_type_key,verbose=verbose)



    def local_alignment(self,adata, layer_C="C"):
        from kontext.utils import zscore,to_dense
        """Per-cell context correlation (Pearson):.cosine similarity in PCA space."""
        Xz, Cz = zscore(to_dense(adata.X)), zscore(to_dense(adata.layers[layer_C]))
        num = (Xz * Cz).sum(1)
        den = np.sqrt((Xz**2).sum(1) * (Cz**2).sum(1)) + 1e-10
        adata.obs["local_alignment"] = np.asarray((num / den).astype(np.float32)).ravel()


# ---- computation (private) ----

    def ensure_weights(self, adata):
        if "wc" in adata.obsm:
            return
        if not self.maybe_load(adata, "weights"):
            self.compute_weights(adata)

    def compute_weights(self, adata):
        if self._is_list:
            wc = None
            ws = None
            for (snn_i, se_i), sig_i in zip(self._pairs, self._pair_sigs):
                wc_i = self._load_intermediate(adata, sig_i)
                if wc_i is None:
                    ws_i = spatial_kernel(adata.obsm["spatial"], snn_i, sparse=True)
                    we = expression_kernel_at_indices(adata.X, ws_i, se_i,chunk=self.chunk)
                    wc_i = ws_i.multiply(we)
                    if self.verbose:
                        print(ws_i.shape,we.shape,'memory of wc_i : ',memory_of(wc_i))
                    self._save_intermediate(adata, wc_i, sig_i)
                ws = ws_i if ws is None else ws + ws_i
                wc = wc_i if wc is None else wc + wc_i
            adata.obsm["ws"] = ws
            adata.obsm["wc"] = wc
            if self.alpha == 1:
                adata.obsm['w'] = sps.identity(len(adata)).tocsr()
            else:
                adata.obsm["w"] = add_diagonal(wc, self.alpha/(1-self.alpha))
            if self.verbose:
                print('memory of w : ',memory_of(adata.obsm['w']))
            self.maybe_save(adata, "weights")
            return
        # --- original scalar path (unchanged) ---
        w = None
        if self.snn is not None:
            ws = spatial_kernel(adata.obsm["spatial"], self.snn, sparse=True)
            adata.obsm["ws"] = ws
            w = ws
        if self.sigma_e is not None:
            we = expression_kernel_at_indices(adata.X, ws, self.sigma_e,chunk=self.chunk)
            adata.obsm["we"] = we
            w = we
        if self.snn is not None and self.sigma_e is not None:
            w = ws.multiply(we)
        adata.obsm["wc"] = w
        if self.alpha == 1:
            adata.obsm['w'] = sps.identity(len(adata)).tocsr()
        else:
            adata.obsm["w"] = add_diagonal(w, self.alpha/(1-self.alpha))

        if self.verbose:
            print('memory of w : ',memory_of(adata.obsm['w']))
        self.maybe_save(adata, "weights")


    def microenvironment_embedding(self,adata,add_layer=True):
        C = neighborhood_product(adata.X, adata.obsm['wc'])
        if sps.issparse(adata.X):
            C = sps.csr_matrix(C)
        if add_layer:
            adata.layers['C'] = C

    def spatial_domains_embedding(self,adata,add_layer=True):
        E = neighborhood_product(adata.X, adata.obsm['w'])
        if sps.issparse(adata.X):
            E = sps.csr_matrix(E)
        if add_layer:
            adata.layers['E'] = E


    def smooth_and_pca(self, adata, weight_key, name,add_layer=False,pca=True):
        result = neighborhood_product(adata.X, adata.obsm[weight_key])
        if sps.issparse(adata.X):
            result = sps.csr_matrix(result)
        if add_layer:
            adata.layers[name] = result
        if self.verbose:
            print(f'memory of layer {name} : ',memory_of(result),result.shape)

        if pca:
            pca_of_mat(adata, result, n_comps=self.n_comps, key=name)
        if self.verbose:
            print(f'memory of {name}_pca: ',memory_of(adata.obsm[f'{name}_pca']))
            


    def compute_spatial_domains(self, adata, matrix_key="E_pca"):
        fn = CLUSTERERS[self.spatial_domains_params["clustering"]]
        params = {k: v for k, v in self.spatial_domains_params.items() if k != "clustering"}
        labels = fn(matrix=adata.obsm[matrix_key], **params)

        adata.obs["spatial_domains"] = pd.Categorical(labels)
        adata.obs[self.spatial_domains_signature] = pd.Categorical(labels)
        self.maybe_save(adata, "spatial_domains")



    def init_spatial_domains(self, adata, n_domains='ground_truth', clustering='gmm', seed=0):
        if n_domains == "ground_truth" and "n_domains" in adata.uns.get("data", {}):
            n_domains = adata.uns["data"]["n_domains"]
        self.spatial_domains_params = dict(clustering=clustering, seed=seed,
                                 n_domains=n_domains)
        self.spatial_domains_sig = signature(self.spatial_domains_params)
        self.spatial_domains_signature = f'{self.sig_E}_{self.spatial_domains_sig}'
        adata.uns['signature'] = self.spatial_domains_signature


    def spatial_domains_of_merged_adata(self,adata,n_domains, clustering, seed):
        print('pca of merged adata')
        if 'C_pca' in adata.obsm:
            adata.obsm.pop('C_pca')
        if 'E_pca' in adata.obsm:
            adata.obsm.pop('E_pca')

        pca_of_mat(adata,adata.layers['E'],n_comps=self.n_comps,key='E')
        pca_of_mat(adata,adata.layers['C'],n_comps=self.n_comps,key='C')

        
        print('spatial domains of merged adata')
        self.init_spatial_domains(adata,n_domains=n_domains, clustering=clustering, seed=seed)
        loaded = self.load_spatial_domains(adata)
        if not loaded:
            print('not loaded, computing spatial domains')
            self.compute_spatial_domains(adata,matrix_key='E_pca')




    # ---- generic save / load ----

    def maybe_save(self, adata, key):
        if self.save.get(key, False):
            self._IO[key]["save"](self, adata)

    def maybe_load(self, adata, key):
        if not self.load.get(key, False):
            return False
        try:
            return self._IO[key]["load"](self, adata)
        except Exception as e:
            log.debug("Could not load %s: %s", key, e)
            return False



    # ---- concrete I/O ----

    def save_weights(self, adata):
        sps.save_npz(self.path(adata, "weights"), sps.csr_matrix(adata.obsm["wc"]))

    def load_weights(self, adata):
        p = self.path(adata, "weights")
        if not p.exists():
            return False
        adata.obsm["wc"] = sps.load_npz(p)
        if self.alpha == 1:
            adata.obsm['w'] = sps.identity(len(adata)).tocsr()
        else:
            adata.obsm["w"] = add_diagonal(adata.obsm["wc"], self.alpha/(1-self.alpha))
        return True

    def save_pca(self, adata, key):
        pd.DataFrame(adata.obsm[key], index=adata.obs.index).to_csv(self.path(adata, key))

    def load_pca(self, adata, key):
        p = self.path(adata, key)
        if not p.exists():
            return False
        adata.obsm[key] = pd.read_csv(p, index_col=0).to_numpy()
        return True

    def save_spatial_domains(self, adata):
        adata.obs["spatial_domains"].to_csv(self.path(adata, "spatial_domains"))

    def load_spatial_domains(self, adata,signed=False):
        p = self.path(adata, "spatial_domains")
        try:
            df = pd.read_csv(p, index_col=0)
            if len(df) != len(adata):
                log.warning("Cached spatial_domains size mismatch")
                return False
            adata.obs["spatial_domains"] = pd.Categorical(df.iloc[:, 0])
            if signed:
                signature =str(p).split('/')[-1].replace('.csv','')
                adata.obs[signature] = pd.Categorical(df.iloc[:, 0])
                adata.uns['signature'] = signature
            return True
        except Exception as e:
            if self.verbose:
                print(e)
                print(p)        
            return False
            
    def save_scores(self, adata):
        sig = adata.uns['kontext']['signature']
        out = adata.uns['kontext'][sig]
        p = self.path(adata, "scores")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)

    def load_scores(self, adata):
        p = self.path(adata, "scores")
        try:
            with open(p, "r") as f:
                data = json.load(f)
            # Restore the same state as scores() creates
            if 'kontext' not in adata.uns:
                adata.uns['kontext'] = {}
            # print(data)
            signature = f"{self.sig_E}_{self.spatial_domains_sig}"
            adata.uns['kontext']['signature'] = signature
            adata.uns['kontext'][signature] = {
                'params_E': data['params_E'],
                'spatial_domains_params': data['spatial_domains_params'],
                'scores': data['scores']
            }
            return data
        except Exception as e:
            if self.verbose:
                print(f'scores not loaded {e} {p}')
                print(e)
                print(p)
            return None

    
    # dispatch table
    _IO = {
        "weights":    {"save": save_weights,    "load": load_weights},
        "E_pca":      {"save": lambda s, a: s.save_pca(a, "E_pca"),
                        "load": lambda s, a: s.load_pca(a, "E_pca")},
        "C_pca":      {"save": lambda s, a: s.save_pca(a, "C_pca"),
                        "load": lambda s, a: s.load_pca(a, "C_pca")},
        "E_umap":      {"save": lambda s, a: s.save_pca(a, "E_umap"),
                        "load": lambda s, a: s.load_pca(a, "E_umap")},
        "C_umap":      {"save": lambda s, a: s.save_pca(a, "C_umap"),
                        "load": lambda s, a: s.load_pca(a, "C_umap")},
        "spatial_domains": {"save": save_spatial_domains, "load": load_spatial_domains},
        "scores": {"save": save_scores, "load": load_scores},
    }

    # ---- paths ----

    def base(self, adata):
        d = adata.uns["data"]
        return Path(f"/pasteur/appa/scratch/aozierla/"
                    f"{d['dataset']}/{d['sample']}/{d['pp_str']}/context")

    def path(self, adata, kind):
        base = self.base(adata)
        mapping = {
            "weights":    (base / "weights",    f"{self.sig_C}.npz"),
            "E_pca":      (base / "pca_E",      f"{self.sig_E}.csv"),
            "C_pca":      (base / "pca_C",      f"{self.sig_C}.csv"),
            "E_umap":      (base / "umap_E",      f"{self.sig_E}.csv"),
            "C_umap":      (base / "umap_C",      f"{self.sig_C}.csv"),
            "spatial_domains": (base / "spatial_domains",
                        f"{self.sig_E}_{self.spatial_domains_sig}.csv"),
            "scores":     (base / "scores",     f"{self.sig_E}_{self.spatial_domains_sig}.json"),
        }
        folder, fname = mapping[kind]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / fname
    
    def _intermediate_path(self, adata, sig):
        folder = self.base(adata) / "weights"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{sig}.npz"

    def _load_intermediate(self, adata, sig):
        if not self.load.get("weights", False):
            return None
        p = self._intermediate_path(adata, sig)
        try:
            mat = sps.load_npz(p) 
            return mat
        except:
            return None


    def _save_intermediate(self, adata, wc, sig):
        if self.save.get("weights", False):
            sps.save_npz(self._intermediate_path(adata, sig), sps.csr_matrix(wc))



    def cell_cell_communication(self,adata,pairs,mode='mean'):
        lr_normalize(adata, pairs)
        lr_availability(adata, None)
        lr_moments(adata, 'X_lr')
        lr_moments(adata, 'A_all')
        lr_moments(adata, 'P_all')
        lr_cell_score(adata, 'all', mode=mode, col='cell_communication')  


    def init_sender_receiver_communication(self,adata,pairs,sender,receiver,groups):
        lr_normalize(adata, pairs)
        lr_availability(adata, receiver, groups)
        lr_availability(adata, sender, groups)
        r = {'col':groups,'value':receiver}
        lr_moments(adata, 'X_lr',r=r)
        
        lr_moments(adata, f'A_{sender}', om=(adata.obs[f'm_{sender}'] > 0).values,r=r)
        lr_moments(adata, f'P_{sender}', om=(adata.obs[f'm_{sender}'] > 0).values,r=r)


    def sender_receiver_scores(self,adata,pairs,sender,receiver,groups,mode='mean',receiver_threshold=0.1,group_by='spatial_domains',by_order=None,
                            genes_of_interest=None,log = True):
        import numpy as np
        r = {'col': groups, 'value': receiver}
        S = lr_score(adata, sender, receiver, groups, mode, r=r,src='P')
        gi = pd.Index(adata.uns['lr']['genes'])
        pos = adata.obs_names.get_indexer(S.index)
        det = pd.Series(np.asarray((adata.obsm['X_lr'][pos][:, gi.get_indexer(pairs.receptor)] > 0).mean(0)).ravel(), index=pairs.index)
        keep = det[det > receiver_threshold].index
        T = lr_aggregate(adata, S, sender, by=group_by)['all']
        if by_order is not None:
            T = T.reindex(by_order)
        if genes_of_interest is None:
            genes_of_interest = pairs.index.tolist()
        T = T[[g for g in genes_of_interest if g in keep and g in S.columns]]
        if log:
            T = lr_log(T)
        return(T)