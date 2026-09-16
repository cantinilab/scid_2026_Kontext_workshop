from sklearn.preprocessing import normalize
import scanpy as sc 
import scipy.sparse as sps


def get_pp_str(preprocessing_params):
    from kontext.utils import format_key,format_value
    pp_str = ''
    if len(preprocessing_params) == 0:
        pp_str = 'ref_J'
    elif preprocessing_params['preprocessing'] is None:
        pp_str = 'raw'
    else: 
        pp_str = '_'.join([f'{format_key(k)}{format_value(v)}' for k,v in preprocessing_params.items()])
    return(pp_str)


def compute_hvg(adata,n_top_genes=2000,hvg_layer='norm',target_sum=None):
    adata_ = adata.copy()
    if hvg_layer=='norm':
        adata.X = adata.layers['counts']
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.highly_variable_genes(adata,flavor="seurat_v3",n_top_genes=n_top_genes)
    elif hvg_layer =='count': 
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='counts', n_top_genes=n_top_genes)
    else :
        print(f"compute_hvg: layer {hvg_layer} not known")
        raise
    adata_ = adata_[:,adata.var['highly_variable']]
    return(adata_)


def preprocess_adata(adata,preprocessing='jsPCA',
                    hvg=False,
                    hvg_layer='norm',
                    n_top_genes=2000,
                    target_sum=None,
                    verbose=False):


    adata = adata.copy()
    if preprocessing == None:
        if verbose:
            print('No preprocessing')
        pass        

    if preprocessing == 'BANKSY': # 2000,norm,None
        if verbose:
            print('pp BANKSY')
        sc.pp.normalize_total(adata,inplace=True,)
        adata = compute_hvg(adata,n_top_genes=2000,hvg_layer='norm',target_sum=None)

    if preprocessing == 'STAGATE': # 3000,norm,1e4
        if verbose:
            print('pp STAGATE')
        adata = compute_hvg(adata,n_top_genes=3000,hvg_layer='norm',target_sum=1e4)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)      
    
    if preprocessing == 'SEDR': # 2000,count,1e6
        if verbose: 
            print('pp SEDR')

        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        adata = compute_hvg(adata,n_top_genes=2000,hvg_layer='count',target_sum=None)
        sc.pp.scale(adata)

    if preprocessing == 'sparse':
        if verbose:
            print('')
        sc.pp.filter_genes(adata,min_cells=20)
        sc.pp.normalize_total(adata, target_sum=1e4)   # L1 / library-size scaling
        sc.pp.log1p(adata)                              # log1p on stored entries only
        adata.X = normalize(adata.X, norm="l2", axis=1) # unit-L2 rows (sklearn, sparse-safe)


    if preprocessing == 'sparse_hvg':
        if verbose:
            print('pp large2')
        sc.pp.filter_genes(adata,min_cells=20)
        sc.pp.normalize_total(adata, target_sum=1e4)   # L1 / library-size scaling
        adata = compute_hvg(adata,n_top_genes=n_top_genes,hvg_layer='norm',target_sum=None)
        sc.pp.log1p(adata)                              # log1p on stored entries only
        adata.X = normalize(adata.X, norm="l2", axis=1) # unit-L2 rows (sklearn, sparse-safe)

    if preprocessing == 'sparse_hvg_LR':
        from kontext.utils import get_LR_pairs

        print('pp large2hvg_LR')
        sc.pp.filter_genes(adata, min_cells=20)                                   # 1. filter
        sc.pp.normalize_total(adata, target_sum=1e4)                              # 3. normalize

        lr_pairs  = get_LR_pairs(adata, verbose=False, organism='human')          # 2. LR genes
        lr_genes  = set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])
        hvg_genes = set(compute_hvg(adata, n_top_genes=n_top_genes,
                                    hvg_layer='norm', target_sum=None).var_names)  #    + HVG
        print(len(hvg_genes))
        adata = adata[:, adata.var_names.isin(hvg_genes | lr_genes)].copy()       # 4. HVG ∪ LR

        sc.pp.log1p(adata)                                                        # 5. log1p
        adata.X = normalize(adata.X, norm="l2", axis=1)                           #    + L2

    if preprocessing == 'image':
        if verbose:
            print('pp xenium')
        sc.pp.log1p(adata)                              # log1p on stored entries only
        adata.X = normalize(adata.X, norm="l2", axis=1) # unit-L2 rows (sklearn, sparse-safe)
        adata.uns['preprocessed'] = True

    return(adata)




def merge_adatas(adatas: dict,label='sample',block_diag_keys=None) -> sc.AnnData:


    adata_list = [adata.copy() for adata in adatas.values()]
    block_diag_keys = block_diag_keys or []

    # Collect block-diagonal obsm entries before sc.concat strips them
    block_diag_mats = {key: [] for key in block_diag_keys}
    for adata in adata_list:
        for key in block_diag_keys:
            if key in adata.obsm:
                block_diag_mats[key].append(sps.csr_matrix(adata.obsm[key]))
            else:
                n = adata.n_obs
                block_diag_mats[key].append(sps.csr_matrix((n, n)))

    merged_adata = sc.concat(
        adata_list,
        label=label,
        keys=list(adatas.keys()),
        index_unique='-',
        join='inner'
    )

    # Attach the block-diagonal matrices to the merged object
    for key, mats in block_diag_mats.items():
        if len(mats) > 0:
            merged_adata.obsm[key] = sps.block_diag(mats, format='csr')

    merged_adata.uns['data'] = {
        'dataset': list(adatas.values())[0].uns['data']['dataset'],
        'sample': 'merged'
    }
    return merged_adata






