import scanpy as sc
import cupy as cp
import time
import rapids_singlecell as rsc
import warnings
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
import gc

# Initialize RMM for managed memory to handle large datasets
rmm.reinitialize(
    managed_memory=True,  # Allows oversubscription
    pool_allocator=False,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)
warnings.filterwarnings("ignore")

# Input data
adata = sc.read("/home/orion1016/data/1M_brain_cells_10X.sparse.h5ad")

cell_counts = [10000, 70000, 100000, 200000, 500000]

# Initialize a dictionary to store times
times = {n_cells: {} for n_cells in cell_counts}

for USE_FIRST_N_CELLS in cell_counts:
    print(f"Processing {USE_FIRST_N_CELLS} cells...")
    start = time.time()

    # Load and Preprocess Data
    load_start = time.time()

    adata.var_names_make_unique()
    adata_subset = adata[:USE_FIRST_N_CELLS, :].copy()
    rsc.get.anndata_to_GPU(adata_subset)

    load_time = time.time()
    times[USE_FIRST_N_CELLS]["Loading data"] = load_time - load_start
    print("Loading data time: %s" % (load_time - load_start))

    # Preprocess
    preprocess_start = time.time()

    rsc.pp.flag_gene_family(adata_subset, gene_family_name="MT", gene_family_prefix="mt-")
    rsc.pp.calculate_qc_metrics(adata_subset, qc_vars=["MT"])
    adata_subset = adata_subset[
        (adata_subset.obs["n_genes_by_counts"] < 5000) &
        (adata_subset.obs["n_genes_by_counts"] > 500) &
        (adata_subset.obs["pct_counts_MT"] < 20)
    ].copy()
    gc.collect()
    rsc.pp.filter_genes(adata_subset, min_count=3)
    adata_subset.layers["counts"] = adata_subset.X.copy()
    rsc.pp.normalize_total(adata_subset, target_sum=1e4)
    rsc.pp.log1p(adata_subset)
    rsc.pp.highly_variable_genes(
        adata_subset, n_top_genes=5000, flavor="seurat_v3", layer="counts"
    )
    adata_subset = adata_subset[:, adata_subset.var["highly_variable"]]

    # Regress out confounding factors
    regress_start = time.time()

    rsc.pp.regress_out(adata_subset, keys=["total_counts", "pct_counts_MT"])

    regress_time = time.time()
    times[USE_FIRST_N_CELLS]["Regression"] = regress_time - regress_start
    print("Regress out confounding factors time: %s" % (regress_time - regress_start))

    # Scale
    scale_start = time.time()

    rsc.pp.scale(adata_subset, max_value=10)

    scale_time = time.time()
    times[USE_FIRST_N_CELLS]["Scale"] = scale_time - scale_start
    print("Scaling time: %s" % (scale_time - scale_start))

    preprocess_time = time.time()
    times[USE_FIRST_N_CELLS]["Preprocessing"] = preprocess_time - preprocess_start
    print("Preprocessing time: %s" % (preprocess_time - preprocess_start))

    # PCA
    pca_start = time.time()

    rsc.pp.pca(adata_subset, n_comps=100, use_highly_variable=False)

    pca_time = time.time()
    times[USE_FIRST_N_CELLS]["PCA"] = pca_time - pca_start
    print("PCA time: %s" % (pca_time - pca_start))

    # t-SNE
    tsne_start = time.time()

    rsc.tl.tsne(adata_subset, n_pcs=40)

    tsne_time = time.time()
    times[USE_FIRST_N_CELLS]["t-SNE"] = tsne_time - tsne_start
    print("t-SNE time: %s" % (tsne_time - tsne_start))

    # K-means
    kmeans_start = time.time()

    rsc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=50)
    
    kmeans_time = time.time()
    times[USE_FIRST_N_CELLS]["K-means"] = kmeans_time - kmeans_start
    print("K-means time: %s" % (kmeans_time - kmeans_start))

    # UMAP
    umap_start = time.time()

    rsc.tl.umap(adata_subset, min_dist=0.3)

    umap_time = time.time()
    times[USE_FIRST_N_CELLS]["UMAP"] = umap_time - umap_start
    print("UMAP time: %s" % (umap_time - umap_start))

    # Clustering
    clustering_start = time.time()

    rsc.tl.leiden(adata_subset, resolution=1.0)

    clustering_time = time.time()
    times[USE_FIRST_N_CELLS]["Clustering"] = clustering_time - clustering_start
    print("Clustering time: %s" % (clustering_time - clustering_start))

    # Differential expression analysis
    de_start = time.time()

    rsc.tl.rank_genes_groups_logreg(adata_subset, groupby="leiden", use_raw=False)

    de_time = time.time()
    times[USE_FIRST_N_CELLS]["DE analysis"] = de_time - de_start
    print("Differential expression analysis time: %s" % (de_time - de_start))

    full_time = time.time() - start
    times[USE_FIRST_N_CELLS]["Full run"] = full_time
    print("Full run time: %s" % full_time)

# Print all recorded times
for n_cells, timing in times.items():
    print(f"Times for {n_cells} cells:")
    for step, t in timing.items():
        print(f"{step}: {t:.2f} seconds")
