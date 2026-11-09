"""Known-answer tests on small synthetic data (runs in about a minute on a laptop)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import viewmastr as vm
from viewmastr._data import size_factors
from viewmastr._rrandom import r_sample


def synthetic(n_per_type: int, seed: int, n_genes: int = 300, n_types: int = 4) -> ad.AnnData:
    """Poisson counts; each type has 20 marker genes at 8x the background rate."""
    rng = np.random.default_rng(seed)
    rate = np.full((n_types, n_genes), 0.5)
    for t in range(n_types):
        rate[t, t * 20:(t + 1) * 20] = 4.0
    y = np.repeat(np.arange(n_types), n_per_type)
    depth = rng.uniform(0.5, 2.0, len(y))[:, None]
    x = rng.poisson(rate[y] * depth)
    obs = pd.DataFrame({"celltype": [f"type{t}" for t in y]}, index=[f"c{seed}_{i}" for i in range(len(y))])
    return ad.AnnData(sp.csr_matrix(x.astype(np.float32)), obs=obs,
                      var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))


@pytest.fixture(scope="module")
def data():
    return synthetic(150, seed=1), synthetic(80, seed=2)


def test_r_sample_matches_r():
    # R >= 3.6: set.seed(1); sample(10, 5) gives 9 4 7 1 2
    assert list(r_sample(10, 5, seed=1) + 1) == [9, 4, 7, 1, 2]


def test_size_factors_are_geometric_mean_scaled():
    lib = np.array([1.0, 4.0, 16.0])
    assert np.allclose(size_factors(lib), [0.25, 1.0, 4.0])


@pytest.mark.parametrize("func,hidden", [("mlr", ()), ("nn", (32,)), ("nn", (32, 16)), ("nb", ())])
def test_known_answer(data, func, hidden, tmp_path):
    ref, query = data
    q = query.copy()
    vm.viewmastR(q, ref, "celltype", func=func, hidden_layers=hidden, max_epochs=10, seed=7,
                 workers=1, model_dir=tmp_path, verbose=False)
    accuracy = (q.obs["viewmastR_pred"].astype(str) == q.obs["celltype"]).mean()
    assert accuracy > 0.95


def test_seeded_single_worker_training_is_reproducible(data, tmp_path):
    ref, query = data
    runs = [vm.viewmastR(query.copy(), ref, "celltype", seed=3, workers=1, max_epochs=3,
                         model_dir=tmp_path / str(i), return_type="list", verbose=False) for i in range(2)]
    assert np.array_equal(runs[0].logits.to_numpy(), runs[1].logits.to_numpy())


def test_infer_reproduces_training_labels(data, tmp_path):
    ref, query = data
    res = vm.viewmastR(query.copy(), ref, "celltype", seed=5, max_epochs=5, model_dir=tmp_path,
                       return_type="list", return_probs=True, verbose=False)
    inf = vm.viewmastR_infer(query.copy(), tmp_path, return_type="list", chunk_size=37)
    assert np.allclose(res.logits.to_numpy(), inf.logits.to_numpy(), atol=1e-5)
    assert vm.get_weights(tmp_path).shape == (len(res.adata.var_names), 4)


def test_infer_is_linear_in_genes(tmp_path):
    # A per-gene rebuild of set(query.var_names) made this O(genes^2): ~40 s at 6,000 genes.
    import time
    ref, query = synthetic(20, seed=1, n_genes=6000), synthetic(5, seed=2, n_genes=6000)
    vm.viewmastR(query.copy(), ref, "celltype", max_epochs=1, workers=1, model_dir=tmp_path, verbose=False)
    start = time.perf_counter()
    vm.viewmastR_infer(query.copy(), tmp_path)
    assert time.perf_counter() - start < 5


def test_split_follows_r(data):
    ref, _ = data
    ts = vm.setup_training(None, ref, "celltype", seed=11)
    assert np.array_equal(ts.train_idx, r_sample(ref.n_obs, round(0.8 * ref.n_obs), 11))
    assert len(ts.train_idx) + len(ts.test_idx) == ref.n_obs


def test_feature_selection():
    # 500 genes with rates spread over two orders of magnitude; 10 genes are on in one
    # group of cells only, so their dispersion sits far above the mean-dispersion trend.
    rng = np.random.default_rng(3)
    rate = np.tile(np.exp(rng.uniform(np.log(0.05), np.log(5), 500)), (1000, 1))
    rate[:500, :10] *= 20
    rate[500:, :10] *= 0.05
    a = ad.AnnData(sp.csr_matrix(rng.poisson(rate).astype(np.float32)),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(500)]))
    vm.select_features(vm.calculate_feature_dispersion(a), top_n=10)
    sel = vm.get_selected_features(a)
    assert len(sel) == 10
    assert len(set(sel) & {f"g{i}" for i in range(10)}) >= 9


def test_splat_is_seeded_and_keeps_bulk_profile(data):
    ref, query = data
    bulk = ad.AnnData(sp.csr_matrix(np.vstack([ref[ref.obs.celltype == t].X.sum(0) for t in ["type0", "type1"]])),
                      obs=pd.DataFrame({"celltype": ["type0", "type1"]}, index=["b0", "b1"]),
                      var=ref.var.copy())
    a = vm.splat_bulk_reference(query, bulk, N=50, seed=1)
    b = vm.splat_bulk_reference(query, bulk, N=50, seed=1)
    assert a.shape == (100, ref.n_vars) and (a.X != b.X).nnz == 0
    markers0 = np.asarray(a.X[:50, :20].sum()) / a.X[:50].sum()
    assert markers0 > 0.2  # type0 cells carry the type0 markers


def test_augment_balances_classes(data):
    ref, _ = data
    small = ref[np.r_[0:150, 150:170, 300:450, 450:600]].copy()
    out = vm.augment_data(small, "celltype", norm_number=150, seed=1)
    assert (out.obs["celltype"].value_counts() == 150).all()


@pytest.mark.parametrize("method", ["gd", "em"])
def test_deconvolution_recovers_mixture(method):
    rng = np.random.default_rng(0)
    genes, types = 400, 3
    sig = pd.DataFrame(rng.gamma(0.5, 1.0, (genes, types)), index=[f"g{i}" for i in range(genes)],
                       columns=["A", "B", "C"])
    sig /= sig.sum(axis=0)
    truth = np.array([[0.6, 0.1], [0.3, 0.2], [0.1, 0.7]])
    lengths = np.full(genes, 500.0)  # length / insert size = 1: reads = molecules
    bulk = pd.DataFrame(rng.poisson(sig.to_numpy() @ truth * 2e5), index=sig.index, columns=["s1", "s2"])
    res = vm.deconvolve_bulk(sig, bulk, lengths, method=method, max_iter=3000, seed=1, verbose=False)
    assert np.allclose(res.proportions.to_numpy(), truth, atol=0.05)


def test_palette_aliases_and_set_colors():
    assert vm.palette.color("11_CD14.Mono") == vm.palette.color("Mono")
    assert vm.palette.color("Late.Eryth") == vm.palette.COLORS["Ery"]
    assert vm.palette.color("no such type") == vm.palette.COLORS["Other"]
    assert not vm.palette.known("no such type")
    a = ad.AnnData(np.zeros((3, 1)), obs=pd.DataFrame({"t": ["HSC", "B", "HSC"]}, index=list("abc")))
    vm.palette.set_colors(a, "t")
    assert list(a.uns["t_colors"]) == [vm.palette.color(c) for c in a.obs["t"].cat.categories]
    assert len(set(vm.palette.COLORS.values())) == len(vm.palette.COLORS)
    bonemarrowmap_broad = [
        "B", "CD4 Memory T", "CD8 Memory T", "Cycling Progenitor", "Early Erythroid", "Early GMP",
        "Early Lymphoid", "EoBasoMast Precursor", "HSC MPP", "LMPP", "Late Erythroid", "Late GMP",
        "MEP", "Megakaryocyte Precursor", "Monocyte", "NK", "Naive T", "Plasma_Cell", "Pre-B",
        "Pro-B", "Pro-Monocyte", "Stromal", "cDC", "pDC"]
    assert all(vm.palette.known(k) for k in bonemarrowmap_broad)
    assert len({vm.palette.color(k) for k in bonemarrowmap_broad}) == len(bonemarrowmap_broad)


def test_confusion_matrix_accuracy_only_for_one_vocabulary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    same = vm.confusion_matrix(["a", "b", "b"], ["a", "b", "a"])
    assert "accuracy 66.67%" in same.figure.axes[-1].get_ylabel()
    mixed = vm.confusion_matrix(["a", "x", "y"], ["a", "b", "c"])  # shares only "a"
    assert "accuracy" not in mixed.figure.axes[-1].get_ylabel()
    plt.close("all")


def test_order_diagonal():
    from viewmastr.plotting import order_diagonal
    tab = pd.crosstab(pd.Series(["b", "a", "c", "c"]), pd.Series(["c", "a", "b", "b"]))
    same = order_diagonal(tab, rows=["a", "b", "c"])
    assert list(same.index) == list(same.columns) == ["a", "b", "c"]
    # Two vocabularies: each column goes to the row holding most of its cells
    tab = pd.crosstab(pd.Series(["a", "a", "b", "c", "c"]), pd.Series(["Z", "Z", "X", "Y", "Y"]))
    assert list(order_diagonal(tab, rows=["c", "b", "a"]).columns) == ["Y", "X", "Z"]


def test_uses_counts_only_not_embeddings(data, tmp_path):
    """Predictions depend on the counts matrix and on nothing in obsm/obsp/varm/uns/layers."""
    ref, query = data

    def run(q, r, name):
        res = vm.viewmastR(q, r, "celltype", seed=5, workers=1, max_epochs=3, model_dir=tmp_path / name,
                           return_type="list", verbose=False)
        return res.probs.to_numpy()

    rng = np.random.default_rng(0)

    def decorate(a):
        a = a.copy()
        a.obsm["X_pca"] = rng.normal(size=(a.n_obs, 10))
        a.obsm["X_umap"] = rng.normal(size=(a.n_obs, 2))
        a.obsp["connectivities"] = sp.random(a.n_obs, a.n_obs, density=0.05, format="csr", random_state=1)
        a.varm["PCs"] = rng.normal(size=(a.n_vars, 10))
        a.uns["pca"] = {"variance": rng.random(10)}
        a.layers["counts"] = sp.csr_matrix(rng.poisson(1.0, a.shape).astype(np.float32))  # decoy
        return a

    base = run(query.copy(), ref, "base")
    # negative control: random embeddings, graphs and a decoy layer change nothing
    assert np.array_equal(run(decorate(query), decorate(ref), "decorated"), base)
    # positive control: shuffling genes within each query cell changes the predictions
    q = query.copy()
    x = q.X.toarray()
    for row in x:
        rng.shuffle(row)
    q.X = sp.csr_matrix(x)
    assert not np.allclose(run(q, ref, "shuffled"), base)
