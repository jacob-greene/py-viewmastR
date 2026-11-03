//! `viewmastr._core`: the Python binding of the viewmastR burn backend.
//!
//! The numerical code lives in `classifier`, `splat` and `deconv`; this file only moves
//! numpy arrays in and out and releases the GIL while Rust works.

mod classifier;
mod deconv;
mod splat;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Rows of a dense (cells x features) array as training items.
fn items(x: &PyReadonlyArray2<f32>, y: &PyReadonlyArray1<i32>) -> PyResult<Vec<classifier::Item>> {
    let (x, y) = (x.as_array(), y.as_array());
    if x.nrows() != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("x and y differ in length"));
    }
    Ok(x.outer_iter().zip(y.iter()).map(|(r, &y)| classifier::Item { x: r.to_vec(), y }).collect())
}

fn mat(a: &PyReadonlyArray2<f32>) -> deconv::Mat {
    let a = a.as_array();
    deconv::Mat { data: a.iter().copied().collect(), rows: a.nrows(), cols: a.ncols() }
}

/// Train an mlr (`hidden=[]`) or perceptron (`hidden=[h]` / `[h1, h2]`) classifier.
///
/// Returns a dict: `weights` = list of (W [in, out], b [out]) float32 arrays,
/// `history` = dict of per-epoch train/test accuracy and loss, `seconds`.
#[pyfunction]
#[pyo3(signature = (train_x, train_y, test_x, test_y, n_classes, hidden, lr, epochs,
                    batch_size=64, num_workers=16, seed=None, model_path=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn train_classifier<'py>(
    py: Python<'py>,
    train_x: PyReadonlyArray2<f32>,
    train_y: PyReadonlyArray1<i32>,
    test_x: PyReadonlyArray2<f32>,
    test_y: PyReadonlyArray1<i32>,
    n_classes: usize,
    hidden: Vec<usize>,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    num_workers: usize,
    seed: Option<u64>,
    model_path: Option<String>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (train, test) = (items(&train_x, &train_y)?, items(&test_x, &test_y)?);
    let cfg = classifier::TrainConfig { hidden, n_classes, lr, epochs, batch_size, num_workers, seed, model_path, verbose };
    let fit = py.allow_threads(|| classifier::train(train, test, &cfg));

    let out = PyDict::new(py);
    let weights = fit
        .weights
        .into_iter()
        .map(|(w, dims, b)| {
            let w = numpy::ndarray::Array2::from_shape_vec((dims[0], dims[1]), w).unwrap();
            (w.into_pyarray(py), b.into_pyarray(py))
        })
        .collect::<Vec<_>>();
    out.set_item("weights", weights)?;
    let h = PyDict::new(py);
    h.set_item("train_acc", fit.history.train_acc)?;
    h.set_item("test_acc", fit.history.test_acc)?;
    h.set_item("train_loss", fit.history.train_loss)?;
    h.set_item("test_loss", fit.history.test_loss)?;
    out.set_item("history", h)?;
    out.set_item("seconds", fit.seconds)?;
    Ok(out)
}

/// Simulate `n_cells` single cells per bulk sample. `counts` is genes x samples.
/// Returns COO (rows, cols, values) over genes x (samples * n_cells).
#[pyfunction]
#[pyo3(signature = (counts, sizes, bandwidth, n_cells, replace, seed=42))]
fn splat_bulk<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<i64>,
    sizes: Vec<f64>,
    bandwidth: f64,
    n_cells: usize,
    replace: bool,
    seed: u64,
) -> (Bound<'py, numpy::PyArray1<i64>>, Bound<'py, numpy::PyArray1<i64>>, Bound<'py, numpy::PyArray1<f64>>) {
    let c = counts.as_array();
    let per_sample: Vec<Vec<usize>> =
        c.columns().into_iter().map(|col| col.iter().map(|&v| v.max(0) as usize).collect()).collect();
    let (r, cc, v) = py.allow_threads(|| splat::splat(&per_sample, &sizes, bandwidth, n_cells, replace, seed));
    (r.into_pyarray(py), cc.into_pyarray(py), v.into_pyarray(py))
}

/// Gradient-descent deconvolution (see `deconv.rs`). Returns (exposures, predicted counts).
#[pyfunction]
#[pyo3(signature = (signatures, bulk, gene_lengths, gene_weights, insert_size, init_log_exp, lr,
                    l1_lambda, l2_lambda, max_iter, poll_interval, ll_tolerance, sparsity_tolerance,
                    seed=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn deconvolve_gd<'py>(
    py: Python<'py>,
    signatures: PyReadonlyArray2<f32>,
    bulk: PyReadonlyArray2<f32>,
    gene_lengths: Vec<f32>,
    gene_weights: Vec<f32>,
    insert_size: f32,
    init_log_exp: f32,
    lr: f64,
    l1_lambda: f32,
    l2_lambda: f32,
    max_iter: usize,
    poll_interval: usize,
    ll_tolerance: f32,
    sparsity_tolerance: f32,
    seed: Option<u64>,
    verbose: bool,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let (s, k) = (mat(&signatures), mat(&bulk));
    let cfg = deconv::GdConfig {
        insert_size, init_log_exposure: init_log_exp, learn_rate: lr, l1_lambda, l2_lambda,
        max_iterations: max_iter, poll_interval, ll_tolerance, sparsity_tolerance, seed, verbose,
    };
    let (e, p) = py.allow_threads(|| deconv::deconvolve_gd(&s, &k, &gene_lengths, &gene_weights, &cfg));
    as_pair(py, e, s.cols + 1, p, k.rows, k.cols)
}

/// EM deconvolution (see `deconv.rs`). Returns (exposures, predicted molecules).
#[pyfunction]
#[pyo3(signature = (signatures, bulk, gene_weights, max_iter, tolerance, l1_lambda, verbose=false))]
fn deconvolve_em<'py>(
    py: Python<'py>,
    signatures: PyReadonlyArray2<f32>,
    bulk: PyReadonlyArray2<f32>,
    gene_weights: Vec<f32>,
    max_iter: usize,
    tolerance: f32,
    l1_lambda: f32,
    verbose: bool,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let (s, k) = (mat(&signatures), mat(&bulk));
    let cfg = deconv::EmConfig { max_iterations: max_iter, tolerance, l1_lambda, verbose };
    let (e, p) = py.allow_threads(|| deconv::deconvolve_em(&s, &k, &gene_weights, &cfg));
    as_pair(py, e, s.cols + 1, p, k.rows, k.cols)
}

fn as_pair(
    py: Python<'_>,
    e: Vec<f32>,
    e_rows: usize,
    p: Vec<f32>,
    p_rows: usize,
    n: usize,
) -> (Bound<'_, PyArray2<f32>>, Bound<'_, PyArray2<f32>>) {
    let e = numpy::ndarray::Array2::from_shape_vec((e_rows, n), e).unwrap();
    let p = numpy::ndarray::Array2::from_shape_vec((p_rows, n), p).unwrap();
    (e.into_pyarray(py), p.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(splat_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(deconvolve_gd, m)?)?;
    m.add_function(wrap_pyfunction!(deconvolve_em, m)?)?;
    Ok(())
}
