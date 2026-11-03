//! Bulk RNA-seq deconvolution into cell-type exposures.
//!
//! Ports of `signal.rs` (gradient descent, "gd") and `em.rs` (expectation maximisation,
//! "em") in furlan-lab/viewmastR (MIT licence), without the R glue. Model:
//!
//!   predicted molecules  q = [S | 1/G] x [E; intercept]
//!   predicted reads      y = q * (gene length / insert size)       (gd only)
//!   observed counts      k ~ Poisson(y), weighted per gene by w
//!
//! S is genes x cell types (columns sum to 1), E is cell types x samples.

use burn::{
    module::Param,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation, backend::AutodiffBackend, Distribution, TensorData},
};

type CpuAd = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
type Cpu = burn::backend::NdArray<f32>;

/// Dense row-major matrix handed over from numpy.
pub struct Mat {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

fn tensor2<B: Backend>(m: &Mat, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(m.data.clone(), [m.rows, m.cols]), device)
}

fn tensor1<B: Backend>(v: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), device)
}

fn to_vec<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

#[derive(Clone)]
pub struct GdConfig {
    pub insert_size: f32,
    pub init_log_exposure: f32,
    pub learn_rate: f64,
    pub l1_lambda: f32,
    pub l2_lambda: f32,
    pub max_iterations: usize,
    pub poll_interval: usize,
    pub ll_tolerance: f32,
    pub sparsity_tolerance: f32,
    pub seed: Option<u64>,
    pub verbose: bool,
}

#[derive(Module, Debug)]
struct GdModel<B: Backend> {
    log_exposures: Param<Tensor<B, 2>>,
    intercept: Param<Tensor<B, 2>>,
    // Fixed inputs, not trained.
    #[module(skip)]
    signatures_with_int: Tensor<B, 2>,
    #[module(skip)]
    conversion_factor: Tensor<B, 2>,
    #[module(skip)]
    observed_counts: Tensor<B, 2>,
    #[module(skip)]
    gene_weights_expanded: Tensor<B, 2>,
    #[module(skip)]
    k_log_k_minus_k: Tensor<B, 2>,
}

impl<B: Backend> GdModel<B> {
    fn random_log_exposures(dims: [usize; 2], cfg: &GdConfig, device: &B::Device) -> Param<Tensor<B, 2>> {
        Param::from_tensor(Tensor::random(dims, Distribution::Normal(cfg.init_log_exposure as f64, 0.1), device))
    }

    fn forward(&self) -> Tensor<B, 2> {
        let exposures = Tensor::cat(vec![self.log_exposures.val().exp(), self.intercept.val().exp()], 0);
        self.signatures_with_int.clone().matmul(exposures).mul(self.conversion_factor.clone())
    }

    /// Weighted Poisson negative log-likelihood per gene and sample (saturated model = 0).
    fn poisson_nll(&self, pred: Tensor<B, 2>) -> Tensor<B, 2> {
        let k = self.observed_counts.clone();
        let fit = k.mul(pred.clone().add_scalar(1e-10).log()).sub(pred);
        self.k_log_k_minus_k.clone().sub(fit).mul(self.gene_weights_expanded.clone())
    }

    /// NLL plus L1 and L2 penalties on all exposures, scaled by NLL / 100 as in R.
    fn loss(&self, cfg: &GdConfig) -> Tensor<B, 1> {
        let nll = self.poisson_nll(self.forward()).sum();
        let scale = nll.clone().into_scalar().elem::<f32>() / 100.0;
        let exposures = || Tensor::cat(vec![self.log_exposures.val().exp(), self.intercept.val().exp()], 0);
        let mut loss = nll;
        if cfg.l1_lambda != 0.0 {
            loss = loss.add(exposures().sum().mul_scalar(cfg.l1_lambda * scale));
        }
        if cfg.l2_lambda != 0.0 {
            loss = loss.add(exposures().powf_scalar(2.0).sum().mul_scalar(cfg.l2_lambda * scale));
        }
        loss
    }

    /// Soft count of active parameters (sigmoid of log exposures), the sparsity monitor.
    fn count_nonzero(&self) -> f32 {
        let all = Tensor::cat(vec![self.log_exposures.val(), self.intercept.val()], 0);
        activation::sigmoid(all).sum().into_scalar().elem::<f32>()
    }
}

/// Phase 1 ("null model"): exposures frozen near zero (log = -100), only the intercept is fit.
fn fit_null<B: AutodiffBackend>(model: &mut GdModel<B>, cfg: &GdConfig, device: &B::Device) {
    let dims = model.log_exposures.dims();
    model.log_exposures = Param::from_tensor(Tensor::zeros(dims, device).sub_scalar(100.0));
    let mut optim = AdamConfig::new().init();
    let mut last = f32::INFINITY;
    for iter in 0..cfg.max_iterations.min(10_000) {
        let loss = model.loss(cfg);
        let grads = GradientsParams::from_grads(loss.backward(), &*model);
        *model = optim.step(cfg.learn_rate, model.clone(), grads);
        if iter % cfg.poll_interval == 0 {
            let current = loss.into_scalar().elem::<f32>();
            if cfg.verbose && iter % 1000 == 0 {
                eprintln!("null iter {iter}: loss={current:.6}");
            }
            if iter > 100 && (last - current).abs() / current < 1e-8 {
                break;
            }
            last = current;
        }
    }
}

/// Phase 2: re-draw the exposures and fit all parameters with Adam until the relative
/// loss change and the sparsity change both fall below tolerance (checked every
/// `poll_interval` iterations after iteration 1000), or 10 checks without improvement.
fn fit_full<B: AutodiffBackend>(mut model: GdModel<B>, cfg: &GdConfig, device: &B::Device) -> GdModel<B> {
    model.log_exposures = GdModel::random_log_exposures(model.log_exposures.dims(), cfg, device);
    let mut optim = AdamConfig::new().init();
    let (mut last_loss, mut last_nz, mut stalls) = (f32::INFINITY, 0.0f32, 0);
    for iter in 0..cfg.max_iterations {
        let loss = model.loss(cfg);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(cfg.learn_rate, model, grads);
        if iter % cfg.poll_interval == 0 {
            let current = loss.into_scalar().elem::<f32>();
            let nz = model.count_nonzero();
            let (d_loss, d_nz) = (last_loss - current, (last_nz - nz).abs());
            if cfg.verbose {
                eprintln!("iter {iter}: loss={current:.6}, nonzero={nz:.1}, dloss={d_loss:.6}, dnz={d_nz:.2}");
            }
            stalls = if d_loss < 0.0 { stalls + 1 } else { 0 };
            if stalls > 10 {
                break;
            }
            if iter > 1000
                && d_loss > 0.0
                && d_loss / last_loss.abs() < cfg.ll_tolerance
                && d_nz / nz.max(1.0) < cfg.sparsity_tolerance
            {
                break;
            }
            last_loss = current;
            last_nz = nz;
        }
    }
    model
}

/// Gradient-descent deconvolution. Returns (exposures incl. intercept row, predicted counts),
/// both row-major: (cell types + 1) x samples and genes x samples.
pub fn deconvolve_gd(sigs: &Mat, bulk: &Mat, lengths: &[f32], weights: &[f32], cfg: &GdConfig) -> (Vec<f32>, Vec<f32>) {
    if let Some(s) = cfg.seed {
        CpuAd::seed(s);
    }
    let device = Default::default();
    let (n_genes, n_samples) = (sigs.rows, bulk.cols);
    let s: Tensor<CpuAd, 2> = tensor2(sigs, &device);
    let k: Tensor<CpuAd, 2> = tensor2(bulk, &device);
    let intercept_sig = Tensor::ones([n_genes, 1], &device).div_scalar(n_genes as f32);
    let k_log_k_minus_k = k.clone().mul(k.clone().add_scalar(1e-10).log()).sub(k.clone());
    let mut model = GdModel {
        log_exposures: GdModel::random_log_exposures([sigs.cols, n_samples], cfg, &device),
        intercept: Param::from_tensor(Tensor::zeros([1, n_samples], &device)),
        signatures_with_int: Tensor::cat(vec![s, intercept_sig], 1),
        conversion_factor: tensor1::<CpuAd>(lengths, &device)
            .reshape([n_genes, 1])
            .repeat(&[1, n_samples])
            .div_scalar(cfg.insert_size),
        observed_counts: k,
        gene_weights_expanded: tensor1::<CpuAd>(weights, &device).reshape([n_genes, 1]).repeat(&[1, n_samples]),
        k_log_k_minus_k,
    };
    fit_null(&mut model, cfg, &device);
    let model = fit_full(model, cfg, &device);
    let exposures = Tensor::cat(vec![model.log_exposures.val().exp(), model.intercept.val().exp()], 0);
    (to_vec(exposures.inner()), to_vec(model.forward().inner()))
}

pub struct EmConfig {
    pub max_iterations: usize,
    pub tolerance: f32,
    pub l1_lambda: f32,
    pub verbose: bool,
}

/// EM deconvolution with a uniform "noise" source (the intercept). E-step: split each
/// gene's counts between sources in proportion to S[g, j] * E[j, n]. M-step: a source's new
/// exposure is its weighted share of the counts; with L1, exposures are soft-thresholded.
/// As in R, gene lengths do not enter the EM model. Returns (exposures incl. intercept row,
/// predicted molecules), row-major.
pub fn deconvolve_em(sigs: &Mat, bulk: &Mat, weights: &[f32], cfg: &EmConfig) -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let (n_genes, n_types, n_samples) = (sigs.rows, sigs.cols, bulk.cols);
    let k: Tensor<Cpu, 2> = tensor2(bulk, &device);
    let s = Tensor::cat(
        vec![tensor2::<Cpu>(sigs, &device), Tensor::ones([n_genes, 1], &device).div_scalar(n_genes as f32)],
        1,
    );
    let w = tensor1::<Cpu>(weights, &device).reshape([n_genes, 1]);
    let total = k.clone().sum_dim(0);
    // Start: total counts spread evenly over cell types, plus 5 % noise.
    let mut exposures = Tensor::cat(
        vec![total.clone().repeat(&[n_types, 1]).div_scalar(n_types as f32), total.mul_scalar(0.05)],
        0,
    );
    let log_lik = |e: &Tensor<Cpu, 2>| -> f32 {
        let pred = s.clone().matmul(e.clone()).add_scalar(1e-10);
        let ll = k.clone().mul(pred.clone().log()).sub(pred).mul(w.clone().repeat(&[1, n_samples]));
        let mut ll = ll.sum().into_scalar().elem::<f32>();
        if cfg.l1_lambda != 0.0 {
            ll -= e.clone().slice([0..n_types, 0..n_samples]).abs().sum().into_scalar().elem::<f32>() * cfg.l1_lambda;
        }
        ll
    };
    let mut last = f32::NEG_INFINITY;
    for iter in 0..cfg.max_iterations {
        // E-step: responsibilities [genes, sources, samples].
        let expected = s.clone().matmul(exposures.clone()).add_scalar(1e-10);
        let resp = s.clone().unsqueeze_dim::<3>(2).mul(exposures.clone().unsqueeze_dim::<3>(0)).div(expected.unsqueeze_dim::<3>(1));
        // M-step: weighted counts assigned to each source.
        let weighted = resp.mul(w.clone().reshape([n_genes, 1, 1])).mul(k.clone().unsqueeze_dim::<3>(1));
        exposures = weighted.sum_dim(0).squeeze::<2>(0).clamp_min(1e-10);
        if cfg.l1_lambda != 0.0 {
            let cells = exposures.clone().slice([0..n_types, 0..n_samples]).sub_scalar(cfg.l1_lambda).clamp_min(0.0);
            let noise = exposures.clone().slice([n_types..n_types + 1, 0..n_samples]);
            exposures = Tensor::cat(vec![cells, noise], 0);
        }
        if iter % 10 == 0 {
            let ll = log_lik(&exposures);
            let change = ll - last;
            if cfg.verbose && iter % 50 == 0 {
                eprintln!("EM iter {iter}: objective={ll:.2}, change={change:.2}");
            }
            if iter > 10 && change.abs() < cfg.tolerance {
                break;
            }
            last = ll;
        }
    }
    let pred = s.matmul(exposures.clone());
    (to_vec(exposures), to_vec(pred))
}
