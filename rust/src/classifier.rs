//! Cell-type classifiers trained with burn on the CPU (ndarray) backend.
//!
//! The models, optimiser settings, batch size and data loader are those of the R
//! package's `scrna_mlr.rs`, `scrna_ann.rs` and `scrna_ann2l.rs`
//! (furlan-lab/viewmastR, MIT licence). Field names are kept (`linear1`,
//! `linear2`, `linear3`, `n_classes`) so a saved `model.mpk` loads in R's
//! `viewmastR_infer()` and vice versa.

use std::time::Instant;

use burn::{
    backend::{Autodiff, NdArray},
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    module::{AutodiffModule, Module},
    nn::{loss::CrossEntropyLoss, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{activation::relu, ElementConversion, Int, Tensor},
};

pub type Cpu = NdArray<f32>;
type CpuAd = Autodiff<Cpu>;

/// One cell: its normalised feature vector and its class index.
#[derive(Clone, Debug)]
pub struct Item {
    pub x: Vec<f32>,
    pub y: i32,
}

#[derive(Clone, Debug)]
pub struct CellBatch<B: Backend> {
    x: Tensor<B, 2>,
    y: Tensor<B, 1, Int>,
}

#[derive(Clone, Default)]
struct CellBatcher;

impl<B: Backend> Batcher<B, Item, CellBatch<B>> for CellBatcher {
    fn batch(&self, items: Vec<Item>, device: &B::Device) -> CellBatch<B> {
        let (n, f) = (items.len(), items[0].x.len());
        let flat: Vec<f32> = items.iter().flat_map(|i| i.x.iter().copied()).collect();
        let y: Vec<i32> = items.iter().map(|i| i.y).collect();
        CellBatch {
            x: Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([n, f]),
            y: Tensor::<B, 1, Int>::from_ints(y.as_slice(), device),
        }
    }
}

/// Anything that maps a batch of cells to class logits.
pub trait Classifier<B: Backend> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
    fn layers(&self) -> Vec<&Linear<B>>;
}

/// Multinomial logistic regression: one linear layer ("mlr").
#[derive(Module, Debug)]
pub struct Mlr<B: Backend> {
    linear1: Linear<B>,
    n_classes: usize,
}

/// Perceptron with one hidden ReLU layer ("nn", one hidden size).
#[derive(Module, Debug)]
pub struct Ann<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    n_classes: usize,
}

/// Perceptron with two hidden ReLU layers ("nn", two hidden sizes).
#[derive(Module, Debug)]
pub struct Ann2<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    n_classes: usize,
}

impl<B: Backend> Classifier<B> for Mlr<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear1.forward(x)
    }
    fn layers(&self) -> Vec<&Linear<B>> {
        vec![&self.linear1]
    }
}

impl<B: Backend> Classifier<B> for Ann<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear2.forward(relu(self.linear1.forward(x)))
    }
    fn layers(&self) -> Vec<&Linear<B>> {
        vec![&self.linear1, &self.linear2]
    }
}

impl<B: Backend> Classifier<B> for Ann2<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.linear2.forward(relu(self.linear1.forward(x))));
        self.linear3.forward(h)
    }
    fn layers(&self) -> Vec<&Linear<B>> {
        vec![&self.linear1, &self.linear2, &self.linear3]
    }
}

/// Per-epoch training curves, as returned by R viewmastR in `history`.
#[derive(Default)]
pub struct History {
    pub train_acc: Vec<f64>,
    pub train_loss: Vec<f64>,
    pub test_acc: Vec<f64>,
    pub test_loss: Vec<f64>,
}

/// A trained model: one (weight [in, out], bias [out]) pair per layer.
pub struct Fit {
    pub weights: Vec<(Vec<f32>, [usize; 2], Vec<f32>)>,
    pub history: History,
    pub seconds: f64,
}

pub struct TrainConfig {
    pub hidden: Vec<usize>,
    pub n_classes: usize,
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: Option<u64>,
    pub model_path: Option<String>,
    pub verbose: bool,
}

/// Train the model selected by `cfg.hidden` (empty = mlr, one size = ann, two = ann2).
pub fn train(train: Vec<Item>, test: Vec<Item>, cfg: &TrainConfig) -> Fit {
    if let Some(s) = cfg.seed {
        CpuAd::seed(s);
    }
    let device = Default::default();
    let n_in = train[0].x.len();
    let lin = |i, o| LinearConfig::new(i, o).init::<CpuAd>(&device);
    let k = cfg.n_classes;
    // R uses Adam eps = 1e-4 without weight decay for mlr, and burn's default Adam for the perceptrons.
    match cfg.hidden.as_slice() {
        [] => {
            let opt = AdamConfig::new().with_weight_decay(None).with_epsilon(1e-4);
            fit(Mlr { linear1: lin(n_in, k), n_classes: k }, opt, train, test, cfg)
        }
        [h] => fit(
            Ann { linear1: lin(n_in, *h), linear2: lin(*h, k), n_classes: k },
            AdamConfig::new(), train, test, cfg,
        ),
        [h1, h2] => fit(
            Ann2 { linear1: lin(n_in, *h1), linear2: lin(*h1, *h2), linear3: lin(*h2, k), n_classes: k },
            AdamConfig::new(), train, test, cfg,
        ),
        _ => panic!("hidden_layers must have 0, 1 or 2 entries"),
    }
}

/// The R viewmastR training loop (`run_custom` in `scrna_mlr.rs`): Adam on cross-entropy,
/// `epochs` passes, one validation pass per epoch. The train set is never reshuffled; with
/// `num_workers > 1` burn splits it into that many contiguous shards whose batches arrive
/// in thread-completion order, which is why two R runs differ slightly.
fn fit<M>(mut model: M, opt: AdamConfig, train: Vec<Item>, test: Vec<Item>, cfg: &TrainConfig) -> Fit
where
    M: AutodiffModule<CpuAd> + Classifier<CpuAd>,
    M::InnerModule: Classifier<Cpu>,
{
    let device = Default::default();
    let mut optim = opt.init::<CpuAd, M>();
    let mut train_loader = DataLoaderBuilder::new(CellBatcher).batch_size(cfg.batch_size);
    let mut test_loader = DataLoaderBuilder::new(CellBatcher).batch_size(cfg.batch_size);
    if cfg.num_workers > 1 {
        train_loader = train_loader.num_workers(cfg.num_workers);
        test_loader = test_loader.num_workers(cfg.num_workers);
    }
    let train_loader = train_loader.build(InMemDataset::new(train));
    let test_loader = test_loader.build(InMemDataset::new(test));

    let mut history = History::default();
    let start = Instant::now();
    for epoch in 1..=cfg.epochs {
        let (mut correct, mut seen, mut losses) = (0usize, 0usize, Vec::new());
        for batch in train_loader.iter() {
            let logits = model.forward(batch.x);
            let loss = CrossEntropyLoss::new(None, &device).forward(logits.clone(), batch.y.clone());
            correct += n_correct(logits, batch.y.clone());
            seen += batch.y.dims()[0];
            losses.push(loss.clone().into_scalar().elem::<f64>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
        history.train_acc.push(correct as f64 / seen as f64);
        history.train_loss.push(mean(&losses));

        let valid = model.valid();
        let (mut correct, mut seen, mut losses) = (0usize, 0usize, Vec::new());
        for batch in test_loader.iter() {
            let logits = valid.forward(batch.x);
            let loss = CrossEntropyLoss::new(None, &device).forward(logits.clone(), batch.y.clone());
            correct += n_correct(logits, batch.y.clone());
            seen += batch.y.dims()[0];
            losses.push(loss.into_scalar().elem::<f64>());
        }
        history.test_acc.push(correct as f64 / seen as f64);
        history.test_loss.push(mean(&losses));
        if cfg.verbose {
            eprintln!(
                "[epoch {epoch} | train accuracy {:.3}% | validation accuracy {:.3}%]",
                100.0 * history.train_acc[epoch - 1],
                100.0 * history.test_acc[epoch - 1]
            );
        }
    }
    let seconds = start.elapsed().as_secs_f64();

    let model = model.valid();
    let weights = model
        .layers()
        .iter()
        .map(|l| {
            let w = l.weight.val();
            let dims = w.dims();
            let b = l.bias.as_ref().expect("linear layers carry a bias").val();
            (w.into_data().to_vec::<f32>().unwrap(), dims, b.into_data().to_vec::<f32>().unwrap())
        })
        .collect();
    if let Some(path) = &cfg.model_path {
        model
            .save_file(path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
            .expect("could not save model.mpk");
    }
    Fit { weights, history, seconds }
}

fn n_correct<B: Backend>(logits: Tensor<B, 2>, y: Tensor<B, 1, Int>) -> usize {
    let hit = logits.argmax(1).squeeze::<1>(1).equal(y).int().sum();
    hit.into_scalar().elem::<i64>() as usize
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

