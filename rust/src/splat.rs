//! Simulate single cells from bulk RNA-seq profiles ("splatting").
//!
//! A line-for-line port of `splat.rs` in furlan-lab/viewmastR (MIT licence), without the
//! R glue. With the same seed, bandwidth and inputs it draws the same cells as R's
//! `splat_bulk_reference()`, because both use rand's `StdRng`.

use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Draw `n` library sizes: a random observed size plus Gaussian noise of sd `bandwidth`
/// (a kernel density draw), kept only when it falls strictly inside (min_size, max_size).
fn sample_sizes_with_kde(
    sizes: &[f64],
    bandwidth: f64,
    n: usize,
    min_size: f64,
    max_size: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let base = sizes[rng.random_range(0..sizes.len())];
        // Box-Muller normal deviate, as in the R package.
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let noisy = base + z * bandwidth;
        if noisy > min_size && noisy < max_size {
            out.push(noisy.round() as usize);
        }
    }
    out
}

/// Draw `target_size` molecules from the bulk "molecule pool" (one entry per count).
/// Sampling is with replacement when asked, or when the cell is larger than the pool.
fn simulate_cell(pool: &[usize], target_size: usize, replace: bool, rng: &mut StdRng) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    if replace || target_size > pool.len() {
        for _ in 0..target_size {
            *counts.entry(pool[rng.random_range(0..pool.len())]).or_insert(0) += 1;
        }
    } else {
        // Partial Fisher-Yates shuffle: sampling without replacement.
        let mut pool = pool.to_vec();
        for i in 0..target_size {
            let j = rng.random_range(i..pool.len());
            pool.swap(i, j);
            *counts.entry(pool[i]).or_insert(0) += 1;
        }
    }
    counts
}

/// Simulate `n_cells` cells from one bulk sample; returns COO triplets (gene, cell, count).
fn splat_sample(
    bulk: &[usize],
    sizes: &[f64],
    bandwidth: f64,
    n_cells: usize,
    replace: bool,
    seed: u64,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let min_size = sizes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_size = sizes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let targets = sample_sizes_with_kde(sizes, bandwidth, n_cells, min_size, max_size, &mut rng);
    let pool: Vec<usize> = bulk.iter().enumerate().flat_map(|(g, &c)| std::iter::repeat(g).take(c)).collect();
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for (cell, &target) in targets.iter().enumerate() {
        for (gene, count) in simulate_cell(&pool, target, replace, &mut rng) {
            rows.push(gene as i64);
            cols.push(cell as i64);
            vals.push(count as f64);
        }
    }
    (rows, cols, vals)
}

/// Splat every bulk sample (columns of `counts`, genes x samples) into `n_cells` cells each.
/// Sample `b` uses seed `seed + b`, so the result does not depend on the thread count.
/// Returns COO triplets over (genes, samples * n_cells); cells of sample `b` are columns
/// `b * n_cells .. (b + 1) * n_cells`.
pub fn splat(
    counts: &[Vec<usize>],
    sizes: &[f64],
    bandwidth: f64,
    n_cells: usize,
    replace: bool,
    seed: u64,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let parts: Vec<_> = counts
        .par_iter()
        .enumerate()
        .map(|(b, bulk)| splat_sample(bulk, sizes, bandwidth, n_cells, replace, seed.wrapping_add(b as u64)))
        .collect();
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for (b, (r, c, v)) in parts.into_iter().enumerate() {
        let offset = (b * n_cells) as i64;
        rows.extend(r);
        cols.extend(c.into_iter().map(|x| x + offset));
        vals.extend(v);
    }
    (rows, cols, vals)
}
