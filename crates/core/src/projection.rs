use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// A point projected into 3D space for visualization.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Index back into the original EmbeddingSet.
    pub index: usize,
}

/// Which dimensionality-reduction algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionMethod {
    #[default]
    Pca,
    TSne,
}

/// Parameters for the t-SNE algorithm.
#[derive(Debug, Clone)]
pub struct TsneParams {
    /// Effective number of neighbours (5–50). Default: 30.
    pub perplexity: f32,
    /// Number of gradient-descent iterations. Default: 500.
    pub iterations: usize,
    /// Learning rate. Default: 200.
    pub learning_rate: f32,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            iterations: 500,
            learning_rate: 200.0,
        }
    }
}

/// Dimensionality reduction from high-dimensional embeddings to 3D.
pub struct Projector;

impl Projector {
    /// Project high-dimensional embeddings down to 3D using PCA.
    ///
    /// `vectors` is a slice of embedding vectors, all with the same dimensionality.
    /// Returns a Vec of 3D projected points.
    pub fn pca(vectors: &[&[f32]]) -> Vec<ProjectedPoint> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let n = vectors.len();
        let d = vectors[0].len();

        // Build the data matrix (n x d) — parallel row fill.
        let mut flat = vec![0.0f32; n * d];
        flat.par_chunks_mut(d)
            .zip(vectors.par_iter())
            .for_each(|(chunk, v)| chunk.copy_from_slice(v));
        let data = DMatrix::from_row_slice(n, d, &flat);

        // Center the data (parallel row-wise subtraction).
        let mean: DVector<f32> = data.row_mean().transpose();
        let mut centered_flat = flat;
        centered_flat
            .par_chunks_mut(d)
            .for_each(|row| {
                for j in 0..d {
                    row[j] -= mean[j];
                }
            });
        let centered = DMatrix::from_row_slice(n, d, &centered_flat);

        // Covariance matrix (d x d).  For efficiency with large d we use the
        // kernel trick: compute the n×n Gram matrix instead when n < d.
        let projected = if n < d {
            // Gram / kernel trick: (1/(n-1)) * centered * centeredᵀ gives n×n
            let gram = &centered * centered.transpose();
            let scale = if n > 1 { 1.0 / (n as f32 - 1.0) } else { 1.0 };
            let gram_scaled = gram * scale;

            // Eigen-decompose the small matrix.
            let eigen = gram_scaled.symmetric_eigen();
            // Take top-3 eigenvectors of the gram matrix, convert back to data space.
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                eigen.eigenvalues[b]
                    .partial_cmp(&eigen.eigenvalues[a])
                    .unwrap()
            });

            let k = 3.min(n);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let mut coords = [0.0f32; 3];
                for c in 0..k {
                    let idx = indices[c];
                    coords[c] = eigen.eigenvectors[(i, idx)];
                }
                result.push(ProjectedPoint {
                    x: coords[0],
                    y: coords[1],
                    z: coords[2],
                    index: i,
                });
            }
            result
        } else {
            // Standard: covariance = (1/(n-1)) * centeredᵀ * centered
            let cov = centered.transpose() * &centered;
            let scale = if n > 1 { 1.0 / (n as f32 - 1.0) } else { 1.0 };
            let cov_scaled = cov * scale;

            let eigen = cov_scaled.symmetric_eigen();
            let mut indices: Vec<usize> = (0..d).collect();
            indices.sort_by(|&a, &b| {
                eigen.eigenvalues[b]
                    .partial_cmp(&eigen.eigenvalues[a])
                    .unwrap()
            });

            let k = 3.min(d);
            // Project: result = centered * top_k_eigenvectors
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let row = centered.row(i);
                let mut coords = [0.0f32; 3];
                for c in 0..k {
                    let ev_col = eigen.eigenvectors.column(indices[c]);
                    coords[c] = row.dot(&ev_col.transpose());
                }
                result.push(ProjectedPoint {
                    x: coords[0],
                    y: coords[1],
                    z: coords[2],
                    index: i,
                });
            }
            result
        };

        projected
    }

    /// Project high-dimensional embeddings to 3D using exact t-SNE.
    ///
    /// Uses perplexity-based Gaussian affinities in the high-dimensional space
    /// and a Student t-distribution in the 3D output space.  Initialises from
    /// PCA for stability and runs gradient descent with early exaggeration and
    /// adaptive per-dimension learning rates (gains).
    pub fn tsne(vectors: &[&[f32]], params: &TsneParams) -> Vec<ProjectedPoint> {
        let n = vectors.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![ProjectedPoint { x: 0.0, y: 0.0, z: 0.0, index: 0 }];
        }

        // ── Step 1: Pairwise squared Euclidean distances (parallel) ──────────
        let mut dist2 = vec![0.0f32; n * n];
        // Each row i is independent — compute in parallel.
        dist2
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..n {
                    if j != i {
                        let sq: f32 = vectors[i]
                            .iter()
                            .zip(vectors[j].iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        row[j] = sq;
                    }
                }
            });

        // ── Step 2: High-dim conditional probabilities (parallel per row) ─────
        let perp = params.perplexity.min((n - 1) as f32 * 0.5).max(2.0);
        let target_entropy = perp.ln();

        let mut p_cond = vec![0.0f32; n * n];
        p_cond
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, p_row)| {
                let dist_row = &dist2[i * n..(i + 1) * n];
                let mut beta_lo = f32::NEG_INFINITY;
                let mut beta_hi = f32::INFINITY;
                let mut beta = 1.0_f32;
                let mut row = vec![0.0f32; n];

                for _ in 0..50 {
                    let mut z = 0.0_f32;
                    for j in 0..n {
                        if j != i {
                            row[j] = (-beta * dist_row[j]).exp();
                            z += row[j];
                        }
                    }
                    if z < f32::EPSILON { z = f32::EPSILON; }

                    let mut h = z.ln();
                    for j in 0..n {
                        if j != i {
                            h += beta * dist_row[j] * row[j] / z;
                        }
                    }
                    for j in 0..n { row[j] /= z; }

                    let diff = h - target_entropy;
                    if diff.abs() < 1e-5 { break; }
                    if diff > 0.0 {
                        beta_lo = beta;
                        beta = if beta_hi.is_infinite() { beta * 2.0 } else { (beta + beta_hi) / 2.0 };
                    } else {
                        beta_hi = beta;
                        beta = if beta_lo.is_infinite() { beta / 2.0 } else { (beta + beta_lo) / 2.0 };
                    }
                }
                p_row.copy_from_slice(&row);
            });

        // ── Step 3: Symmetrise P_ij = (P(j|i) + P(i|j)) / 2n ───────────────
        let mut p = vec![0.0f32; n * n];
        p.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..n {
                    let v = (p_cond[i * n + j] + p_cond[j * n + i]) / (2.0 * n as f32);
                    row[j] = v.max(1e-12);
                }
            });

        // ── Step 4: Initialise Y from PCA (scaled small for stability) ───────
        let pca_pts = Self::pca(vectors);
        let mut y: Vec<[f32; 3]> = pca_pts
            .iter()
            .map(|pt| [pt.x * 0.1, pt.y * 0.1, pt.z * 0.1])
            .collect();

        // ── Step 5: Gradient descent ─────────────────────────────────────────
        let mut gains = vec![[1.0_f32; 3]; n];
        let mut vel = vec![[0.0_f32; 3]; n];
        // Reusable buffer for Q numerators, reset each iteration.
        let mut q_num = vec![0.0_f32; n * n];

        for iter in 0..params.iterations {
            let exag: f32 = if iter < 250 { 12.0 } else { 1.0 };
            let momentum: f32 = if iter < 250 { 0.5 } else { 0.8 };

            // Low-dim affinities Q — each row computed independently (no cross-row reads).
            q_num
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in 0..n {
                        if j != i {
                            let dx = y[i][0] - y[j][0];
                            let dy = y[i][1] - y[j][1];
                            let dz = y[i][2] - y[j][2];
                            row[j] = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz);
                        } else {
                            row[j] = 0.0;
                        }
                    }
                });
            // sum of all non-diagonal entries = 2 * Σ_{i<j} q_{ij}, same as original.
            let sum_q = q_num.par_iter().sum::<f32>().max(f32::EPSILON);

            // Gradient — parallel over i, each writes only grad[i].
            let grad: Vec<[f32; 3]> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut g = [0.0_f32; 3];
                    let p_row = &p[i * n..(i + 1) * n];
                    let q_row = &q_num[i * n..(i + 1) * n];
                    for j in 0..n {
                        if j != i {
                            let q_ij = (q_row[j] / sum_q).max(1e-12);
                            let factor = 4.0 * (exag * p_row[j] - q_ij) * q_row[j];
                            g[0] += factor * (y[i][0] - y[j][0]);
                            g[1] += factor * (y[i][1] - y[j][1]);
                            g[2] += factor * (y[i][2] - y[j][2]);
                        }
                    }
                    g
                })
                .collect();

            // Adaptive gains + velocity (sequential, cheap).
            for i in 0..n {
                for k in 0..3 {
                    let same_sign = (grad[i][k] > 0.0) == (vel[i][k] > 0.0);
                    gains[i][k] = if same_sign { (gains[i][k] * 0.8).max(0.01) } else { gains[i][k] + 0.2 };
                    vel[i][k] = momentum * vel[i][k] - params.learning_rate * gains[i][k] * grad[i][k];
                    y[i][k] += vel[i][k];
                }
            }

            // Re-centre Y.
            for k in 0..3 {
                let mean = y.iter().map(|pt| pt[k]).sum::<f32>() / n as f32;
                for pt in y.iter_mut() { pt[k] -= mean; }
            }
        }

        y.into_iter()
            .enumerate()
            .map(|(i, pt)| ProjectedPoint { x: pt[0], y: pt[1], z: pt[2], index: i })
            .collect()
    }

    /// Assign each projected point to one of `k` clusters using k-means (Lloyd's
    /// algorithm).  Returns a `Vec<usize>` of cluster indices, one per point.
    /// Falls back gracefully when `n < k` by capping k at n.
    pub fn kmeans(points: &[ProjectedPoint], k: usize) -> Vec<usize> {
        let n = points.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);

        // Initialise centroids with k-means++ seeding for better convergence.
        let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
        // Pick the first centroid at index 0 (deterministic).
        centroids.push([points[0].x, points[0].y, points[0].z]);

        for _ in 1..k {
            // For each point compute its distance to the nearest existing centroid.
            let dists: Vec<f32> = points
                .iter()
                .map(|p| {
                    centroids
                        .iter()
                        .map(|c| {
                            let dx = p.x - c[0];
                            let dy = p.y - c[1];
                            let dz = p.z - c[2];
                            dx * dx + dy * dy + dz * dz
                        })
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Pick the point with the maximum distance (deterministic approximation).
            let next = dists
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            centroids.push([points[next].x, points[next].y, points[next].z]);
        }

        let mut assignments = vec![0usize; n];

        for _ in 0..100 {
            // Assignment step.
            let mut changed = false;
            for (i, p) in points.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| {
                        let dx = p.x - c[0];
                        let dy = p.y - c[1];
                        let dz = p.z - c[2];
                        (ci, dx * dx + dy * dy + dz * dz)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);
                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step.
            let mut sums = vec![[0.0f32; 3]; k];
            let mut counts = vec![0usize; k];
            for (i, p) in points.iter().enumerate() {
                let c = assignments[i];
                sums[c][0] += p.x;
                sums[c][1] += p.y;
                sums[c][2] += p.z;
                counts[c] += 1;
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    centroids[ci][0] = sums[ci][0] / counts[ci] as f32;
                    centroids[ci][1] = sums[ci][1] / counts[ci] as f32;
                    centroids[ci][2] = sums[ci][2] / counts[ci] as f32;
                }
            }
        }

        assignments
    }

    /// Normalize projected points to fit within a [-1, 1] cube.
    pub fn normalize(points: &mut [ProjectedPoint]) {
        if points.is_empty() {
            return;
        }

        let max_abs = points
            .iter()
            .flat_map(|p| [p.x.abs(), p.y.abs(), p.z.abs()])
            .fold(0.0f32, f32::max);

        if max_abs > 0.0 {
            for p in points.iter_mut() {
                p.x /= max_abs;
                p.y /= max_abs;
                p.z /= max_abs;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];
        let v4 = vec![1.0, 1.0, 0.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3, &v4];

        let points = Projector::pca(&vectors);
        assert_eq!(points.len(), 4);

        // Each point should have an index
        for (i, p) in points.iter().enumerate() {
            assert_eq!(p.index, i);
        }
    }
}
