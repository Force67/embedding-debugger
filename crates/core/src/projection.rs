use nalgebra::{DMatrix, DVector};

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

        // Build the data matrix (n x d).
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let data = DMatrix::from_row_slice(n, d, &flat);

        // Center the data.
        let mean: DVector<f32> = data.row_mean().transpose();
        let mut centered = data.clone();
        for mut row in centered.row_iter_mut() {
            for j in 0..d {
                row[j] -= mean[j];
            }
        }

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

        // ── Step 1: Pairwise squared Euclidean distances ─────────────────────
        let mut dist2 = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sq: f32 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                dist2[i * n + j] = sq;
                dist2[j * n + i] = sq;
            }
        }

        // ── Step 2: High-dim conditional probabilities (perplexity search) ───
        // Clamp perplexity to a reasonable range for the dataset size.
        let perp = params.perplexity.min((n - 1) as f32 * 0.5).max(2.0);
        let target_entropy = perp.ln();

        let mut p_cond = vec![0.0f32; n * n]; // row-normalised P(j|i)
        for i in 0..n {
            let mut beta_lo = f32::NEG_INFINITY;
            let mut beta_hi = f32::INFINITY;
            let mut beta = 1.0_f32;
            let mut p_row = vec![0.0f32; n];

            for _ in 0..50 {
                let mut z = 0.0_f32;
                for j in 0..n {
                    if j != i {
                        p_row[j] = (-beta * dist2[i * n + j]).exp();
                        z += p_row[j];
                    }
                }
                if z < f32::EPSILON {
                    z = f32::EPSILON;
                }

                // Shannon entropy H = log Z + beta * E_p[d²]
                let mut h = z.ln();
                for j in 0..n {
                    if j != i {
                        h += beta * dist2[i * n + j] * p_row[j] / z;
                    }
                }

                // Normalise before checking convergence.
                for j in 0..n {
                    p_row[j] /= z;
                }

                let diff = h - target_entropy;
                if diff.abs() < 1e-5 {
                    break;
                }
                if diff > 0.0 {
                    // Entropy too high → sharpen Gaussian (increase beta).
                    beta_lo = beta;
                    beta = if beta_hi.is_infinite() { beta * 2.0 } else { (beta + beta_hi) / 2.0 };
                } else {
                    // Entropy too low → widen Gaussian (decrease beta).
                    beta_hi = beta;
                    beta = if beta_lo.is_infinite() { beta / 2.0 } else { (beta + beta_lo) / 2.0 };
                }
            }

            for j in 0..n {
                p_cond[i * n + j] = p_row[j];
            }
        }

        // ── Step 3: Symmetrise P_ij = (P(j|i) + P(i|j)) / 2n ───────────────
        let mut p = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = (p_cond[i * n + j] + p_cond[j * n + i]) / (2.0 * n as f32);
                p[i * n + j] = v.max(1e-12);
            }
        }

        // ── Step 4: Initialise Y from PCA (scaled small for stability) ───────
        let pca_pts = Self::pca(vectors);
        let mut y: Vec<[f32; 3]> = pca_pts
            .iter()
            .map(|pt| [pt.x * 0.1, pt.y * 0.1, pt.z * 0.1])
            .collect();

        // ── Step 5: Gradient descent ─────────────────────────────────────────
        let mut gains = vec![[1.0_f32; 3]; n];
        let mut vel = vec![[0.0_f32; 3]; n];

        for iter in 0..params.iterations {
            // Early exaggeration for the first 250 iterations.
            let exag: f32 = if iter < 250 { 12.0 } else { 1.0 };
            let momentum: f32 = if iter < 250 { 0.5 } else { 0.8 };

            // Low-dim affinities Q (Student t, unnormalised numerator).
            let mut q_num = vec![0.0_f32; n * n];
            let mut sum_q = 0.0_f32;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = y[i][0] - y[j][0];
                    let dy = y[i][1] - y[j][1];
                    let dz = y[i][2] - y[j][2];
                    let v = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz);
                    q_num[i * n + j] = v;
                    q_num[j * n + i] = v;
                    sum_q += 2.0 * v;
                }
            }
            if sum_q < f32::EPSILON {
                sum_q = f32::EPSILON;
            }

            // Gradient: dKL/dy_i = 4 Σ_j (p_ij - q_ij) * q_num_ij * (y_i - y_j)
            let mut grad = vec![[0.0_f32; 3]; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let q_ij = (q_num[i * n + j] / sum_q).max(1e-12);
                        let factor =
                            4.0 * (exag * p[i * n + j] - q_ij) * q_num[i * n + j];
                        for k in 0..3 {
                            grad[i][k] += factor * (y[i][k] - y[j][k]);
                        }
                    }
                }
            }

            // Adaptive gains + velocity update.
            for i in 0..n {
                for k in 0..3 {
                    let same_sign = (grad[i][k] > 0.0) == (vel[i][k] > 0.0);
                    gains[i][k] = if same_sign {
                        (gains[i][k] * 0.8).max(0.01)
                    } else {
                        gains[i][k] + 0.2
                    };
                    vel[i][k] = momentum * vel[i][k]
                        - params.learning_rate * gains[i][k] * grad[i][k];
                    y[i][k] += vel[i][k];
                }
            }

            // Re-centre Y.
            for k in 0..3 {
                let mean = y.iter().map(|pt| pt[k]).sum::<f32>() / n as f32;
                for pt in y.iter_mut() {
                    pt[k] -= mean;
                }
            }
        }

        y.into_iter()
            .enumerate()
            .map(|(i, pt)| ProjectedPoint { x: pt[0], y: pt[1], z: pt[2], index: i })
            .collect()
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
