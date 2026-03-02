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
