use ndarray::{Array1, Array2, Axis};

use ndarray_linalg::cholesky::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RidgeError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    #[error("Zero sum of weights")]
    ZeroSumWeights,
    #[error("Model not fitted")]
    ModelNotFitted,
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
}

pub struct WeightedRidge {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub weights: Option<Array1<f64>>,
}

impl WeightedRidge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            weights: None,
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<(), RidgeError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if y.len() != n_samples {
            return Err(RidgeError::ShapeMismatch {
                expected: (n_samples, 1),
                got: (y.len(), 1),
            });
        }
        if sample_weight.len() != n_samples {
            return Err(RidgeError::ShapeMismatch {
                expected: (n_samples, 1),
                got: (sample_weight.len(), 1),
            });
        }

        // 1. Create the Weight Matrix W (as a view or transformation)
        // For efficiency, we perform element-wise multiplication instead of creating
        // a massive diagonal matrix. X_weighted = W * X
        let mut x_weighted = x.clone();
        for (mut row, &w) in x_weighted.axis_iter_mut(Axis(0)).zip(sample_weight.iter()) {
            row *= w;
        }

        // 2. Compute X^T * W * X
        let mut xt_w_x = x.t().dot(&x_weighted);

        // 3. Regularization: Add alpha directly to the diagonal to avoid creating an Identity matrix
        let start_idx = if self.fit_intercept { 1 } else { 0 };
        for i in start_idx..n_features {
            xt_w_x[[i, i]] += self.alpha;
        }

        // 4. Solve the system using Cholesky Decomposition
        let b = x_weighted.t().dot(y);

        // Factorize the Symmetric Positive-Definite matrix into L * L^T in-place.
        // UPLO::Lower tells LAPACK to only read the lower half of the matrix,
        // skipping redundant calculations for the upper half.
        let cholesky_factor = xt_w_x.factorizec_into(UPLO::Lower)?;

        // Solve for the weights using the factorized matrix
        self.weights = Some(cholesky_factor.solvec_into(b)?);

        Ok(())
    }

    pub fn score_r2_weighted(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let weights = self.weights.as_ref().ok_or(RidgeError::ModelNotFitted)?;
        let y_pred = x.dot(weights);
        let epsilon = 1e-38;

        // Weighted Average of (y_pred - y_true)^2
        let diff_sq = (&y_pred - y_true).mapv(|a| a.powi(2));
        let num = self
            .weighted_average(&diff_sq, sample_weight)
            .ok_or(RidgeError::ZeroSumWeights)?;

        // Weighted Average of (y_true)^2
        let true_sq = y_true.mapv(|a| a.powi(2));
        let den = self
            .weighted_average(&true_sq, sample_weight)
            .ok_or(RidgeError::ZeroSumWeights)?;

        Ok(1.0 - (num / (den + epsilon)))
    }

    /// Standard weighted R² (sklearn-style):
    /// R² = 1 − Σw(y−ŷ)² / Σw(y−ȳ_w)²
    pub fn score_r2_standard(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let weights = self.weights.as_ref().ok_or(RidgeError::ModelNotFitted)?;
        let y_pred = x.dot(weights);

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }

        // Weighted mean of y_true
        let y_mean = (y_true * sample_weight).sum() / sum_w;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..y_true.len() {
            let w = sample_weight[i];
            let diff = y_true[i] - y_pred[i];
            ss_res += w * diff * diff;
            let diff_mean = y_true[i] - y_mean;
            ss_tot += w * diff_mean * diff_mean;
        }

        if ss_tot < 1e-38 {
            return Ok(0.0);
        }
        Ok(1.0 - (ss_res / ss_tot))
    }

    pub fn predict(&self, x: &Array2<f64>) -> Option<Array1<f64>> {
        self.weights.as_ref().map(|w| x.dot(w))
    }

    fn weighted_average(&self, values: &Array1<f64>, weights: &Array1<f64>) -> Option<f64> {
        let sum_weights = weights.sum();
        if sum_weights == 0.0 {
            return None;
        }
        Some((values * weights).sum() / sum_weights)
    }
}

// Online Ridge with Forgetting Factor
pub struct AdaptiveRidge {
    pub beta: Array1<f64>,
    pub p_matrix: Array2<f64>,
    pub gamma: f64, // Forgetting factor (e.g., 0.99)
    pub weights: Option<Array1<f64>>,
}

impl AdaptiveRidge {
    /// Initialize with number of features, ridge penalty (alpha), and forgetting factor (gamma).
    pub fn new(n_features: usize, alpha: f64, gamma: f64) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");

        // Initial precision matrix: (1 / alpha) * I
        let mut p_matrix = Array2::<f64>::eye(n_features);
        p_matrix *= 1.0 / alpha;

        Self {
            beta: Array1::zeros(n_features),
            p_matrix,
            gamma,
            weights: None,
        }
    }

    /// Update the model with a new observation
    pub fn update(&mut self, x: &Array1<f64>, y: f64, weight: f64) {
        let n = self.beta.len();

        // 1. Calculate P * x
        let px = self.p_matrix.dot(x);

        // 2. Denominator: (gamma / weight) + x^T * P * x
        let x_px = x.dot(&px);
        let denominator = (self.gamma / weight) + x_px;

        // 3. Gain vector K
        let k = &px / denominator;

        // 4. Update weights: beta = beta + K * error
        let y_pred = x.dot(&self.beta);
        let error = y - y_pred;
        self.beta = &self.beta + &(&k * error);

        // 5. Compute the outer product: K * (P * x)^T
        // We reshape 1D arrays into 2D matrices (N x 1) and (1 x N) to compute
        // the outer product efficiently using Intel MKL's underlying dot implementation.
        let k_mat = k.into_shape_with_order((n, 1)).expect("Reshape failed");
        let px_mat = px.into_shape_with_order((1, n)).expect("Reshape failed");
        let outer_product = k_mat.dot(&px_mat);

        // 6. Update P matrix: P = (P - OuterProduct) / gamma
        self.p_matrix = (&self.p_matrix - &outer_product) / self.gamma;
    }

    /// Predict a single value
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.beta)
    }

    pub fn score_r2_weighted(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let y_pred = x.dot(self.weights.as_ref().ok_or(RidgeError::ModelNotFitted)?);
        let epsilon = 1e-38;

        // Weighted Average of (y_pred - y_true)^2
        let diff_sq = (&y_pred - y_true).mapv(|a| a.powi(2));
        let num = self
            .weighted_average(&diff_sq, sample_weight)
            .ok_or(RidgeError::ZeroSumWeights)?;

        // Weighted Average of (y_true)^2
        let true_sq = y_true.mapv(|a| a.powi(2));
        let den = self
            .weighted_average(&true_sq, sample_weight)
            .ok_or(RidgeError::ZeroSumWeights)?;

        Ok(1.0 - (num / (den + epsilon)))
    }

    /// Standard weighted R² (sklearn-style):
    /// R² = 1 − Σw(y−ŷ)² / Σw(y−ȳ_w)²
    pub fn score_r2_standard(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let y_pred = x.dot(self.weights.as_ref().ok_or(RidgeError::ModelNotFitted)?);

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }

        // Weighted mean of y_true
        let y_mean = (y_true * sample_weight).sum() / sum_w;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..y_true.len() {
            let w = sample_weight[i];
            let diff = y_true[i] - y_pred[i];
            ss_res += w * diff * diff;
            let diff_mean = y_true[i] - y_mean;
            ss_tot += w * diff_mean * diff_mean;
        }

        if ss_tot < 1e-38 {
            return Ok(0.0);
        }
        Ok(1.0 - (ss_res / ss_tot))
    }

    fn weighted_average(&self, values: &Array1<f64>, weights: &Array1<f64>) -> Option<f64> {
        let sum_weights = weights.sum();
        if sum_weights == 0.0 {
            return None;
        }
        Some((values * weights).sum() / sum_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_weighted_ridge_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![3.0, 7.0, 11.0]; // y = 1*x1 + 1*x2 roughly
        let weights = array![1.0, 1.0, 1.0];

        let mut ridge = WeightedRidge::new(0.1, false);
        ridge.fit(&x, &y, &weights).unwrap();

        let pred = ridge.predict(&x).unwrap();
        // Check if predictions are close to actual y
        for (p, a) in pred.iter().zip(y.iter()) {
            assert!((p - a).abs() < 0.5);
        }
    }

    #[test]
    fn test_adaptive_ridge_basic() {
        let mut adaptive = AdaptiveRidge::new(2, 1.0, 0.99);
        let x = array![1.0, 2.0];
        let y = 5.0; // 1*1 + 2*2 = 5
        let weight = 1.0;

        adaptive.update(&x, y, weight);
        let pred = adaptive.predict(&x);
        assert!((pred - y).abs() < 5.0); // Initial prediction might be off, just check it runs
    }

    #[test]
    fn test_error_handling() {
        let mut ridge = WeightedRidge::new(1.0, false);
        let x = array![[1.0]];
        let y = array![1.0, 2.0]; // Mismatch
        let weights = array![1.0];

        let result = ridge.fit(&x, &y, &weights);
        assert!(matches!(result, Err(RidgeError::ShapeMismatch { .. })));
    }
}
