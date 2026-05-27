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
    #[error("Negative sample weight")]
    NegativeWeight,
    #[error("Model not fitted")]
    ModelNotFitted,
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
}

pub struct WeightedRidge {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub coefficients: Option<Array1<f64>>,
    pub intercept: f64,
}

impl WeightedRidge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            coefficients: None,
            intercept: 0.0,
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
        if sample_weight.iter().any(|&w| w < 0.0) {
            return Err(RidgeError::NegativeWeight);
        }
        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }

        // Center data if fitting intercept (sklearn-style):
        // Solve on centered data, then recover intercept = ȳ - x̄ᵀβ
        let (x_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let mut x_mean = Array1::<f64>::zeros(n_features);
            for (row, &w) in x.axis_iter(Axis(0)).zip(sample_weight.iter()) {
                x_mean.scaled_add(w, &row);
            }
            x_mean /= sum_w;

            let y_mean = (y * sample_weight).sum() / sum_w;

            let mut x_centered = x.clone();
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &x_mean;
            }
            let y_centered = y - y_mean;

            (x_centered, y_centered, Some(x_mean), y_mean)
        } else {
            (x.clone(), y.clone(), None, 0.0)
        };

        // Apply sample weights: X_w = diag(w) · X
        let mut x_weighted = x_work.clone();
        for (mut row, &w) in x_weighted.axis_iter_mut(Axis(0)).zip(sample_weight.iter()) {
            row *= w;
        }

        // X^T W X
        let mut xt_w_x = x_work.t().dot(&x_weighted);

        // Ridge penalty on all feature dimensions (intercept handled by centering)
        for i in 0..n_features {
            xt_w_x[[i, i]] += self.alpha;
        }

        // X^T W y
        let b = x_weighted.t().dot(&y_work);

        let cholesky_factor = xt_w_x.factorizec_into(UPLO::Lower)?;
        let coeffs = cholesky_factor.solvec_into(b)?;

        self.intercept = if self.fit_intercept {
            y_mean - x_mean.unwrap().dot(&coeffs)
        } else {
            0.0
        };

        self.coefficients = Some(coeffs);
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Option<Array1<f64>> {
        self.coefficients
            .as_ref()
            .map(|w| x.dot(w) + self.intercept)
    }

    /// Weighted R² using vectorized ops.
    /// R² = 1 − Σw(y−ŷ)² / Σw(y−ȳ_w)²
    pub fn score_r2_weighted(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let coeffs = self.coefficients.as_ref().ok_or(RidgeError::ModelNotFitted)?;
        let y_pred = x.dot(coeffs) + self.intercept;

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }
        let y_mean = (y_true * sample_weight).sum() / sum_w;

        let diff_sq = (&y_pred - y_true).mapv(|a| a.powi(2));
        let num = (&diff_sq * sample_weight).sum();

        let var_sq = y_true.mapv(|a| (a - y_mean).powi(2));
        let den = (&var_sq * sample_weight).sum();

        if den < 1e-38 {
            return Ok(0.0);
        }
        Ok(1.0 - (num / den))
    }

    /// Standard weighted R² (sklearn-style):
    /// R² = 1 − Σw(y−ŷ)² / Σw(y−ȳ_w)²
    pub fn score_r2_standard(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let coeffs = self.coefficients.as_ref().ok_or(RidgeError::ModelNotFitted)?;
        let y_pred = x.dot(coeffs) + self.intercept;

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }

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
}

// Online Ridge with Forgetting Factor (Recursive Least Squares)
pub struct AdaptiveRidge {
    pub beta: Array1<f64>,
    pub p_matrix: Array2<f64>,
    pub gamma: f64,
}

impl AdaptiveRidge {
    pub fn new(n_features: usize, alpha: f64, gamma: f64) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");

        let mut p_matrix = Array2::<f64>::eye(n_features);
        p_matrix *= 1.0 / alpha;

        Self {
            beta: Array1::zeros(n_features),
            p_matrix,
            gamma,
        }
    }

    pub fn update(&mut self, x: &Array1<f64>, y: f64, weight: f64) {
        assert!(weight > 0.0, "Sample weight must be positive");

        let n = self.beta.len();

        // K = P·x / (γ/w + xᵀ·P·x)
        let px = self.p_matrix.dot(x);
        let x_px = x.dot(&px);
        let denominator = (self.gamma / weight) + x_px;
        let k = &px / denominator;

        // β = β + K·(y − xᵀβ)
        let y_pred = x.dot(&self.beta);
        let error = y - y_pred;
        self.beta = &self.beta + &(&k * error);

        // P = (P − K·(Px)ᵀ) / γ
        let k_mat = k.into_shape_with_order((n, 1)).expect("Reshape failed");
        let px_mat = px.into_shape_with_order((1, n)).expect("Reshape failed");
        let outer_product = k_mat.dot(&px_mat);
        self.p_matrix = (&self.p_matrix - &outer_product) / self.gamma;
    }

    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.beta)
    }

    pub fn score_r2_weighted(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let y_pred = x.dot(&self.beta);

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }
        let y_mean = (y_true * sample_weight).sum() / sum_w;

        let diff_sq = (&y_pred - y_true).mapv(|a| a.powi(2));
        let num = (&diff_sq * sample_weight).sum();

        let var_sq = y_true.mapv(|a| (a - y_mean).powi(2));
        let den = (&var_sq * sample_weight).sum();

        if den < 1e-38 {
            return Ok(0.0);
        }
        Ok(1.0 - (num / den))
    }

    /// Standard weighted R² (sklearn-style):
    /// R² = 1 − Σw(y−ŷ)² / Σw(y−ȳ_w)²
    pub fn score_r2_standard(
        &self,
        x: &Array2<f64>,
        y_true: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<f64, RidgeError> {
        let y_pred = x.dot(&self.beta);

        let sum_w = sample_weight.sum();
        if sum_w == 0.0 {
            return Err(RidgeError::ZeroSumWeights);
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── Bug fix #1: AdaptiveRidge scoring used vestigial `weights` (always None)
    //    instead of `beta`. Both score methods would always return ModelNotFitted.
    #[test]
    fn test_adaptive_ridge_scoring_uses_beta() {
        let mut adaptive = AdaptiveRidge::new(2, 0.1, 0.99);

        // Train on y = x1 + 2·x2
        for &(ref x, y) in &[
            (array![1.0, 1.0], 3.0),
            (array![2.0, 1.0], 4.0),
            (array![1.0, 2.0], 5.0),
            (array![3.0, 2.0], 7.0),
            (array![2.0, 3.0], 8.0),
        ] {
            adaptive.update(x, y, 1.0);
        }

        let x_test = array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0], [2.0, 3.0]];
        let y_test = array![3.0, 4.0, 5.0, 7.0, 8.0];
        let w_test = array![1.0, 1.0, 1.0, 1.0, 1.0];

        // Before fix: Err(ModelNotFitted). After fix: Ok with reasonable R².
        let r2 = adaptive
            .score_r2_weighted(&x_test, &y_test, &w_test)
            .expect("score_r2_weighted must not return ModelNotFitted after training");
        assert!(r2 > 0.5, "R² should be reasonable after training, got {r2}");

        let r2_std = adaptive
            .score_r2_standard(&x_test, &y_test, &w_test)
            .expect("score_r2_standard must not return ModelNotFitted after training");
        assert!(r2_std > 0.5, "R² standard should be reasonable, got {r2_std}");
    }

    // ── Bug fix #2: R² denominator used Σw·y² instead of Σw·(y−ȳ)².
    //    For data with a large offset (ȳ >> 0), the old code severely underestimates R².
    #[test]
    fn test_r2_correct_denominator() {
        // y ≈ 100 + 2·x  →  ȳ ≈ 106, var(y) ≈ 8
        // Old buggy denominator ≈ Σw·y² ≈ 11240  →  R² ≈ 0.001
        // Correct denominator   ≈ Σw·(y−ȳ)² ≈ 40 →  R² ≈ 0.99
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![102.0, 104.0, 106.0, 108.0, 110.0];
        let weights = array![1.0, 1.0, 1.0, 1.0, 1.0];

        let mut ridge = WeightedRidge::new(0.001, true);
        ridge.fit(&x, &y, &weights).unwrap();

        let r2_w = ridge.score_r2_weighted(&x, &y, &weights).unwrap();
        let r2_s = ridge.score_r2_standard(&x, &y, &weights).unwrap();

        assert!(
            (r2_w - r2_s).abs() < 1e-10,
            "Both R² methods must agree: weighted={r2_w}, standard={r2_s}",
        );
        // With the old buggy denominator (Σw·y²), R² ≈ 0.001 here because
        // the denominator is dominated by the ~106² offset.
        // With the correct denominator (Σw·(y−ȳ)²), R² ≈ 1.0.
        assert!(
            r2_w > 0.9,
            "R² should be near 1.0 for a near-perfect linear fit, got {r2_w}",
        );
    }

    // ── Bug fix #3: fit_intercept=true did nothing useful — it just skipped
    //    regularization on the first *feature* column instead of fitting an intercept.
    #[test]
    fn test_fit_intercept() {
        // y = 10 + 2·x  (intercept = 10, slope = 2)
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![12.0, 14.0, 16.0, 18.0, 20.0];
        let weights = array![1.0, 1.0, 1.0, 1.0, 1.0];

        let mut ridge = WeightedRidge::new(0.001, true);
        ridge.fit(&x, &y, &weights).unwrap();

        assert!(
            (ridge.intercept - 10.0).abs() < 0.5,
            "Intercept should be ~10, got {}",
            ridge.intercept,
        );
        let coeffs = ridge.coefficients.as_ref().unwrap();
        assert!(
            (coeffs[0] - 2.0).abs() < 0.5,
            "Slope should be ~2, got {}",
            coeffs[0],
        );

        // In-sample predictions
        let pred = ridge.predict(&x).unwrap();
        for (p, a) in pred.iter().zip(y.iter()) {
            assert!((p - a).abs() < 0.5, "Prediction {p} too far from actual {a}");
        }

        // Out-of-sample: x=10 → y ≈ 30
        let pred_new = ridge.predict(&array![[10.0]]).unwrap();
        assert!(
            (pred_new[0] - 30.0).abs() < 1.0,
            "Prediction for x=10 should be ~30, got {}",
            pred_new[0],
        );
    }

    // ── Bug fix #4: No validation on sample weights.
    #[test]
    fn test_negative_weight_rejected() {
        let mut ridge = WeightedRidge::new(1.0, false);
        let result = ridge.fit(&array![[1.0]], &array![1.0], &array![-1.0]);
        assert!(
            matches!(result, Err(RidgeError::NegativeWeight)),
            "Negative weights must be rejected",
        );
    }

    #[test]
    fn test_zero_weight_sum_rejected() {
        let mut ridge = WeightedRidge::new(1.0, false);
        let result = ridge.fit(&array![[1.0], [2.0]], &array![1.0, 2.0], &array![0.0, 0.0]);
        assert!(
            matches!(result, Err(RidgeError::ZeroSumWeights)),
            "All-zero weights must be rejected",
        );
    }

    #[test]
    fn test_adaptive_positive_weight_required() {
        let mut adaptive = AdaptiveRidge::new(2, 1.0, 0.99);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            adaptive.update(&array![1.0, 2.0], 5.0, -1.0);
        }));
        assert!(result.is_err(), "Negative weight must panic");
    }

    // ── Existing / regression tests ────────────────────────────────────────────

    #[test]
    fn test_weighted_ridge_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![3.0, 7.0, 11.0];
        let weights = array![1.0, 1.0, 1.0];

        let mut ridge = WeightedRidge::new(0.1, false);
        ridge.fit(&x, &y, &weights).unwrap();

        let pred = ridge.predict(&x).unwrap();
        for (p, a) in pred.iter().zip(y.iter()) {
            assert!((p - a).abs() < 0.5);
        }
    }

    #[test]
    fn test_adaptive_ridge_basic() {
        let mut adaptive = AdaptiveRidge::new(2, 1.0, 0.99);
        adaptive.update(&array![1.0, 2.0], 5.0, 1.0);
        let pred = adaptive.predict(&array![1.0, 2.0]);
        assert!((pred - 5.0).abs() < 5.0);
    }

    #[test]
    fn test_error_handling() {
        let mut ridge = WeightedRidge::new(1.0, false);
        let result = ridge.fit(&array![[1.0]], &array![1.0, 2.0], &array![1.0]);
        assert!(matches!(result, Err(RidgeError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_r2_methods_agree() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![5.0, 11.0, 17.0, 23.0];
        let weights = array![1.0, 2.0, 1.0, 3.0];

        let mut ridge = WeightedRidge::new(0.01, false);
        ridge.fit(&x, &y, &weights).unwrap();

        let r2_w = ridge.score_r2_weighted(&x, &y, &weights).unwrap();
        let r2_s = ridge.score_r2_standard(&x, &y, &weights).unwrap();
        assert!(
            (r2_w - r2_s).abs() < 1e-10,
            "Both R² methods must agree: {r2_w} vs {r2_s}",
        );
    }

    #[test]
    fn test_sample_weights_matter() {
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![10.0, 10.0, 100.0];

        let mut ridge_eq = WeightedRidge::new(0.01, false);
        ridge_eq.fit(&x, &y, &array![1.0, 1.0, 1.0]).unwrap();

        let mut ridge_heavy = WeightedRidge::new(0.01, false);
        ridge_heavy
            .fit(&x, &y, &array![1.0, 1.0, 100.0])
            .unwrap();

        let pred_eq = ridge_eq.predict(&array![[1.0]]).unwrap()[0];
        let pred_heavy = ridge_heavy.predict(&array![[1.0]]).unwrap()[0];

        assert!(
            pred_heavy > pred_eq,
            "Heavy-weighted prediction {pred_heavy} should exceed equal-weighted {pred_eq}",
        );
    }

    #[test]
    fn test_predict_unfitted_returns_none() {
        let ridge = WeightedRidge::new(1.0, false);
        assert!(ridge.predict(&array![[1.0]]).is_none());
    }

    #[test]
    fn test_fit_intercept_with_weighted_samples() {
        // y = 5 + 3·x, but weight the last sample heavily
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![8.0, 11.0, 14.0, 17.0, 20.0];
        let weights = array![1.0, 1.0, 1.0, 1.0, 10.0];

        let mut ridge = WeightedRidge::new(0.001, true);
        ridge.fit(&x, &y, &weights).unwrap();

        // Should still recover intercept ≈ 5, slope ≈ 3
        assert!(
            (ridge.intercept - 5.0).abs() < 1.0,
            "Intercept should be ~5, got {}",
            ridge.intercept,
        );
        let coeffs = ridge.coefficients.as_ref().unwrap();
        assert!(
            (coeffs[0] - 3.0).abs() < 1.0,
            "Slope should be ~3, got {}",
            coeffs[0],
        );
    }
}
