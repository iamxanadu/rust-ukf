//! A crate providing an implementation of the unscented Kalman filter algorithm.
//!
//! This implementation follows the steps presented in ['this guide by mathworks']. It is able to support multiple different kinds of measurement updates through separation of the update and innovation steps of the filter.
//!
//! Examples for the usage of this crate can be found in the examples folder.
//!
//! TODO
//! - Change the filter methods to not take control. This can be handled in the provided closures instead. Is a more general interface for systems that do not specifically break out control.
//! - Support no_std
//! - Add some more examples: one with only one measurement update and one that does not require ode_solvers
//!
//! ['this guide by mathworks']: https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html

#![feature(generic_const_exprs)]

use nalgebra::{SMatrix, SVector};

/// Type for state vectors of dimension S
pub type State<const S: usize> = SVector<f64, S>;
/// Type for covariance matricies of dimension SxS
pub type Covariance<const S: usize> = SMatrix<f64, S, S>;
/// Type for control vectors of dimension C
pub type Control<const C: usize> = SVector<f64, C>;
/// Type for measurement outputs of dimension Y
pub type Output<const Y: usize> = SVector<f64, Y>;
/// Type for cross covariance matrices of dimension SxY
pub type CovarianceSY<const S: usize, const Y: usize> = SMatrix<f64, S, Y>;

/// Unscented Kalman filter structure. Contains the state of the filter and its covariance as well as a set of weights calculated for the algorithm based on the initialization parameters.
pub struct UnscentedKalmanFilter<const S: usize>
where
    [(); 2 * S + 1]:,
{
    pub state: State<S>,
    pub covariance: Covariance<S>,
    wm: [f64; 2 * S + 1],
    wc: [f64; 2 * S + 1],
    c: f64,
}

impl<const S: usize> UnscentedKalmanFilter<S>
where
    [(); 2 * S + 1]:,
{
    /// Constructor for the filter. Takes three filter parameters and the initial filter state and its covariance.
    ///
    /// The parameters are alpha, beta, and kappa. Alpha is usually a small positive constant which tunes the spread of the sigma points that the filter calculates. Beta helps incorporate knowledge about the underlying distribution of the filter. In most cases, this will be assumed to be Gaussian, and in this case beta=2.0 is optimal. Kappa is another parameter that tunes the sigma points and can usually be set to zero. Please see ['the description of the algorithm by Mathworks'] for a more in-depth explanation of these parameters.
    ///
    /// ['the description of the algorithm by Mathworks']: https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
    pub fn new(alpha: f64, beta: f64, kappa: f64, x0: State<S>, p0: Covariance<S>) -> Self {
        // Weights TODO might need to check these with different implementations
        let mut wm: [f64; 2 * S + 1] = [0.0; 2 * S + 1];
        let mut wc: [f64; 2 * S + 1] = [0.0; 2 * S + 1];

        wm[0] = 1.0 - S as f64 / (alpha.powi(2) * (S as f64 + kappa));
        wc[0] = (2.0 - alpha.powi(2) + beta) - S as f64 / (alpha.powi(2) * (S as f64 + kappa));

        for i in 1..(2 * S + 1) {
            wm[i] = 1.0 / (2.0 * alpha.powi(2) + (S as f64 + kappa));
            wc[i] = 1.0 / (2.0 * alpha.powi(2) + (S as f64 + kappa));
        }

        let c = alpha.powi(2) * (S as f64 + kappa);

        Self {
            state: x0,
            covariance: p0,
            wm: wm,
            wc: wc,
            c: c,
        }
    }
    /// Calculate the sigmapoint deltas. Used for adjusting the sigmapoints at every iteration. Takes the current covariance of the filter state.
    pub fn calc_sp_deltas(&self, p: &Covariance<S>) -> [State<S>; 2 * S] {
        let mut deltas = [State::<S>::zeros(); 2 * S];

        let cp = self.c * p;

        let lower = cp.cholesky().unwrap().l();

        for i in 0..S {
            deltas[i].set_column(0, &lower.column(i));
        }
        for i in S..2 * S {
            deltas[i].set_column(0, &lower.column(i - S));
            deltas[i] *= -1.0;
        }

        return deltas;
    }

    /// Calculate the sigma points using the filter state and its covariance.
    pub fn calc_sp(&self, x: &State<S>, p: &Covariance<S>) -> [State<S>; 2 * S + 1] {
        let mut sp = [State::<S>::zeros(); 2 * S + 1];
        let deltas = self.calc_sp_deltas(p);
        sp[0] = *x;
        for i in 1..(2 * S + 1) {
            sp[i] = sp[0] + deltas[i - 1];
        }
        return sp;
    }

    /// Run the filter state update. Takes the filter estimate from timestep k given all measurements up to and including k to timestep k+1 given all measurements up to and including k. Takes the current control of the system, the dynamics function, and the covariance of the assumed zero-mean process noise.
    pub fn update<const C: usize, D>(
        &mut self,
        control_k: &Control<C>,
        dt: f64,
        mut dynamics_model: D,
        dynamics_model_cov: &Covariance<S>,
    ) where
        D: FnMut(&State<S>, &Control<C>, &f64) -> State<S>,
    {
        // Sigma points for update
        let sp_k_k = self.calc_sp(&self.state, &self.covariance);

        // Apply dynamics model to the sigma points at k given k
        let mut sp_kp1_k = [State::<S>::zeros(); 2 * S + 1];
        for i in 0..(2 * S + 1) {
            sp_kp1_k[i] = dynamics_model(&sp_k_k[i], &control_k, &dt);
        }

        // Obtain the state at time k+1 given k
        let mut x_kp1_k = State::<S>::zeros();
        for i in 0..(2 * S + 1) {
            x_kp1_k += self.wm[i] * sp_kp1_k[i];
        }

        // Compute predicted state covariance at k+1 given k
        let mut cov_kp1_k = Covariance::<S>::zeros();
        for i in 0..(2 * S + 1) {
            cov_kp1_k += self.wc[i] * (sp_kp1_k[i] - x_kp1_k) * (sp_kp1_k[i] - x_kp1_k).transpose();
        }
        cov_kp1_k += dynamics_model_cov; // Adding additive process noise

        self.state = x_kp1_k;
        self.covariance = cov_kp1_k;
    }

    /// Run a filter measurement update. Takes the filter estimate from timestep k given all measurements relevant to the provided measurement function up to and including k-1 to timestep k given all measurements relevant to the provided measurement function up to and including k. Takes the current control of the system, the relevant measurement values (output), the measurement function, and the covariance of the assumed zero-mean measurement noise.
    pub fn innovate<const C: usize, const Y: usize, M>(
        &mut self,
        control_k: &Control<C>,
        output_k: &Output<Y>,
        mut measurement_model: M,
        measurement_model_cov: &Covariance<Y>,
    ) where
        M: FnMut(&State<S>, &Control<C>) -> Output<Y>,
    {
        // Sigma points for update from measurements at the current step k given k-1
        let sp_k_km1 = self.calc_sp(&self.state, &self.covariance);

        // Apply measurement model to the sigma points to get predicted measurements at k given k-1
        let mut y_k_km1 = [Output::<Y>::zeros(); 2 * S + 1];
        for i in 0..(2 * S + 1) {
            y_k_km1[i] = measurement_model(&sp_k_km1[i], &control_k);
        }

        // Combine predicted sigma points to get predicted measurement at k
        let mut y_k = Output::<Y>::zeros();
        for i in 0..(2 * S + 1) {
            y_k += self.wm[i] * y_k_km1[i];
        }

        // Estimate covariance for the predicted measurement at k
        let mut cov_y = Covariance::<Y>::zeros();
        for i in 0..(2 * S + 1) {
            cov_y += self.wc[i] * (y_k_km1[i] - y_k) * (y_k_km1[i] - y_k).transpose();
        }
        cov_y += measurement_model_cov; // Additive noise

        // Estimate cross covariance between predicted measurement at k and state at k given k-1
        let mut p_xy = CovarianceSY::<S, Y>::zeros();
        for i in 1..(2 * S + 1) {
            p_xy += (sp_k_km1[i] - sp_k_km1[0]) * (y_k_km1[i] - y_k).transpose();
        }
        p_xy *= self.wm[1];

        // Calculate Kalman gain
        let k = p_xy * cov_y.try_inverse().unwrap();

        // Calculate new state and covariance
        self.state = sp_k_km1[0] + k * (output_k - y_k);
        self.covariance = self.covariance - k * cov_y * k.transpose();
    }

    /// A convenience function. This runs both a measurement update and a state update in that order. The arguments are passed directly on to those functions.
    ///
    /// This can be used when there is only one measurement function and therefore no need to run the steps separately.
    pub fn progress<const C: usize, const Y: usize, D, M>(
        &mut self,
        dt: f64,
        control_k: Control<C>,
        output_k: Output<Y>,
        dynamics_model: D,
        dynamics_model_cov: &Covariance<S>,
        measurement_model: M,
        measurement_model_cov: &Covariance<Y>,
    ) where
        D: FnMut(&State<S>, &Control<C>, &f64) -> State<S>,
        M: FnMut(&State<S>, &Control<C>) -> Output<Y>,
    {
        self.innovate(
            &control_k,
            &output_k,
            measurement_model,
            measurement_model_cov,
        );
        self.update(&control_k, dt, dynamics_model, dynamics_model_cov);
    }
}
