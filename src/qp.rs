//! Dense quadratic programming solver with pluggable backends.
//!
//! Solves problems of the form:
//!
//! $$\min_x \frac{1}{2} x^T H x + c^T x$$
//!
//! subject to:
//! - $A_{eq}\, x = b_{eq}$  (equality constraints)
//! - $A_{iq}\, x \le b_{iq}$  (inequality constraints)
//!
//! # Backends
//!
//! | `QpSolver` variant | Algorithm | Feature flag |
//! |----|-------|----|
//! | `ActiveSet` | Primal active-set (dense, self-contained) | *always available* |
//! | `Clarabel` | Interior-point conic solver ([clarabel](https://crates.io/crates/clarabel)) | `clarabel` |
//!
//! The default backend is `ActiveSet`.  To use Clarabel, enable the `clarabel`
//! Cargo feature and set `QpSolver::Clarabel` in your [`QpConfig`].
//!
//! # Example
//!
//! ```
//! use nalgebra::{DMatrix, DVector};
//! use misa_wbc::qp::{solve_qp, QpConfig, QpSolver, QpStatus};
//!
//! // min 0.5*((x₁−2)² + (x₂−2)²)  s.t.  x₁ ≤ 1, x₂ ≤ 1
//! let h = DMatrix::identity(2, 2);
//! let c = DVector::from_vec(vec![-2.0, -2.0]); // c = -[2, 2]
//! let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
//! let b_iq = DVector::from_vec(vec![1.0, 1.0]);
//!
//! let sol = solve_qp(&h, &c, None, None,
//!                    Some(&a_iq), Some(&b_iq), None, &QpConfig::default());
//! assert_eq!(sol.status, QpStatus::Optimal);
//! assert!((sol.x[0] - 1.0).abs() < 1e-6);
//! assert!((sol.x[1] - 1.0).abs() < 1e-6);
//! ```

use nalgebra::{DMatrix, DVector};

// ─── Solver backend selection ───────────────────────────────────────────────

/// Which QP solver backend to use.
///
/// New variants can be added here (e.g. `Osqp`, `Proxqp`) to extend the set
/// of available solvers without breaking existing code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum QpSolver {
    /// Built-in primal active-set method (dense, no external dependencies).
    /// Efficient for the small QPs (n ≤ 50) typical in constrained IK.
    #[default]
    ActiveSet,
    /// Clarabel interior-point conic solver.
    /// Requires the `clarabel` Cargo feature.
    Clarabel,
}

// ─── Public types ───────────────────────────────────────────────────────────

/// Status of the QP solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// KKT conditions satisfied within tolerance.
    Optimal,
    /// Active-set iteration limit exceeded.
    MaxIterations,
    /// No feasible point could be found.
    Infeasible,
    /// A singular matrix or other numerical issue was encountered.
    NumericalFailure,
}

/// Solution returned by [`solve_qp`].
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// Optimal (or best-found) decision variable.
    pub x: DVector<f64>,
    /// Objective value $\frac{1}{2} x^T H x + c^T x$.
    pub objective: f64,
    /// Lagrange multipliers for equality constraints (length `m_eq`).
    pub lambda_eq: DVector<f64>,
    /// Lagrange multipliers for inequality constraints (length `m_iq`).
    /// Non-zero only for active inequalities at the solution.
    pub lambda_iq: DVector<f64>,
    /// Solver status.
    pub status: QpStatus,
    /// Number of active-set iterations performed.
    pub iterations: usize,
}

/// Configuration parameters for [`solve_qp`].
#[derive(Debug, Clone)]
pub struct QpConfig {
    /// Which solver backend to use.
    pub solver: QpSolver,
    /// Maximum active-set iterations.
    pub max_iters: usize,
    /// Tolerance for constraint feasibility checks.
    pub feasibility_tol: f64,
    /// Tolerance for step-norm and multiplier optimality checks.
    pub optimality_tol: f64,
    /// Proximal warm-start weight. When > 0 **and** an `x0` is passed
    /// to [`solve_qp`], the cost is augmented with
    /// `(prox_weight / 2) · ‖x − x0‖²`. The augmentation biases the
    /// optimum toward `x0`, which is the cheapest way to dampen
    /// tick-to-tick jitter when the same QP is solved repeatedly with
    /// slightly perturbed data and a wide null space (typical of WBC).
    ///
    /// Backend handling:
    /// - **ActiveSet**: in addition to the prox cost, `x0` is still
    ///   used as the initial feasible point (existing behaviour).
    /// - **Clarabel**: the IPM has no public warm-start hook in
    ///   clarabel 0.11, but the prox augmentation reshapes the
    ///   problem so the new optimum is close to `x0` (operator-
    ///   splitting-style warm-start at the cost level).
    ///
    /// 0.0 disables the prox term (default — preserves the original
    /// solver behaviour for callers that don't need warm-start).
    pub prox_weight: f64,
}

impl Default for QpConfig {
    fn default() -> Self {
        Self {
            solver: QpSolver::default(),
            max_iters: 500,
            feasibility_tol: 1e-10,
            optimality_tol: 1e-8,
            prox_weight: 0.0,
        }
    }
}

// ─── Solver (dispatch) ──────────────────────────────────────────────────────

/// Solve a dense QP, dispatching to the backend specified in `config.solver`.
///
/// # Arguments
///
/// * `h` — Hessian (n × n), must be positive (semi-)definite.
/// * `c` — Linear cost (n).
/// * `a_eq`, `b_eq` — Equality constraints $A_{eq} x = b_{eq}$.
///   Pass `None` for both when there are no equalities.
/// * `a_iq`, `b_iq` — Inequality constraints $A_{iq} x \le b_{iq}$.
///   Pass `None` for both when there are no inequalities.
/// * `x0` — Optional initial feasible point (used only by `ActiveSet`).
/// * `config` — Solver parameters (includes backend selection).
pub fn solve_qp(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
) -> QpSolution {
    // Apply proximal warm-start: when prox_weight > 0 AND x0 is given,
    // augment the cost with (ρ/2)·‖x − x0‖² = (ρ/2)·xᵀx − ρ·x0ᵀx + const.
    // → H' = H + ρ·I,  c' = c − ρ·x0
    // The augmented problem has the same constraints and a unique-r
    // optimum close to x0 (controlled by ρ). The const term shifts the
    // objective value but doesn't affect the argmin or the multipliers.
    let (h_owned, c_owned) = if config.prox_weight > 0.0 {
        if let Some(x0v) = x0 {
            let n = h.nrows();
            assert_eq!(
                x0v.nrows(),
                n,
                "solve_qp: x0 length must match H dimension when prox_weight > 0"
            );
            let mut h_aug = h.clone();
            let mut c_aug = c.clone();
            for i in 0..n {
                h_aug[(i, i)] += config.prox_weight;
                c_aug[i] -= config.prox_weight * x0v[i];
            }
            (Some(h_aug), Some(c_aug))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    let h_eff: &DMatrix<f64> = h_owned.as_ref().unwrap_or(h);
    let c_eff: &DVector<f64> = c_owned.as_ref().unwrap_or(c);

    let mut sol = match config.solver {
        QpSolver::ActiveSet => {
            solve_qp_active_set(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, x0, config)
        }
        QpSolver::Clarabel => {
            #[cfg(feature = "clarabel")]
            {
                solve_qp_clarabel(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, config)
            }
            #[cfg(not(feature = "clarabel"))]
            {
                panic!(
                    "QpSolver::Clarabel requires the `clarabel` Cargo feature.\n\
                     Add `misa-wbc = {{ features = [\"clarabel\"] }}` to your Cargo.toml."
                );
            }
        }
    };

    // If we ran on an augmented (h_eff, c_eff), report `objective` for the
    // **original** problem — callers expect ½ xᵀHx + cᵀx, not the prox-
    // augmented value. Multipliers are unchanged because the prox term
    // is an unconstrained quadratic addition (gradient at x* matches).
    if h_owned.is_some() {
        sol.objective = 0.5 * sol.x.dot(&(h * &sol.x)) + c.dot(&sol.x);
    }
    sol
}

// ─── Active-set backend ─────────────────────────────────────────────────────

/// Built-in primal active-set QP solver.
fn solve_qp_active_set(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
) -> QpSolution {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!(c.nrows(), n, "c length must match H dimension");

    // ── Unpack / default equality & inequality matrices ──────────────
    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();

    // ── Cholesky of H (with regularisation fallback) ─────────────────
    let chol = match h.clone().cholesky() {
        Some(c) => c,
        None => {
            let h_reg = h + &DMatrix::identity(n, n) * 1e-12;
            match h_reg.cholesky() {
                Some(c) => c,
                None => return fail(n, m_eq, m_iq, QpStatus::NumericalFailure),
            }
        }
    };

    // ── Initial feasible point ───────────────────────────────────────
    let mut x = match x0 {
        Some(v) => {
            assert_eq!(v.nrows(), n, "x0 length must match H dimension");
            v.clone()
        }
        None => initial_feasible(n, &ae, &be, &ai, &bi, config),
    };

    // Verify feasibility
    if m_eq > 0 {
        let residual = (&ae * &x - &be).norm();
        if residual > config.feasibility_tol * (1.0 + be.norm().max(1.0)) {
            return fail(n, m_eq, m_iq, QpStatus::Infeasible);
        }
    }
    if m_iq > 0 {
        let vals = &ai * &x;
        for i in 0..m_iq {
            if vals[i] > bi[i] + config.feasibility_tol {
                return fail(n, m_eq, m_iq, QpStatus::Infeasible);
            }
        }
    }

    // ── Working set: active inequalities at x ────────────────────────
    let mut ws: Vec<usize> = Vec::new();
    if m_iq > 0 {
        let vals = &ai * &x;
        for i in 0..m_iq {
            if vals[i] >= bi[i] - config.feasibility_tol {
                ws.push(i);
            }
        }
    }

    // ── Active-set iterations ────────────────────────────────────────
    let mut lam_eq = DVector::zeros(m_eq);
    let mut lam_iq = DVector::zeros(m_iq);

    for iter in 0..config.max_iters {
        let grad = h * &x + c;
        let m_w = ws.len();
        let m_act = m_eq + m_w;

        if m_act == 0 {
            // Unconstrained step
            let p = chol.solve(&(-&grad));
            if p.norm() < config.optimality_tol {
                return optimal(x, h, c, lam_eq, lam_iq, iter);
            }
            let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &ws, config);
            x += alpha * &p;
            if let Some(idx) = blocking {
                ws.push(idx);
            }
        } else {
            // Build active constraint matrix  Â = [A_eq; A_W]
            let a_act = build_active_matrix(&ae, &ai, &ws, m_eq, n);

            // Schur complement: S = Â H⁻¹ Âᵀ
            let h_inv_at = chol.solve(&a_act.transpose());
            let s = &a_act * &h_inv_at;

            let r = -&grad;
            let h_inv_r = chol.solve(&r);
            let rhs = &a_act * &h_inv_r;

            let nu = match s.clone().lu().solve(&rhs) {
                Some(v) => v,
                None => {
                    // Singular Schur complement → dependent constraints
                    if m_w > 0 {
                        ws.pop();
                        continue;
                    }
                    return make_sol(x, h, c, lam_eq, lam_iq, QpStatus::NumericalFailure, iter);
                }
            };

            let p = &h_inv_r - &h_inv_at * &nu;

            if p.norm() < config.optimality_tol {
                // ── Check multipliers for active inequalities ────
                let mut all_ok = true;
                let mut worst_val = 0.0;
                let mut worst_k = 0usize;

                for k in 0..m_w {
                    let mu = nu[m_eq + k];
                    if mu < -config.optimality_tol && mu < worst_val {
                        all_ok = false;
                        worst_val = mu;
                        worst_k = k;
                    }
                }

                if all_ok {
                    for i in 0..m_eq {
                        lam_eq[i] = nu[i];
                    }
                    for (k, &wi) in ws.iter().enumerate() {
                        lam_iq[wi] = nu[m_eq + k];
                    }
                    return optimal(x, h, c, lam_eq, lam_iq, iter);
                } else {
                    ws.remove(worst_k);
                }
            } else {
                let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &ws, config);
                x += alpha * &p;
                if let Some(idx) = blocking {
                    ws.push(idx);
                }
            }
        }
    }

    make_sol(x, h, c, lam_eq, lam_iq, QpStatus::MaxIterations, config.max_iters)
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn unpack_pair(
    a: Option<&DMatrix<f64>>,
    b: Option<&DVector<f64>>,
    n: usize,
    name: &str,
) -> (DMatrix<f64>, DVector<f64>) {
    match (a, b) {
        (Some(a), Some(b)) => {
            assert_eq!(a.ncols(), n, "{name}: column count must match n");
            assert_eq!(a.nrows(), b.nrows(), "{name}: row counts must match");
            (a.clone(), b.clone())
        }
        (None, None) => (DMatrix::zeros(0, n), DVector::zeros(0)),
        _ => panic!("{name}: must both be Some or both be None"),
    }
}

fn initial_feasible(
    n: usize,
    ae: &DMatrix<f64>,
    be: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    config: &QpConfig,
) -> DVector<f64> {
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();

    if m_eq == 0 {
        return DVector::zeros(n);
    }

    // Least-norm: x = Aᵀ (A Aᵀ)⁻¹ b
    let aat = ae * ae.transpose();
    let x0 = match aat.clone().lu().solve(be) {
        Some(y) => ae.transpose() * y,
        None => return DVector::zeros(n),
    };

    // Check inequality feasibility
    if m_iq == 0 {
        return x0;
    }
    let vals = ai * &x0;
    let mut feasible = true;
    for i in 0..m_iq {
        if vals[i] > bi[i] + config.feasibility_tol {
            feasible = false;
            break;
        }
    }
    if feasible {
        return x0;
    }

    // The least-norm equality-feasible point violates some inequality.
    // Project into the feasible set via the null space of A_eq.
    // Null-space projector: P = I − Aᵀ (A Aᵀ)⁻¹ A
    let aat_inv = match aat.lu().solve(&DMatrix::identity(m_eq, m_eq)) {
        Some(v) => v,
        None => return x0, // fallback
    };
    let proj_null = DMatrix::identity(n, n) - ae.transpose() * &aat_inv * ae;

    let mut x = x0;
    for _ in 0..200 {
        let vals = ai * &x;
        let mut max_viol = f64::NEG_INFINITY;
        let mut worst = 0usize;
        for i in 0..m_iq {
            let v = vals[i] - bi[i];
            if v > max_viol {
                max_viol = v;
                worst = i;
            }
        }
        if max_viol <= config.feasibility_tol {
            return x;
        }

        // Move x along the null-space projection of a_worst to reduce violation.
        let ai_col: DVector<f64> = ai.row(worst).transpose().into_owned();
        let p_ai = &proj_null * &ai_col;
        let denom = ai_col.dot(&p_ai);
        if denom < 1e-15 {
            break; // cannot move in null space
        }
        let alpha = max_viol / denom;
        x -= alpha * p_ai;
    }
    x
}

fn build_active_matrix(
    ae: &DMatrix<f64>,
    ai: &DMatrix<f64>,
    ws: &[usize],
    m_eq: usize,
    n: usize,
) -> DMatrix<f64> {
    let m_act = m_eq + ws.len();
    let mut a = DMatrix::zeros(m_act, n);
    for i in 0..m_eq {
        a.row_mut(i).copy_from(&ae.row(i));
    }
    for (k, &wi) in ws.iter().enumerate() {
        a.row_mut(m_eq + k).copy_from(&ai.row(wi));
    }
    a
}

fn step_length(
    x: &DVector<f64>,
    p: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    ws: &[usize],
    config: &QpConfig,
) -> (f64, Option<usize>) {
    let m_iq = ai.nrows();
    if m_iq == 0 {
        return (1.0, None);
    }
    let mut alpha = 1.0;
    let mut blocking = None;

    for i in 0..m_iq {
        if ws.contains(&i) {
            continue;
        }
        let ai_p = row_dot(ai, i, p);
        if ai_p > config.feasibility_tol {
            let slack = bi[i] - row_dot(ai, i, x);
            let alpha_i = (slack / ai_p).max(0.0);
            if alpha_i < alpha {
                alpha = alpha_i;
                blocking = Some(i);
            }
        }
    }
    (alpha, blocking)
}

/// Row-vector · column-vector dot product (avoids nalgebra shape mismatch).
#[inline]
fn row_dot(mat: &DMatrix<f64>, row: usize, v: &DVector<f64>) -> f64 {
    let n = v.nrows();
    let mut s = 0.0;
    for k in 0..n {
        s += mat[(row, k)] * v[k];
    }
    s
}

fn fail(n: usize, m_eq: usize, m_iq: usize, status: QpStatus) -> QpSolution {
    QpSolution {
        x: DVector::zeros(n),
        objective: 0.0,
        lambda_eq: DVector::zeros(m_eq),
        lambda_iq: DVector::zeros(m_iq),
        status,
        iterations: 0,
    }
}

fn optimal(
    x: DVector<f64>,
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    lam_eq: DVector<f64>,
    lam_iq: DVector<f64>,
    iters: usize,
) -> QpSolution {
    make_sol(x, h, c, lam_eq, lam_iq, QpStatus::Optimal, iters)
}

fn make_sol(
    x: DVector<f64>,
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    lam_eq: DVector<f64>,
    lam_iq: DVector<f64>,
    status: QpStatus,
    iters: usize,
) -> QpSolution {
    let obj = 0.5 * x.dot(&(h * &x)) + c.dot(&x);
    QpSolution {
        x,
        objective: obj,
        lambda_eq: lam_eq,
        lambda_iq: lam_iq,
        status,
        iterations: iters,
    }
}

// ─── Clarabel backend ───────────────────────────────────────────────────────

/// Clarabel interior-point conic solver backend.
///
/// Converts the dense QP to the Clarabel native format:
/// - Hessian `P` as upper-triangular CscMatrix
/// - Constraint matrix `A` stacks equality rows ($A_{eq} x = b_{eq}$) and
///   inequality rows ($A_{iq} x \le b_{iq}$)
/// - Cone spec: `ZeroConeT` for equalities, `NonnegativeConeT` for
///   inequalities (slack form)
#[cfg(feature = "clarabel")]
fn solve_qp_clarabel(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    config: &QpConfig,
) -> QpSolution {
    use clarabel::solver::{
        DefaultSettings, DefaultSettingsBuilder, DefaultSolver, IPSolver,
        SolverStatus, SupportedConeT,
    };

    let n = h.nrows();
    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();
    let m_total = m_eq + m_iq;

    // ── Build Hessian P (upper-triangular CSC) ──────────────────────
    let p = dense_to_csc_upper(h);

    // ── Build linear cost q ─────────────────────────────────────────
    let q: Vec<f64> = c.iter().copied().collect();

    // ── Build constraint matrix A (CSC) ─────────────────────────────
    //   [  A_eq  ]        [  b_eq  ]
    //   [  A_iq  ]  x ≤   [  b_iq  ]
    //
    // Clarabel standard form:  A x + s = b,  s ∈ K
    //   ZeroConeT(m_eq)         : s = 0  →  A_eq x = b_eq
    //   NonnegativeConeT(m_iq)  : s ≥ 0  →  b_iq - A_iq x ≥ 0  →  A_iq x ≤ b_iq
    let mut a_dense = DMatrix::zeros(m_total, n);
    let mut b_vec = Vec::with_capacity(m_total);

    for i in 0..m_eq {
        for j in 0..n {
            a_dense[(i, j)] = ae[(i, j)];
        }
        b_vec.push(be[i]);
    }
    for i in 0..m_iq {
        for j in 0..n {
            a_dense[(m_eq + i, j)] = ai[(i, j)];
        }
        b_vec.push(bi[i]);
    }

    let a_csc = dense_to_csc_full(&a_dense);

    // ── Cone specification ──────────────────────────────────────────
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if m_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(m_eq));
    }
    if m_iq > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(m_iq));
    }

    // ── Solver settings ─────────────────────────────────────────────
    let settings = DefaultSettingsBuilder::default()
        .max_iter(config.max_iters as u32)
        .tol_gap_abs(config.optimality_tol)
        .tol_gap_rel(config.optimality_tol)
        .tol_feas(config.feasibility_tol)
        .verbose(false)
        .build()
        .unwrap_or_else(|_| DefaultSettings::default());

    // ── Solve ───────────────────────────────────────────────────────
    let mut solver = DefaultSolver::new(&p, &q, &a_csc, &b_vec, &cones, settings)
        .expect("Clarabel: failed to construct solver (bad problem dimensions?)");
    solver.solve();

    let status = match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => QpStatus::Optimal,
        SolverStatus::MaxIterations => QpStatus::MaxIterations,
        SolverStatus::PrimalInfeasible
        | SolverStatus::DualInfeasible
        | SolverStatus::AlmostPrimalInfeasible
        | SolverStatus::AlmostDualInfeasible => QpStatus::Infeasible,
        _ => QpStatus::NumericalFailure,
    };

    let x = DVector::from_vec(solver.solution.x.clone());

    // Extract multipliers from the dual variable z.
    // Clarabel dual z has length m_total = m_eq + m_iq.
    let z = &solver.solution.z;
    let mut lam_eq = DVector::zeros(m_eq);
    let mut lam_iq = DVector::zeros(m_iq);
    for i in 0..m_eq {
        lam_eq[i] = z[i];
    }
    for i in 0..m_iq {
        // Clarabel dual for NonnegativeCone: λ ≥ 0 for the slack constraint.
        lam_iq[i] = z[m_eq + i].max(0.0);
    }

    let obj = 0.5 * x.dot(&(h * &x)) + c.dot(&x);
    QpSolution {
        x,
        objective: obj,
        lambda_eq: lam_eq,
        lambda_iq: lam_iq,
        status,
        iterations: solver.solution.iterations as usize,
    }
}

/// Convert a dense (n×n) matrix to upper-triangular CscMatrix for Clarabel.
#[cfg(feature = "clarabel")]
fn dense_to_csc_upper(m: &DMatrix<f64>) -> clarabel::algebra::CscMatrix<f64> {
    let n = m.nrows();
    let mut col_ptr = vec![0usize; n + 1];
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();

    for j in 0..n {
        for i in 0..=j {
            let v = m[(i, j)];
            if v.abs() > 1e-15 {
                row_idx.push(i);
                vals.push(v);
            }
        }
        col_ptr[j + 1] = row_idx.len();
    }

    clarabel::algebra::CscMatrix::new(n, n, col_ptr, row_idx, vals)
}

/// Convert a dense (m×n) matrix to full CscMatrix for Clarabel.
#[cfg(feature = "clarabel")]
fn dense_to_csc_full(m: &DMatrix<f64>) -> clarabel::algebra::CscMatrix<f64> {
    let (rows, cols) = m.shape();
    let mut col_ptr = vec![0usize; cols + 1];
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();

    for j in 0..cols {
        for i in 0..rows {
            let v = m[(i, j)];
            if v.abs() > 1e-15 {
                row_idx.push(i);
                vals.push(v);
            }
        }
        col_ptr[j + 1] = row_idx.len();
    }

    clarabel::algebra::CscMatrix::new(rows, cols, col_ptr, row_idx, vals)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unconstrained_minimum() {
        // min 0.5*(x1² + 2*x2²) - 3*x1 - x2
        // H = diag(1, 2), c = [-3, -1]
        // Solution: x = H^{-1} (-c) = [3, 0.5]
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-3.0, -1.0]);

        let sol = solve_qp(&h, &c, None, None, None, None, None, &QpConfig::default());
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn inequality_active_at_optimum() {
        // min 0.5*(x1-2)² + 0.5*(x2-2)²  s.t.  x1 ≤ 1, x2 ≤ 1
        // H = I, c = [-2, -2]
        // Unconstrained: (2,2).  Constrained: (1,1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn inequality_not_active() {
        // min 0.5*(x1² + x2²)  s.t.  x1 ≤ 5, x2 ≤ 5
        // Unconstrained min at (0,0) is already feasible.
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![5.0, 5.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn one_active_one_inactive() {
        // min 0.5*(x1-3)² + 0.5*(x2-0.5)²  s.t.  x1 ≤ 1, x2 ≤ 2
        // Unconstrained: (3, 0.5).  x1 ≤ 1 is active, x2 ≤ 2 is not.
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-3.0, -0.5]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 2.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn box_constraints() {
        // min 0.5*(x1-5)² + 0.5*(x2+3)²  s.t.  -1 ≤ x ≤ 2
        // Unconstrained: (5, -3).  Box: (2, -1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-5.0, 3.0]);
        // x1 ≤ 2, x2 ≤ 2, -x1 ≤ 1, -x2 ≤ 1
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0,  0.0,
            0.0,  1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![2.0, 2.0, 1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn equality_only() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 = 1
        // Solution on the line x1+x2=1: closest to origin is (0.5, 0.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            None, None, None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn equality_and_inequality() {
        // min 0.5*(x1-3)² + 0.5*(x2-3)²  s.t.  x1 + x2 = 2, x1 ≥ 0, x2 ≥ 0
        // On the line x1+x2=2, closest to (3,3) is (1,1).
        // But with x1 ≥ 0, x2 ≥ 0 (not active), solution is still (1,1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-3.0, -3.0]);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 2.0);
        // -x1 ≤ 0, -x2 ≤ 0
        let a_iq = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
        let b_iq = DVector::zeros(2);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            Some(&a_iq), Some(&b_iq),
            None,
            &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn equality_and_active_inequality() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 = 2, x1 ≤ 0.5
        // On x1+x2=2 the unconstrained closest-to-origin is (1,1). But x1 ≤ 0.5
        // forces (0.5, 1.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 2.0);
        let a_iq = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let b_iq = DVector::from_element(1, 0.5);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            Some(&a_iq), Some(&b_iq),
            None,
            &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn user_provided_x0() {
        // Same as box_constraints but with user-provided x0 = (0,0).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-5.0, 3.0]);
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![2.0, 2.0, 1.0, 1.0]);
        let x0 = DVector::from_vec(vec![0.0, 0.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn larger_problem() {
        // min 0.5 ||x||²  s.t.  Σx_i ≥ 5 and 0 ≤ x_i ≤ 3 for i=0..4
        // Closest to origin on Σx≥5 with box: each x_i = 1 (sum = 5).
        let n = 5;
        let h = DMatrix::identity(n, n);
        let c = DVector::zeros(n);

        // -Σx_i ≤ -5, x_i ≤ 3, -x_i ≤ 0
        let mut rows = Vec::new();
        // sum >= 5
        let mut sum_row = vec![0.0; n];
        for v in &mut sum_row {
            *v = -1.0;
        }
        rows.push((sum_row, -5.0));
        for i in 0..n {
            let mut row_upper = vec![0.0; n];
            row_upper[i] = 1.0;
            rows.push((row_upper, 3.0));
            let mut row_lower = vec![0.0; n];
            row_lower[i] = -1.0;
            rows.push((row_lower, 0.0));
        }

        let m = rows.len();
        let mut a_data = Vec::with_capacity(m * n);
        let mut b_data = Vec::with_capacity(m);
        for (r, b_val) in &rows {
            a_data.extend_from_slice(r);
            b_data.push(*b_val);
        }
        let a_iq = DMatrix::from_row_slice(m, n, &a_data);
        let b_iq = DVector::from_vec(b_data);

        let x0 = DVector::from_element(n, 1.0); // feasible start: sum=5
        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        for i in 0..n {
            assert_relative_eq!(sol.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn objective_value_correct() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        // x = (1,1), obj = 0.5*(1+1) + (-2-2) = 1 - 4 = -3
        assert_relative_eq!(sol.objective, -3.0, epsilon = 1e-8);
    }

    #[test]
    fn multipliers_positive_for_active_inequality() {
        // min 0.5*(x-2)²  s.t.  x ≤ 1
        // Active at x=1, multiplier = ∂f/∂b = 2-1 = 1
        let h = DMatrix::from_element(1, 1, 1.0);
        let c = DVector::from_element(1, -2.0);
        let a_iq = DMatrix::from_element(1, 1, 1.0);
        let b_iq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-10);
        assert!(sol.lambda_iq[0] > 0.0, "active inequality multiplier should be positive");
        assert_relative_eq!(sol.lambda_iq[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn coupled_inequality_constraints() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 ≤ 1, x1 - x2 ≤ 1
        // Unconstrained: (0,0), which satisfies both → solution is (0,0).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, -1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn coupled_inequality_active() {
        // min 0.5*((x1-2)² + (x2-2)²)  s.t.  x1 + x2 ≤ 1
        // Unconstrained: (2,2), violates x1+x2≤1.
        // Constrained min on x1+x2=1 closest to (2,2): (0.5, 0.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_iq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn non_identity_hessian() {
        // min 0.5*(3*x1² + x2² + 2*x1*x2) - x1  s.t.  -1 ≤ x ≤ 1
        // H = [[3, 1], [1, 1]], c = [-1, 0]
        // Unconstrained: H^{-1}(-c) = [[1,-1],[-1,3]]/2 * [1,0] = [0.5, -0.5]
        let h = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0]);
        let c = DVector::from_vec(vec![-1.0, 0.0]);
        // box: x1 ≤ 1, x2 ≤ 1, -x1 ≤ 1, -x2 ≤ 1
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], -0.5, epsilon = 1e-6);
    }

    // ── Clarabel cross-validation tests ─────────────────────────────

    /// Helper: solve the same QP with both ActiveSet and Clarabel and compare.
    #[cfg(feature = "clarabel")]
    fn cross_validate(
        h: &DMatrix<f64>,
        c: &DVector<f64>,
        a_eq: Option<&DMatrix<f64>>,
        b_eq: Option<&DVector<f64>>,
        a_iq: Option<&DMatrix<f64>>,
        b_iq: Option<&DVector<f64>>,
        x0: Option<&DVector<f64>>,
        tol: f64,
    ) {
        let cfg_as = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        let cfg_cl = QpConfig { solver: QpSolver::Clarabel, ..Default::default() };
        let sol_as = solve_qp(h, c, a_eq, b_eq, a_iq, b_iq, x0, &cfg_as);
        let sol_cl = solve_qp(h, c, a_eq, b_eq, a_iq, b_iq, None, &cfg_cl);
        assert_eq!(sol_as.status, QpStatus::Optimal, "ActiveSet not Optimal");
        assert_eq!(sol_cl.status, QpStatus::Optimal, "Clarabel not Optimal");
        assert_relative_eq!(sol_as.objective, sol_cl.objective, epsilon = tol);
        for i in 0..sol_as.x.len() {
            assert_relative_eq!(sol_as.x[i], sol_cl.x[i], epsilon = tol);
        }
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_unconstrained() {
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-3.0, -1.0]);
        cross_validate(&h, &c, None, None, None, None, None, 1e-6);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_inequality_active() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);
        cross_validate(&h, &c, None, None, Some(&a_iq), Some(&b_iq), None, 1e-6);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_equality_only() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 1.0);
        cross_validate(&h, &c, Some(&a_eq), Some(&b_eq), None, None, None, 1e-6);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_equality_and_inequality() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 2.0);
        let a_iq = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let b_iq = DVector::from_element(1, 0.5);
        cross_validate(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            Some(&a_iq), Some(&b_iq),
            None,
            1e-6,
        );
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_box_constraints() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-5.0, 3.0]);
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![2.0, 2.0, 1.0, 1.0]);
        cross_validate(&h, &c, None, None, Some(&a_iq), Some(&b_iq), None, 1e-6);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_larger_problem() {
        let n = 5;
        let h = DMatrix::identity(n, n);
        let c = DVector::zeros(n);
        let mut rows = Vec::new();
        let mut sum_row = vec![0.0; n];
        for v in &mut sum_row { *v = -1.0; }
        rows.push((sum_row, -5.0));
        for i in 0..n {
            let mut r_u = vec![0.0; n]; r_u[i] = 1.0;
            rows.push((r_u, 3.0));
            let mut r_l = vec![0.0; n]; r_l[i] = -1.0;
            rows.push((r_l, 0.0));
        }
        let m = rows.len();
        let mut a_data = Vec::with_capacity(m * n);
        let mut b_data = Vec::with_capacity(m);
        for (r, b_val) in &rows {
            a_data.extend_from_slice(r);
            b_data.push(*b_val);
        }
        let a_iq = DMatrix::from_row_slice(m, n, &a_data);
        let b_iq = DVector::from_vec(b_data);
        let x0 = DVector::from_element(n, 1.0); // feasible start: sum=5
        cross_validate(&h, &c, None, None, Some(&a_iq), Some(&b_iq), Some(&x0), 1e-5);
    }

    /// Proximal warm-start: a tiny prox term should not move the
    /// optimum noticeably when the original problem is already strictly
    /// convex (unique optimum). ρ = 1e-6 is small relative to H = I.
    #[test]
    fn prox_weight_preserves_strict_optimum() {
        // Same as `inequality_active_at_optimum` but with prox toward
        // the (incorrect) origin. The strict optimum (1, 1) is unaffected
        // up to O(ρ) bias.
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);
        let x0 = DVector::zeros(2);

        let cfg = QpConfig {
            prox_weight: 1e-6,
            ..QpConfig::default()
        };
        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &cfg,
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        // Objective must report on the **original** cost (½‖x‖² − 2(x₁+x₂)).
        // At x=(1,1): 0.5*(1+1) − 2*(1+1) = 1 − 4 = −3.
        assert_relative_eq!(sol.objective, -3.0, epsilon = 5e-6);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(sol.x[1], 1.0, epsilon = 1e-4);
    }

    /// Proximal warm-start: in a degenerate problem with a wide null
    /// space, the prox term should pull the optimum toward `x0`.
    #[test]
    fn prox_weight_picks_solution_near_x0_in_null_space() {
        // min 0.5*(x₁ + x₂ − 1)² (rank-1 H) — infinitely many optima
        // along the line x₁ + x₂ = 1. Without prox the solver could
        // pick anything on that line; with prox toward x0 = (0, 0)
        // the unique optimum is the closest point to origin: (0.5, 0.5).
        // We build H = aaᵀ where a = [1; 1].
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let c = DVector::from_vec(vec![-1.0, -1.0]);
        let x0 = DVector::zeros(2);

        let cfg = QpConfig {
            prox_weight: 1e-3,
            ..QpConfig::default()
        };
        let sol = solve_qp(
            &h, &c, None, None, None, None, Some(&x0), &cfg,
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        // Both coordinates should be close to 0.5 (the prox-anchored
        // closest point on the optimum line). Tolerance accounts for
        // ρ = 1e-3 finite bias.
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-2);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-2);
    }

    /// Proximal warm-start: prox_weight = 0 must keep the solver fully
    /// backward-compatible (no augmentation, x0 retains its existing
    /// "initial point" semantics for ActiveSet).
    #[test]
    fn prox_weight_zero_is_a_noop() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);
        let x0 = DVector::zeros(2); // far from optimum but irrelevant

        let cfg = QpConfig::default();
        assert_eq!(cfg.prox_weight, 0.0);
        let sol_no_x0 = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &cfg,
        );
        let sol_with_x0 = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &cfg,
        );
        // Both reach the same optimum (1, 1).
        assert_relative_eq!(sol_no_x0.x[0], sol_with_x0.x[0], epsilon = 1e-6);
        assert_relative_eq!(sol_no_x0.x[1], sol_with_x0.x[1], epsilon = 1e-6);
        assert_relative_eq!(sol_no_x0.objective, sol_with_x0.objective, epsilon = 1e-6);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn clarabel_non_identity_hessian() {
        let h = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0]);
        let c = DVector::from_vec(vec![-1.0, 0.0]);
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        cross_validate(&h, &c, None, None, Some(&a_iq), Some(&b_iq), None, 1e-6);
    }
}
