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

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

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
    /// Built-in primal-dual interior-point method (Mehrotra predictor-
    /// corrector), dense, no external dependencies. A from-scratch,
    /// textbook implementation — pedagogical / comparison counterpart
    /// to [`ActiveSet`](QpSolver::ActiveSet) and
    /// [`Clarabel`](QpSolver::Clarabel), which is *also* an interior-
    /// point method but a mature conic one; this variant exists to
    /// study the IPM approach itself, not to outperform Clarabel.
    Ipm,
    /// [`Ipm`](QpSolver::Ipm) plus Gondzio's multiple centrality
    /// correctors (Colombo & Gondzio 2008): after the usual
    /// predictor + Mehrotra-corrector step, up to a few extra cheap
    /// back-substitutions against the *same* factorisation nudge
    /// outlier complementarity products back toward the target,
    /// enlarging the step length. **Measured to help on large,
    /// sparse LPs (the setting the technique was designed for) but
    /// to be a net loss at misa-wbc's scale** — on dense random QPs
    /// from n=5 to n=320 the extra solves/line-search per iteration
    /// cost more than the (near-zero, at this density/scale) drop in
    /// outer iterations bought back. Kept as a separate variant
    /// (rather than folded into `Ipm`'s default behaviour) precisely
    /// because of that measured regression — pick it only to compare
    /// against `Ipm`, not as a general speed-up.
    IpmMcc,
    /// Built-in operator-splitting (ADMM) QP solver, dense, no
    /// external dependencies. A from-scratch, textbook implementation
    /// of the OSQP algorithm (Stellato et al. 2020) — a third
    /// paradigm alongside [`ActiveSet`](QpSolver::ActiveSet) (vertex
    /// hopping) and [`Ipm`](QpSolver::Ipm) (barrier path-following):
    /// it splits the problem into an equality-constrained QP (solved
    /// exactly via one linear system) and a box projection (solved in
    /// closed form), alternating between them. The linear system's
    /// matrix never changes across iterations, so it is factorised
    /// **once** and every iteration is just a solve — no incremental
    /// updates (active-set) or re-factorisation (interior-point)
    /// needed at all.
    Admm,
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

// ─── Warm-start workspace ───────────────────────────────────────────────────

/// Reusable warm-start state for the [`QpSolver::ActiveSet`] backend —
/// qpOASES's "online active set" idea, pragmatically: keep the previous
/// solution and working set, and let the next (identical or gently
/// perturbed) solve start from them instead of a cold Phase-1. For a
/// repeated QP the re-solve converges in O(1) iterations; for a
/// perturbed one the previous point is re-projected onto the new
/// equality manifold and the working set re-seeded.
///
/// Feed it to [`solve_qp_warm`]; every solve updates it in place.
/// Ignored by the Clarabel backend (which is warm-start-free here).
#[derive(Clone, Debug, Default)]
pub struct QpWorkspace {
    x: Option<DVector<f64>>,
    working_set: Vec<usize>,
}

impl QpWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop the stored state — the next solve starts cold.
    pub fn clear(&mut self) {
        self.x = None;
        self.working_set.clear();
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
    solve_qp_impl(h, c, a_eq, b_eq, a_iq, b_iq, x0, config, None)
}

/// [`solve_qp`] with a persistent [`QpWorkspace`]: the active-set
/// backend starts from the workspace's previous solution / working set
/// and stores the new ones back — the cross-tick warm start that makes
/// a sequence of similar QPs cheap. Other backends ignore the
/// workspace.
#[allow(clippy::too_many_arguments)]
pub fn solve_qp_warm(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
    workspace: &mut QpWorkspace,
) -> QpSolution {
    solve_qp_impl(h, c, a_eq, b_eq, a_iq, b_iq, x0, config, Some(workspace))
}

#[allow(clippy::too_many_arguments)]
fn solve_qp_impl(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
    workspace: Option<&mut QpWorkspace>,
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
            solve_qp_active_set(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, x0, config, workspace)
        }
        QpSolver::Ipm => solve_qp_ipm(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, config, false),
        QpSolver::IpmMcc => solve_qp_ipm(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, config, true),
        QpSolver::Admm => solve_qp_admm(h_eff, c_eff, a_eq, b_eq, a_iq, b_iq, config),
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

/// Built-in primal active-set QP solver, with two qpOASES-style
/// upgrades over the textbook method:
///
/// 1. **Incremental factor updates** ([`ActiveFactor`]): the Schur
///    complement `S = Â·H⁻¹·Âᵀ` and the cached `H⁻¹·Âᵀ` columns are
///    maintained across working-set changes (O(m²) append / delete)
///    instead of being rebuilt and LU-factorised every iteration
///    (O(n²·m + m³)).
/// 2. **Warm-started working set** ([`QpWorkspace`]): start from the
///    previous solve's optimum (re-projected onto the new equality
///    manifold) and its working set — a repeated QP re-solves in O(1)
///    iterations, a gently perturbed one in a few.
///
/// Plus Bland's anti-cycling rule as a fallback once the iteration
/// count suggests degeneracy.
#[allow(clippy::too_many_arguments)]
fn solve_qp_active_set(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
    mut workspace: Option<&mut QpWorkspace>,
) -> QpSolution {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!(c.nrows(), n, "c length must match H dimension");

    // ── Unpack / default equality & inequality matrices ──────────────
    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();

    // ── Cholesky of H, with qpOASES-style conditional ridge ──────────
    // An ill-conditioned H (κ ≳ 1e10 — e.g. AᵀA of a rank-deficient
    // stack plus a 1e-12 tie-break) makes the EQP steps blow up along
    // the near-null directions; the step-length clip then advances in
    // microscopic increments and the method crawls into MaxIterations.
    // qpOASES's answer is a ridge: iterate on H + ρᵣ·I with ρᵣ scaled
    // to H, then recover the unregularised optimum with one KKT polish
    // on the final active set (see the `Optimal` exit below).
    let mut ridge = 0.0_f64;
    let chol = {
        let attempt = |ridge: f64| -> Option<nalgebra::Cholesky<f64, Dyn>> {
            if ridge == 0.0 {
                h.clone().cholesky()
            } else {
                (h + DMatrix::identity(n, n) * ridge).cholesky()
            }
        };
        let mut c = attempt(0.0);
        // Conditioning check via the Cholesky pivots: κ(H) ≈ (max/min)².
        let ill = c.as_ref().is_none_or(|c| {
            let l = c.l_dirty();
            let mut lo = f64::INFINITY;
            let mut hi = 0.0_f64;
            for i in 0..n {
                let d = l[(i, i)].abs();
                lo = lo.min(d);
                hi = hi.max(d);
            }
            // κ(H) ≈ (hi/lo)² ≥ 1e8 already crawls (microscopic
            // step-length clipping); the polish restores exactness, so
            // over-ridging is safe and under-ridging is not.
            lo <= 1e-4 * hi
        });
        if ill {
            let scale = (0..n).map(|i| h[(i, i)].abs()).fold(0.0, f64::max).max(1e-12);
            // Cap the iteration Hessian at κ ≈ 1e6 — comfortably
            // inside the crawl-free regime; the polish undoes the bias.
            ridge = 1e-6 * scale;
            c = attempt(ridge);
        }
        match c {
            Some(c) => c,
            None => return fail(n, m_eq, m_iq, QpStatus::NumericalFailure),
        }
    };
    // The Hessian the ITERATIONS see (grad must match the factor).
    let h_owned_ridge;
    let h_it: &DMatrix<f64> = if ridge > 0.0 {
        h_owned_ridge = h + DMatrix::identity(n, n) * ridge;
        &h_owned_ridge
    } else {
        h
    };

    // ── Starting point: warm workspace → caller x0 → cold Phase-1 ────
    let mut seed: Vec<usize> = Vec::new();
    let mut warm = false;
    let mut x = 'start: {
        if let Some(w) = workspace.as_deref() {
            if let Some(px) = &w.x {
                if px.len() == n && w.working_set.iter().all(|&i| i < m_iq) {
                    let mut xw = px.clone();
                    if m_eq > 0 {
                        let r = &ae * &xw - &be;
                        if r.norm() > config.feasibility_tol * (1.0 + be.norm().max(1.0)) {
                            // Re-project onto the new equality manifold:
                            //   x ← x − Aᵀ(AAᵀ)⁻¹(Ax − b)
                            let aat = &ae * ae.transpose();
                            match aat.lu().solve(&r) {
                                Some(y) => xw -= ae.transpose() * y,
                                None => break 'start cold_start(n, &ae, &be, &ai, &bi, x0, config),
                            }
                        }
                    }
                    // The previous optimum sits ON its active rows, so a
                    // perturbed problem leaves it slightly infeasible —
                    // repair instead of rejecting (the pragmatic stand-in
                    // for qpOASES's homotopy).
                    if push_into_iq_feasible(&mut xw, &ae, &ai, &bi, config) {
                        // Keep only the previously-active rows that are
                        // still tight at the (possibly projected) point.
                        seed = w
                            .working_set
                            .iter()
                            .copied()
                            .filter(|&i| {
                                (row_dot(&ai, i, &xw) - bi[i]).abs()
                                    <= config.feasibility_tol.max(1e-12)
                            })
                            .collect();
                        warm = true;
                        break 'start xw;
                    }
                }
            }
        }
        cold_start(n, &ae, &be, &ai, &bi, x0, config)
    };

    // Verify feasibility (guards both start paths).
    if m_eq > 0 {
        let residual = (&ae * &x - &be).norm();
        if residual > config.feasibility_tol * (1.0 + be.norm().max(1.0)) {
            return fail(n, m_eq, m_iq, QpStatus::Infeasible);
        }
    }
    for i in 0..m_iq {
        if row_dot(&ai, i, &x) > bi[i] + config.feasibility_tol {
            return fail(n, m_eq, m_iq, QpStatus::Infeasible);
        }
    }

    // ── Working set: warm seed, or every inequality active at x ──────
    if !warm {
        for i in 0..m_iq {
            if row_dot(&ai, i, &x) >= bi[i] - config.feasibility_tol {
                seed.push(i);
            }
        }
    }

    // ── Incremental active factor: equalities, then the seed ─────────
    let mut fac = ActiveFactor::with_capacity(m_eq + m_iq.min(n) + 4);
    for i in 0..m_eq {
        let row: DVector<f64> = ae.row(i).transpose().into_owned();
        if !fac.try_push(row, &chol) {
            // Linearly dependent equalities (singular Schur complement).
            return make_sol(
                x,
                h,
                c,
                DVector::zeros(m_eq),
                DVector::zeros(m_iq),
                QpStatus::NumericalFailure,
                0,
            );
        }
    }
    let mut ws_idx: Vec<usize> = Vec::new();
    let mut in_ws = vec![false; m_iq];
    for i in seed {
        if in_ws[i] {
            continue;
        }
        let row: DVector<f64> = ai.row(i).transpose().into_owned();
        if fac.try_push(row, &chol) {
            ws_idx.push(i);
            in_ws[i] = true;
        } // linearly dependent seed rows are simply skipped
    }

    // ── Active-set iterations ────────────────────────────────────────
    let mut lam_eq = DVector::zeros(m_eq);
    let mut lam_iq = DVector::zeros(m_iq);
    // After this many iterations assume degeneracy and switch the
    // leaving-constraint choice to Bland's rule (lowest index), which
    // cannot cycle.
    let bland_after = 2 * (m_iq + n) + 10;

    for iter in 0..config.max_iters {
        let grad = h_it * &x + c;
        let m_w = ws_idx.len();

        if fac.len() == 0 {
            // Unconstrained step
            let p = chol.solve(&(-&grad));
            if p.norm() < config.optimality_tol {
                if ridge > 0.0 {
                    polish(
                        &mut x, &mut lam_eq, &mut lam_iq, h, c, &ae, &be, &ai, &bi, &ws_idx,
                        config,
                    );
                }
                stash(workspace.as_deref_mut(), &x, &ws_idx);
                return optimal(x, h, c, lam_eq, lam_iq, iter);
            }
            let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &in_ws, config);
            x += alpha * &p;
            if let Some(idx) = blocking {
                let row: DVector<f64> = ai.row(idx).transpose().into_owned();
                if fac.try_push(row, &chol) {
                    ws_idx.push(idx);
                    in_ws[idx] = true;
                }
            }
        } else {
            // Equality-constrained subproblem through the live factor:
            //   min ½pᵀHp + gᵀp  s.t.  Â p = 0
            let h_inv_r = chol.solve(&(-&grad));
            let rhs = fac.dot_rows(&h_inv_r);
            let nu = fac.solve_schur(&rhs);
            let p = fac.project(&h_inv_r, &nu);

            if p.norm() < config.optimality_tol {
                // ── Choose a leaving constraint (if any) ──────────
                let mut leave: Option<usize> = None; // position in ws_idx
                if iter >= bland_after {
                    // Bland: lowest constraint index with μ < 0.
                    let mut best: Option<(usize, usize)> = None;
                    for k in 0..m_w {
                        let mu = nu[m_eq + k];
                        if mu < -config.optimality_tol {
                            let ci = ws_idx[k];
                            if best.is_none_or(|(bc, _)| ci < bc) {
                                best = Some((ci, k));
                            }
                        }
                    }
                    leave = best.map(|(_, k)| k);
                } else {
                    // Standard: most negative multiplier.
                    let mut worst = 0.0;
                    for k in 0..m_w {
                        let mu = nu[m_eq + k];
                        if mu < -config.optimality_tol && mu < worst {
                            worst = mu;
                            leave = Some(k);
                        }
                    }
                }

                match leave {
                    None => {
                        for i in 0..m_eq {
                            lam_eq[i] = nu[i];
                        }
                        for (k, &wi) in ws_idx.iter().enumerate() {
                            lam_iq[wi] = nu[m_eq + k];
                        }
                        if ridge > 0.0 {
                            polish(
                                &mut x, &mut lam_eq, &mut lam_iq, h, c, &ae, &be, &ai, &bi,
                                &ws_idx, config,
                            );
                        }
                        stash(workspace.as_deref_mut(), &x, &ws_idx);
                        return optimal(x, h, c, lam_eq, lam_iq, iter);
                    }
                    Some(k) => {
                        fac.remove(m_eq + k);
                        in_ws[ws_idx[k]] = false;
                        ws_idx.remove(k);
                    }
                }
            } else {
                let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &in_ws, config);
                x += alpha * &p;
                if let Some(idx) = blocking {
                    let row: DVector<f64> = ai.row(idx).transpose().into_owned();
                    if fac.try_push(row, &chol) {
                        ws_idx.push(idx);
                        in_ws[idx] = true;
                    } else if let Some(&last) = ws_idx.last() {
                        // Blocking row linearly dependent on the active
                        // set: relax the most recent working constraint
                        // and retry (mirrors the previous
                        // implementation's singular-Schur handling).
                        fac.remove(m_eq + ws_idx.len() - 1);
                        in_ws[last] = false;
                        ws_idx.pop();
                    } else {
                        stash(workspace.as_deref_mut(), &x, &ws_idx);
                        return make_sol(
                            x,
                            h,
                            c,
                            lam_eq,
                            lam_iq,
                            QpStatus::NumericalFailure,
                            iter,
                        );
                    }
                }
            }
        }
    }

    stash(workspace.as_deref_mut(), &x, &ws_idx);
    make_sol(x, h, c, lam_eq, lam_iq, QpStatus::MaxIterations, config.max_iters)
}

/// Cold-start point: the caller's `x0`, else Phase-1.
fn cold_start(
    n: usize,
    ae: &DMatrix<f64>,
    be: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
) -> DVector<f64> {
    match x0 {
        Some(v) => {
            assert_eq!(v.nrows(), n, "x0 length must match H dimension");
            v.clone()
        }
        None => initial_feasible(n, ae, be, ai, bi, config),
    }
}

/// Store the exit state into the caller's workspace (if any).
fn stash(w: Option<&mut QpWorkspace>, x: &DVector<f64>, ws: &[usize]) {
    if let Some(w) = w {
        w.x = Some(x.clone());
        w.working_set = ws.to_vec();
    }
}

/// KKT polish (qpOASES's refinement idea): the ridged iterations found
/// the active set; re-solve the equality-constrained KKT system on that
/// active set with the ORIGINAL Hessian to recover the unregularised
/// optimum. Applied only if the polished point stays primal/dual
/// feasible — otherwise the ridged solution is kept (it solves the
/// ridged problem exactly and the original one to O(ridge)).
#[allow(clippy::too_many_arguments)]
fn polish(
    x: &mut DVector<f64>,
    lam_eq: &mut DVector<f64>,
    lam_iq: &mut DVector<f64>,
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    ae: &DMatrix<f64>,
    be: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    ws_idx: &[usize],
    config: &QpConfig,
) {
    let n = h.nrows();
    let m_eq = ae.nrows();
    let m = m_eq + ws_idx.len();

    // KKT:  [H  Âᵀ][x]   [−c]
    //       [Â  0 ][ν] = [ b̂]
    let mut kkt = DMatrix::zeros(n + m, n + m);
    kkt.view_mut((0, 0), (n, n)).copy_from(h);
    let mut bhat = DVector::zeros(m);
    for i in 0..m_eq {
        for j in 0..n {
            kkt[(n + i, j)] = ae[(i, j)];
            kkt[(j, n + i)] = ae[(i, j)];
        }
        bhat[i] = be[i];
    }
    for (k, &wi) in ws_idx.iter().enumerate() {
        for j in 0..n {
            kkt[(n + m_eq + k, j)] = ai[(wi, j)];
            kkt[(j, n + m_eq + k)] = ai[(wi, j)];
        }
        bhat[m_eq + k] = bi[wi];
    }
    let mut rhs = DVector::zeros(n + m);
    for i in 0..n {
        rhs[i] = -c[i];
    }
    for i in 0..m {
        rhs[n + i] = bhat[i];
    }

    let Some(sol) = kkt.lu().solve(&rhs) else { return };
    let xp = sol.rows(0, n).into_owned();

    // Primal feasibility of the inactive inequalities…
    for i in 0..ai.nrows() {
        if !ws_idx.contains(&i) && row_dot(ai, i, &xp) > bi[i] + 10.0 * config.feasibility_tol {
            return;
        }
    }
    // …and dual feasibility of the active ones.
    for k in 0..ws_idx.len() {
        if sol[n + m_eq + k] < -10.0 * config.optimality_tol {
            return;
        }
    }

    *x = xp;
    for i in 0..m_eq {
        lam_eq[i] = sol[n + i];
    }
    for (k, &wi) in ws_idx.iter().enumerate() {
        lam_iq[wi] = sol[n + m_eq + k];
    }
}

/// Incrementally-factorised active-constraint block: the active rows Â
/// (equalities first, then the working set), their cached `H⁻¹·âᵢᵀ`
/// columns, and a Cholesky factor of the Schur complement
/// `S = Â·H⁻¹·Âᵀ`, maintained by O(m²) append / delete updates instead
/// of an O(m³) refactorisation per active-set change — the qpOASES
/// factor-update idea that makes each iteration cheap.
struct ActiveFactor {
    rows: Vec<DVector<f64>>,
    y: Vec<DVector<f64>>,
    /// Lower-triangular factor of S; the live block is `m × m`.
    l: DMatrix<f64>,
    m: usize,
}

impl ActiveFactor {
    fn with_capacity(cap: usize) -> Self {
        ActiveFactor {
            rows: Vec::with_capacity(cap),
            y: Vec::with_capacity(cap),
            l: DMatrix::zeros(cap.max(1), cap.max(1)),
            m: 0,
        }
    }

    fn len(&self) -> usize {
        self.m
    }

    fn ensure_capacity(&mut self) {
        if self.m == self.l.nrows() {
            let cap = self.l.nrows() * 2;
            let mut nl = DMatrix::zeros(cap, cap);
            nl.view_mut((0, 0), (self.m, self.m))
                .copy_from(&self.l.view((0, 0), (self.m, self.m)));
            self.l = nl;
        }
    }

    /// Append one constraint row. Returns `false` (and leaves the factor
    /// untouched) if the row is linearly dependent on the active rows —
    /// the case that made the previous implementation's Schur LU
    /// singular.
    fn try_push(&mut self, row: DVector<f64>, chol: &Cholesky<f64, Dyn>) -> bool {
        self.ensure_capacity();
        let y = chol.solve(&row);
        let m = self.m;
        // New Schur column sⱼ = âⱼ·y, forward-substituted through L.
        let mut w = DVector::zeros(m);
        for j in 0..m {
            let mut acc = self.rows[j].dot(&y);
            for t in 0..j {
                acc -= self.l[(j, t)] * w[t];
            }
            w[j] = acc / self.l[(j, j)];
        }
        let d = row.dot(&y);
        let d2 = d - w.norm_squared();
        if d2 <= 1e-12 * d.abs().max(1.0) {
            return false;
        }
        for j in 0..m {
            self.l[(m, j)] = w[j];
        }
        self.l[(m, m)] = d2.sqrt();
        self.rows.push(row);
        self.y.push(y);
        self.m += 1;
        true
    }

    /// Remove the active row at position `k`: shift the trailing factor
    /// block up-left and repair it with a rank-one update by the deleted
    /// column (classic `cholupdate`), O((m−k)²).
    fn remove(&mut self, k: usize) {
        let m = self.m;
        debug_assert!(k < m);
        let t = m - k - 1;
        let mut v = DVector::zeros(t);
        for i in 0..t {
            v[i] = self.l[(k + 1 + i, k)];
        }
        for i in k..(m - 1) {
            for j in 0..=i {
                let jj = if j < k { j } else { j + 1 };
                self.l[(i, j)] = self.l[(i + 1, jj)];
            }
        }
        self.m = m - 1;
        // Trailing block: L'·L'ᵀ = L·Lᵀ + v·vᵀ.
        for j in 0..t {
            let jj = k + j;
            let ljj = self.l[(jj, jj)];
            let r = ljj.hypot(v[j]);
            let cth = r / ljj;
            let sth = v[j] / ljj;
            self.l[(jj, jj)] = r;
            for i in (j + 1)..t {
                let ii = k + i;
                let lij = self.l[(ii, jj)];
                self.l[(ii, jj)] = (lij + sth * v[i]) / cth;
                v[i] = cth * v[i] - sth * self.l[(ii, jj)];
            }
        }
        self.rows.remove(k);
        self.y.remove(k);
    }

    /// `Â·z`.
    fn dot_rows(&self, z: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(self.m, |j, _| self.rows[j].dot(z))
    }

    /// Solve `S·ν = rhs` through the maintained factor (two triangular
    /// substitutions).
    fn solve_schur(&self, rhs: &DVector<f64>) -> DVector<f64> {
        let m = self.m;
        let mut w = DVector::zeros(m);
        for i in 0..m {
            let mut acc = rhs[i];
            for j in 0..i {
                acc -= self.l[(i, j)] * w[j];
            }
            w[i] = acc / self.l[(i, i)];
        }
        let mut nu = DVector::zeros(m);
        for i in (0..m).rev() {
            let mut acc = w[i];
            for j in (i + 1)..m {
                acc -= self.l[(j, i)] * nu[j];
            }
            nu[i] = acc / self.l[(i, i)];
        }
        nu
    }

    /// `p = h_inv_r − (H⁻¹Âᵀ)·ν` from the cached columns.
    fn project(&self, h_inv_r: &DVector<f64>, nu: &DVector<f64>) -> DVector<f64> {
        let mut p = h_inv_r.clone();
        for j in 0..self.m {
            p.axpy(-nu[j], &self.y[j], 1.0);
        }
        p
    }
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
    let mut x = x0;
    push_into_iq_feasible(&mut x, ae, ai, bi, config);
    x
}

/// Reduce inequality violations of `x` by moving along the equality
/// null space (violated rows are projected out one at a time, worst
/// first). Returns whether `x` ended inside the feasible set. Used by
/// the cold Phase-1 and by the warm-start path, whose previous-tick
/// optimum sits ON its active constraints and therefore drifts slightly
/// outside after any perturbation.
fn push_into_iq_feasible(
    x: &mut DVector<f64>,
    ae: &DMatrix<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    config: &QpConfig,
) -> bool {
    let n = x.len();
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();
    if m_iq == 0 {
        return true;
    }

    // Null-space projector: P = I − Aᵀ (A Aᵀ)⁻¹ A  (identity when no eq).
    let proj_null = if m_eq > 0 {
        let aat = ae * ae.transpose();
        match aat.lu().solve(&DMatrix::identity(m_eq, m_eq)) {
            Some(aat_inv) => DMatrix::identity(n, n) - ae.transpose() * &aat_inv * ae,
            None => return false,
        }
    } else {
        DMatrix::identity(n, n)
    };

    for _ in 0..200 {
        let vals = ai * &*x;
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
            return true;
        }

        // Move x along the null-space projection of a_worst.
        let ai_col: DVector<f64> = ai.row(worst).transpose().into_owned();
        let p_ai = &proj_null * &ai_col;
        let denom = ai_col.dot(&p_ai);
        if denom < 1e-15 {
            return false; // cannot move in null space
        }
        let alpha = max_viol / denom;
        *x -= alpha * p_ai;
    }
    false
}

fn step_length(
    x: &DVector<f64>,
    p: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    in_ws: &[bool],
    config: &QpConfig,
) -> (f64, Option<usize>) {
    let m_iq = ai.nrows();
    if m_iq == 0 {
        return (1.0, None);
    }
    let mut alpha = 1.0;
    let mut blocking = None;

    for i in 0..m_iq {
        if in_ws[i] {
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

// ─── ADMM (operator-splitting) backend ──────────────────────────────────────

/// Built-in dense ADMM QP solver — a from-scratch implementation of
/// the OSQP algorithm (Stellato, Banjac, Goulart, Bemporad & Boyd,
/// *OSQP: An Operator Splitting Solver for Quadratic Programs*, 2020).
///
/// Standard form: `min ½xᵀHx + cᵀx  s.t.  l ≤ A·x ≤ u`, with
/// `A = [A_eq; A_iq]`, `l = [b_eq; −∞]`, `u = [b_eq; b_iq]` (equalities
/// as a zero-width box, inequalities as a one-sided box). Splitting
/// introduces an auxiliary `z = A·x` and alternates:
///
/// ```text
///   x̃ ← argmin  ½xᵀHx + cᵀx + (σ/2)‖x−xᵏ‖² + (ρ/2)‖A·x − zᵏ + yᵏ/ρ‖²
///   z̃ ← A·x̃
///   x ← α·x̃ + (1−α)·xᵏ            (over-relaxation)
///   z_r ← α·z̃ + (1−α)·zᵏ
///   z ← Π_{[l,u]}(z_r + yᵏ/ρ)      (box projection, closed form)
///   y ← yᵏ + ρ·(z_r − z)
/// ```
///
/// The `x̃` step is one linear solve against
/// `M = H + σ·I + ρ·Aᵀ·A`. For **fixed** `σ, ρ` this matrix never
/// changes, so it would be Cholesky-factorised **once** and every
/// iteration would be a back-substitution — no re-factorisation
/// (unlike [`solve_qp_ipm`]) or incremental update (unlike
/// [`solve_qp_active_set`]) at all. This implementation adds OSQP's
/// **adaptive ρ retuning** (§5.2 of the paper) on top of that base
/// case: `M` is re-factorised only when the primal/dual residual
/// balance drifts outside a `[0.2, 5]×` band (checked every
/// [`RHO_CHECK_EVERY`] iterations, since a factorisation is the
/// expensive step), so most iterations are still cheap
/// back-substitutions against a fixed `M` — the retuning is
/// deliberately infrequent, not a return to per-iteration
/// re-factorisation. This is the standard OSQP "reduced KKT"
/// derivation: eliminating the dual variable of the linear system's
/// second row from `[H+σI Aᵀ; A −ρ⁻¹I]·[x̃;ν] = [σxᵏ−c; zᵏ−yᵏ/ρ]` gives
/// `M·x̃ = σxᵏ − c + Aᵀ(ρzᵏ − yᵏ)`.
///
/// `σ = 1e-6` (a light Tikhonov term the OSQP paper uses for
/// numerical stability / strong convexity) and `α = 1.6`
/// (over-relaxation factor from the paper's recommended default) are
/// fixed constants throughout, as in the paper — only `ρ` adapts.
/// `ρ`'s update rule is `ρ ← ρ·√[(r_prim/scale_p)/(r_dual/scale_d)]`,
/// clamped to `[1e-6, 1e6]`: grow `ρ` when the primal (feasibility)
/// residual dominates, shrink it when the dual (optimality) residual
/// dominates. This is exactly the fixed-`σ`-mixed-task-weight
/// pathology measured without retuning (see the module-level
/// benchmark notes in `ref/wbc_comparison.md` — a fixed `ρ=10` needed
/// ~2000 iterations/tick on the Go2-scale WBC problem); retuning
/// exists specifically to correct a badly-guessed initial `ρ` without
/// paying for re-factorisation on every single iteration.
///
/// **Trade-off inherent to ADMM, not this implementation**: the cheap,
/// factorisation-free iterations come at the cost of only *linear*
/// convergence (vs the interior-point method's quadratic convergence
/// near the optimum), so reaching tight tolerances can take many more
/// iterations than [`solve_qp_ipm`] or [`solve_qp_active_set`] — a
/// fixed random full-rank 10-variable QP that both of those solve in
/// single digits took ~770 ADMM iterations to reach `optimality_tol =
/// 1e-8`, exceeding [`QpConfig::default`]'s `max_iters = 500` (the
/// returned `x` was still accurate to 3e-8 — ADMM degrades gracefully,
/// it does not diverge). Callers on tight tolerances should raise
/// `max_iters` accordingly; this is the standard ADMM/IPM trade-off
/// (cheap-but-slow vs expensive-but-fast per iteration), not a defect
/// to fix.
fn solve_qp_admm(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    config: &QpConfig,
) -> QpSolution {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!(c.nrows(), n, "c length must match H dimension");

    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();
    let m = m_eq + m_iq;

    const SIGMA: f64 = 1e-6;
    const RHO_INIT: f64 = 10.0;
    const RHO_MIN: f64 = 1e-6;
    const RHO_MAX: f64 = 1e6;
    const ALPHA: f64 = 1.6;
    // How far the residual-balance ratio must move before ρ is worth
    // retuning (OSQP §5.2): retuning means re-factorising `M`, so it
    // only pays off when the imbalance is large.
    const RHO_RETUNE_BAND: (f64, f64) = (0.2, 5.0);
    // Check every few iterations, not every one — factorisation is the
    // expensive part, so amortise the residual-ratio check over a
    // handful of cheap back-substitution iterations.
    const RHO_CHECK_EVERY: usize = 3;

    if m == 0 {
        // No constraints at all: x = -H⁻¹c directly.
        let chol = match h.clone().cholesky() {
            Some(chol) => chol,
            None => return fail(n, 0, 0, QpStatus::NumericalFailure),
        };
        let x = chol.solve(&(-c));
        return optimal(x, h, c, DVector::zeros(0), DVector::zeros(0), 0);
    }

    // Stacked A, l, u (equality rows first, matching lambda_eq/lambda_iq
    // extraction from y below).
    let mut a = DMatrix::zeros(m, n);
    let mut l = DVector::zeros(m);
    let mut u = DVector::zeros(m);
    a.rows_mut(0, m_eq).copy_from(&ae);
    l.rows_mut(0, m_eq).copy_from(&be);
    u.rows_mut(0, m_eq).copy_from(&be);
    a.rows_mut(m_eq, m_iq).copy_from(&ai);
    for i in 0..m_iq {
        l[m_eq + i] = f64::NEG_INFINITY;
    }
    u.rows_mut(m_eq, m_iq).copy_from(&bi);

    let a_t = a.transpose();
    let ata = &a_t * &a; // ρ·AᵀA is rebuilt as a scalar multiple of this cached product.
    let factorise = |rho: f64| -> Option<Cholesky<f64, Dyn>> {
        (h + DMatrix::identity(n, n) * SIGMA + rho * &ata).cholesky()
    };

    let mut rho = RHO_INIT;
    let mut chol = match factorise(rho) {
        Some(chol) => chol,
        None => return fail(n, m_eq, m_iq, QpStatus::NumericalFailure),
    };

    let mut x = DVector::<f64>::zeros(n);
    let mut z = DVector::<f64>::zeros(m);
    let mut y = DVector::<f64>::zeros(m);

    let mut iters = 0;
    for iter in 0..config.max_iters {
        iters = iter + 1;

        let rhs = &(SIGMA * &x - c) + &a_t * &(rho * &z - &y);
        let x_tilde = chol.solve(&rhs);
        let z_tilde = &a * &x_tilde;

        let x_new = ALPHA * &x_tilde + (1.0 - ALPHA) * &x;
        let z_relaxed = ALPHA * &z_tilde + (1.0 - ALPHA) * &z;
        let z_new = DVector::from_fn(m, |i, _| (z_relaxed[i] + y[i] / rho).clamp(l[i], u[i]));
        let y_new = &y + rho * (&z_relaxed - &z_new);

        // ── Convergence check (OSQP's primal/dual residual test) ──
        let r_prim = &a * &x_new - &z_new;
        let r_dual = h * &x_new + c + &a_t * &y_new;
        let scale_p = 1.0 + (&a * &x_new).amax().max(z_new.amax());
        let scale_d = 1.0 + (h * &x_new).amax().max((&a_t * &y_new).amax()).max(c.amax());
        x = x_new;
        z = z_new;
        y = y_new;

        if r_prim.amax() < config.feasibility_tol * scale_p
            && r_dual.amax() < config.optimality_tol * scale_d
        {
            return make_sol(
                x,
                h,
                c,
                y.rows(0, m_eq).into_owned(),
                y.rows(m_eq, m_iq).into_owned(),
                QpStatus::Optimal,
                iters,
            );
        }

        // ── Adaptive ρ retuning (OSQP §5.2) ────────────────────────
        // ρ_new = ρ·√[(r_prim/scale_p) / (r_dual/scale_d)]: grow ρ
        // when the primal residual dominates (push harder toward
        // feasibility), shrink it when the dual residual dominates
        // (push harder toward optimality). Only retune periodically
        // and only outside a tolerance band around 1×, since it costs
        // a full re-factorisation of `M`.
        if (iter + 1) % RHO_CHECK_EVERY == 0 {
            let num = r_prim.amax() / scale_p;
            let den = (r_dual.amax() / scale_d).max(1e-300);
            let ratio = (num / den).sqrt();
            if !(RHO_RETUNE_BAND.0..=RHO_RETUNE_BAND.1).contains(&ratio) {
                let rho_new = (rho * ratio).clamp(RHO_MIN, RHO_MAX);
                if let Some(new_chol) = factorise(rho_new) {
                    rho = rho_new;
                    chol = new_chol;
                }
                // If re-factorisation fails (shouldn't, since H is
                // PSD-safe and ρ>0), keep the current ρ/chol and carry
                // on rather than aborting a solve that was progressing.
            }
        }
    }

    make_sol(
        x,
        h,
        c,
        y.rows(0, m_eq).into_owned(),
        y.rows(m_eq, m_iq).into_owned(),
        QpStatus::MaxIterations,
        iters,
    )
}

// ─── Interior-point backend (Mehrotra predictor-corrector) ──────────────────

/// Built-in dense primal-dual interior-point QP solver.
///
/// A from-scratch implementation of Mehrotra's predictor-corrector
/// method (Nocedal & Wright, *Numerical Optimization*, Algorithm 16.4)
/// — the textbook path-following IPM, kept deliberately simple (dense
/// linear algebra, no Mehrotra multiple-correction refinements beyond
/// the standard predictor+corrector pair) since its purpose is to make
/// the IPM approach concretely comparable to the active-set method in
/// this same module, not to match Clarabel's maturity.
///
/// Standard form: introduce a slack `s ≥ 0` for the inequalities,
/// `A_iq·x + s = b_iq`, and multipliers `y` (equalities), `z ≥ 0`
/// (inequalities). The KKT system, perturbed by a barrier parameter
/// `μ = sᵀz / m_iq`:
///
/// ```text
///   H·x + c + A_eqᵀ·y + A_iqᵀ·z = 0        (stationarity)
///   A_eq·x = b_eq                          (equality feasibility)
///   A_iq·x + s = b_iq                      (inequality feasibility)
///   S·z = μ·e                              (perturbed complementarity)
/// ```
///
/// Each iteration takes an **affine** (μ = 0, predictor) Newton step to
/// estimate how much duality gap reduction is achievable, derives
/// Mehrotra's centering parameter `σ = (μ_aff/μ)³` from it, then solves
/// once more with a **corrector** right-hand side (centering term +
/// the affine step's second-order `Δs_aff∘Δz_aff` correction). Both
/// solves reuse the same factorisation (only the RHS differs). Step
/// lengths use the standard `τ = 0.995` fraction-to-boundary rule so
/// `s, z` stay strictly positive.
fn solve_qp_ipm(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    config: &QpConfig,
    mcc: bool,
) -> QpSolution {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!(c.nrows(), n, "c length must match H dimension");

    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();

    // Fraction-to-boundary safety margin (Wächter & Biegler / standard
    // IPM practice): never step all the way to the s/z boundary.
    const TAU: f64 = 0.995;

    let mut x = DVector::<f64>::zeros(n);
    // s, z start at a fixed strictly-feasible-in-sign point (a proper
    // implementation would use Mehrotra's initialisation heuristic;
    // this keeps the algorithm's structure legible).
    let mut s = DVector::<f64>::repeat(m_iq, 1.0);
    let mut z = DVector::<f64>::repeat(m_iq, 1.0);
    let mut y = DVector::<f64>::zeros(m_eq);

    let mut iters = 0;
    for iter in 0..config.max_iters {
        iters = iter + 1;

        // ── Residuals at the current point ────────────────────────
        let r_stat = h * &x + c + ae.transpose() * &y + ai.transpose() * &z;
        let r_eq = if m_eq > 0 { &ae * &x - &be } else { DVector::zeros(0) };
        let r_iq = if m_iq > 0 { &ai * &x + &s - &bi } else { DVector::zeros(0) };
        let mu = if m_iq > 0 { s.dot(&z) / m_iq as f64 } else { 0.0 };

        // ── Convergence check ─────────────────────────────────────
        let scale = 1.0 + c.norm();
        if r_stat.norm() < config.optimality_tol * scale
            && r_eq.norm() < config.feasibility_tol * scale
            && r_iq.norm() < config.feasibility_tol * scale
            && mu < config.optimality_tol
        {
            return make_sol(x, h, c, y, z, QpStatus::Optimal, iters);
        }

        if m_iq == 0 {
            // No inequalities: one exact Newton step solves the (linear)
            // KKT system directly — no barrier, nothing to predict.
            let Some((dx, dy)) = solve_reduced_kkt(h, &ae, &(-&r_stat), &(-&r_eq)) else {
                return make_sol(x, h, c, y, z, QpStatus::NumericalFailure, iters);
            };
            x += dx;
            y += dy;
            continue;
        }

        // ── Shared machinery for both the predictor and corrector solves ──
        //
        // Eliminating (ds, dz) from the linearised complementarity
        // equation `Z·ds + S·dz = t` (t is the perturbed-complementarity
        // target — different for the affine vs corrector step) against
        // the inequality-feasibility equation `A_iq·dx + ds = -r_iq`
        // gives:
        //
        //   dz = t/s − (z/s)·ds                              (dz_from)
        //   [H + A_iqᵀ(Z/S)A_iq]·dx + A_eqᵀ·dy
        //       = −r_stat − A_iqᵀ·[(z/s)·r_iq + t/s]          (rhs_x)
        //
        // (both derived by substituting dz into the stationarity
        // equation `H·dx + A_eqᵀ·dy + A_iqᵀ·dz = −r_stat`).
        let z_over_s = DVector::from_fn(m_iq, |i, _| z[i] / s[i]);
        let h_bar = h + weighted_normal_eq(&ai, &z_over_s);
        let rhs_x = |t: &DVector<f64>| -> DVector<f64> {
            &(-&r_stat)
                - ai.transpose()
                    * DVector::from_fn(m_iq, |i, _| z_over_s[i] * r_iq[i] + t[i] / s[i])
        };
        let dz_from = |t: &DVector<f64>, ds: &DVector<f64>| -> DVector<f64> {
            DVector::from_fn(m_iq, |i, _| t[i] / s[i] - z_over_s[i] * ds[i])
        };

        // ── Predictor (affine-scaling): drive complementarity to 0 ────
        let t_aff = DVector::from_fn(m_iq, |i, _| -(s[i] * z[i]));
        let Some((dx_aff, _dy_aff)) = solve_reduced_kkt(&h_bar, &ae, &rhs_x(&t_aff), &(-&r_eq))
        else {
            return make_sol(x, h, c, y, z, QpStatus::NumericalFailure, iters);
        };
        let ds_aff = -&r_iq - &ai * &dx_aff;
        let dz_aff = dz_from(&t_aff, &ds_aff);

        let alpha_aff_p = fraction_to_boundary(&s, &ds_aff, 1.0);
        let alpha_aff_d = fraction_to_boundary(&z, &dz_aff, 1.0);
        let s_aff = &s + alpha_aff_p * &ds_aff;
        let z_aff = &z + alpha_aff_d * &dz_aff;
        let mu_aff = s_aff.dot(&z_aff) / m_iq as f64;

        // Mehrotra's centering parameter.
        let sigma = if mu > 1e-14 { (mu_aff / mu).powi(3).clamp(0.0, 1.0) } else { 0.0 };
        let sigma_mu = sigma * mu;

        // ── Corrector: centered target + the affine step's 2nd-order term ──
        let t_cor = DVector::from_fn(m_iq, |i, _| {
            -(s[i] * z[i]) + sigma_mu - ds_aff[i] * dz_aff[i]
        });
        let Some((dx, dy)) = solve_reduced_kkt(&h_bar, &ae, &rhs_x(&t_cor), &(-&r_eq)) else {
            return make_sol(x, h, c, y, z, QpStatus::NumericalFailure, iters);
        };
        let mut dp_x = dx;
        let mut dp_y = dy;
        let mut dp_s = -&r_iq - &ai * &dp_x;
        let mut dp_z = dz_from(&t_cor, &dp_s);

        // ── Gondzio's multiple centrality correctors ───────────────
        //
        // The predictor+Mehrotra-corrector direction Δp above is a good
        // guess, but it is optimistic (it targets complementarity = μ
        // everywhere) and its step length is limited by whichever
        // component of (s, z) hits the boundary first. Gondzio's
        // technique (Colombo & Gondzio 2008, eq. 7) reuses the SAME
        // factorised `h_bar` — no new factorisation, just extra cheap
        // back-substitutions — to compute extra correction directions
        // that push only the OUTLIER complementarity products (those
        // that would end up far from μ at a trial step) back toward a
        // reasonable range, which enlarges the step Δp can take.
        const MAX_CORRECTORS: usize = 3;
        const ASPIRATION: f64 = 0.1; // δ: how much extra step length we chase per round
        const GAMMA: f64 = 0.1; // γ: outlier band (γμ, γ⁻¹μ) that needs no correction
        const LINE_SEARCH_POINTS: usize = 8; // ω grid resolution in [0, 1]

        for _ in 0..(if mcc { MAX_CORRECTORS } else { 0 }) {
            let alpha_p = fraction_to_boundary(&s, &dp_s, 1.0);
            let alpha_d = fraction_to_boundary(&z, &dp_z, 1.0);
            if alpha_p >= 1.0 - 1e-9 && alpha_d >= 1.0 - 1e-9 {
                break; // already reaches the boundary; no aspiration room left
            }

            // Trial point at the aspired (enlarged) step length.
            let target_p = (alpha_p + ASPIRATION).min(1.0);
            let target_d = (alpha_d + ASPIRATION).min(1.0);
            let s_trial = &s + target_p * &dp_s;
            let z_trial = &z + target_d * &dp_z;

            // Target only the outliers toward the (γμ, γ⁻¹μ) band —
            // components already inside it are left alone (eq. 7).
            let t_mcc = DVector::from_fn(m_iq, |i, _| {
                let v = s_trial[i] * z_trial[i];
                if v <= GAMMA * mu {
                    GAMMA * mu - v
                } else if v >= mu / GAMMA {
                    mu / GAMMA - v
                } else {
                    0.0
                }
            });

            // Pure centrality corrector: stationarity/eq-feasibility
            // residuals are 0 here (only the complementarity target is
            // non-zero), reusing the SAME h_bar factorisation.
            let zero_eq = DVector::zeros(m_eq);
            let rhs_m = -ai.transpose() * DVector::from_fn(m_iq, |i, _| t_mcc[i] / s[i]);
            let Some((dm_x, dm_y)) = solve_reduced_kkt(&h_bar, &ae, &rhs_m, &zero_eq) else {
                break; // keep Δp as-is rather than aborting an otherwise-good step
            };
            let dm_s = -&ai * &dm_x;
            let dm_z = dz_from(&t_mcc, &dm_s);

            // Line-search ω ∈ [0, 1] for Δ = Δp + ω·Δm maximising the
            // achievable step-length product α_P·α_D.
            let base_product = alpha_p * alpha_d;
            let mut best = (0.0_f64, base_product);
            for k in 1..=LINE_SEARCH_POINTS {
                let omega = k as f64 / LINE_SEARCH_POINTS as f64;
                let cs = &dp_s + omega * &dm_s;
                let cz = &dp_z + omega * &dm_z;
                let ap = fraction_to_boundary(&s, &cs, 1.0);
                let ad = fraction_to_boundary(&z, &cz, 1.0);
                if ap * ad > best.1 {
                    best = (omega, ap * ad);
                }
            }
            if best.0 <= 0.0 {
                break; // no ω improved the step length; further correctors won't help
            }

            dp_x += best.0 * dm_x;
            dp_y += best.0 * dm_y;
            dp_s += best.0 * dm_s;
            dp_z += best.0 * dm_z;
        }

        let alpha_p = fraction_to_boundary(&s, &dp_s, TAU);
        let alpha_d = fraction_to_boundary(&z, &dp_z, TAU);

        x += alpha_p * &dp_x;
        s += alpha_p * &dp_s;
        y += alpha_d * &dp_y;
        z += alpha_d * &dp_z;
    }

    make_sol(x, h, c, y, z, QpStatus::MaxIterations, iters)
}

/// `A_iqᵀ · diag(w) · A_iq`, the rank-`m_iq` term the IPM adds to `H`
/// each iteration (the inequality "barrier Hessian"). Dense — this
/// solver targets the same small/medium problems as [`ActiveSet`].
fn weighted_normal_eq(ai: &DMatrix<f64>, w: &DVector<f64>) -> DMatrix<f64> {
    let scaled = DMatrix::from_fn(ai.nrows(), ai.ncols(), |r, c| ai[(r, c)] * w[r]);
    ai.transpose() * scaled
}

/// Solve the reduced (equality-only) KKT system
/// `[H Aᵀ; A 0]·[dx;dy] = [rx;ry]` via one dense Cholesky of `H` and a
/// Schur complement on the (typically small) equality block — the same
/// two-stage solve the active-set backend uses per iteration, just
/// without incremental factor updates (the IPM's `H` changes every
/// iteration via the barrier term, so there is nothing to reuse across
/// iterations here).
fn solve_reduced_kkt(
    h: &DMatrix<f64>,
    ae: &DMatrix<f64>,
    rx: &DVector<f64>,
    ry: &DVector<f64>,
) -> Option<(DVector<f64>, DVector<f64>)> {
    let n = h.nrows();
    let m_eq = ae.nrows();
    let chol = match h.clone().cholesky() {
        Some(c) => c,
        None => (h + DMatrix::identity(n, n) * 1e-10).cholesky()?,
    };
    if m_eq == 0 {
        return Some((chol.solve(rx), DVector::zeros(0)));
    }
    let h_inv_at = chol.solve(&ae.transpose());
    let s = ae * &h_inv_at; // m_eq × m_eq Schur complement
    let h_inv_rx = chol.solve(rx);
    let rhs = ae * &h_inv_rx - ry;
    let dy = s.lu().solve(&rhs)?;
    let dx = h_inv_rx - &h_inv_at * &dy;
    Some((dx, dy))
}

/// The fraction-to-boundary step length: the largest `α ∈ (0, α_max]`
/// such that `v + α·dv ≥ (1 − τ)·v` component-wise (i.e. stays strictly
/// positive with a `τ` safety margin). `α_max` is `1.0` for the final
/// corrector step and also `1.0` for the affine predictor step (the
/// predictor is allowed to reach the boundary — only used to measure
/// achievable centering, never applied to `x`/`y` directly).
fn fraction_to_boundary(v: &DVector<f64>, dv: &DVector<f64>, tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for i in 0..v.len() {
        if dv[i] < 0.0 {
            alpha = alpha.min(-tau * v[i] / dv[i]);
        }
    }
    alpha.clamp(0.0, 1.0)
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
    // ─── Warm-start workspace (qpOASES-style online active set) ─────

    /// A tiny deterministic LCG for the randomised warm-start tests.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// A well-conditioned random QP with equalities and inequalities.
    fn random_qp(
        rng: &mut Lcg,
        n: usize,
        m_eq: usize,
        m_iq: usize,
    ) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
        let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
        let h = &a * a.transpose() + DMatrix::identity(n, n);
        let c = DVector::from_fn(n, |_, _| rng.next_f64());
        let ae = DMatrix::from_fn(m_eq, n, |_, _| rng.next_f64());
        let be = DVector::from_fn(m_eq, |_, _| rng.next_f64() * 0.3);
        let ai = DMatrix::from_fn(m_iq, n, |_, _| rng.next_f64());
        // Loose enough that a feasible region exists, tight enough that
        // several rows go active at the optimum.
        let bi = DVector::from_fn(m_iq, |_, _| rng.next_f64().abs() * 0.5 + 0.05);
        (h, c, ae, be, ai, bi)
    }

    /// KKT residuals of a returned solution — backend-independent
    /// optimality certificate (stationarity, primal feasibility, dual
    /// feasibility, complementary slackness).
    fn assert_kkt(
        h: &DMatrix<f64>,
        c: &DVector<f64>,
        ae: &DMatrix<f64>,
        be: &DVector<f64>,
        ai: &DMatrix<f64>,
        bi: &DVector<f64>,
        sol: &QpSolution,
    ) {
        assert_eq!(sol.status, QpStatus::Optimal, "not optimal");
        let x = &sol.x;
        let stat = h * x + c + ae.transpose() * &sol.lambda_eq + ai.transpose() * &sol.lambda_iq;
        assert!(stat.norm() < 1e-5, "stationarity violated: {}", stat.norm());
        if ae.nrows() > 0 {
            assert!((ae * x - be).norm() < 1e-6, "eq violated");
        }
        for i in 0..ai.nrows() {
            let slack = bi[i] - (ai.row(i) * x)[0];
            assert!(slack > -1e-6, "iq {i} violated: slack {slack}");
            assert!(sol.lambda_iq[i] > -1e-6, "dual feasibility violated at {i}");
            assert!(
                sol.lambda_iq[i].abs() * slack.abs() < 1e-4,
                "complementary slackness violated at {i}"
            );
        }
    }

    #[test]
    fn active_set_satisfies_kkt_on_random_problems() {
        // Hammers the incremental add / remove factor paths.
        let mut rng = Lcg(0xFEED);
        let cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        for _ in 0..25 {
            let (h, c, ae, be, ai, bi) = random_qp(&mut rng, 10, 3, 12);
            let sol = solve_qp(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg);
            assert_kkt(&h, &c, &ae, &be, &ai, &bi, &sol);
        }
    }

    #[test]
    fn warm_resolve_of_identical_qp_is_one_iteration() {
        let mut rng = Lcg(0xBEEF);
        let (h, c, ae, be, ai, bi) = random_qp(&mut rng, 12, 3, 14);
        let cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        let mut ws = QpWorkspace::new();

        let cold =
            solve_qp_warm(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg, &mut ws);
        assert_eq!(cold.status, QpStatus::Optimal);
        assert!(cold.iterations > 0, "cold solve should iterate");

        let warm =
            solve_qp_warm(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg, &mut ws);
        assert_eq!(warm.status, QpStatus::Optimal);
        assert!(
            warm.iterations <= 1,
            "warm re-solve took {} iterations (cold took {})",
            warm.iterations,
            cold.iterations
        );
        assert!((&warm.x - &cold.x).norm() < 1e-8, "warm x drifted");
    }

    #[test]
    fn warm_workspace_tracks_a_perturbed_tick_sequence() {
        // A drifting QP sequence (b_eq and c move every tick) — warm
        // restarts must stay correct (KKT) and cheaper than cold.
        let mut rng = Lcg(0xCAFE);
        let (h, c0, ae, be0, ai, bi) = random_qp(&mut rng, 12, 3, 14);
        let cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        let mut ws = QpWorkspace::new();
        let mut warm_iters = 0usize;
        let mut cold_iters = 0usize;

        for t in 0..20 {
            let phase = t as f64 * 0.15;
            let c = &c0 + DVector::from_fn(c0.len(), |i, _| 0.01 * (phase + i as f64).sin());
            let be = &be0 + DVector::from_fn(be0.len(), |i, _| 0.005 * (phase * 1.3 + i as f64).cos());

            let warm = solve_qp_warm(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg, &mut ws);
            assert_kkt(&h, &c, &ae, &be, &ai, &bi, &warm);
            warm_iters += warm.iterations;

            let cold = solve_qp(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg);
            cold_iters += cold.iterations;
            assert!((&warm.x - &cold.x).norm() < 1e-6, "tick {t}: warm and cold disagree");
        }
        assert!(
            warm_iters < cold_iters,
            "warm restarts should beat cold starts: {warm_iters} vs {cold_iters}"
        );
    }

    /// The HoQp-shaped failure mode: H = AᵀA + tiny·I with A rank-
    /// deficient (κ ≈ 1e8) used to crawl into MaxIterations through
    /// microscopic clipped steps. The conditional ridge + KKT polish
    /// must solve it in a handful of iterations AND return the
    /// unregularised optimum (verified against the KKT conditions of
    /// the ORIGINAL problem).
    #[test]
    fn ill_conditioned_hessian_is_ridged_and_polished() {
        let n = 42;
        for rows in [6usize, 15] {
            let a = DMatrix::from_fn(rows, n, |i, j| ((i * 13 + j * 7) as f64 * 0.37).sin());
            let h = a.transpose() * &a + DMatrix::identity(n, n) * 1e-8;
            let b = DVector::from_fn(rows, |i, _| ((i as f64) * 0.9).cos());
            let c = -(a.transpose() * &b);
            let d = DMatrix::from_fn(44, n, |i, j| ((i * 5 + j * 3) as f64 * 0.23).cos());
            let f = DVector::from_fn(44, |i, _| 0.5 + 0.1 * ((i as f64) * 0.7).sin().abs());
            let cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };

            let sol = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &cfg);
            assert_eq!(sol.status, QpStatus::Optimal, "rows={rows}: crawled");
            assert!(sol.iterations < 50, "rows={rows}: {} iterations", sol.iterations);
            // KKT of the ORIGINAL (unridged) problem.
            let stat = &h * &sol.x + &c + d.transpose() * &sol.lambda_iq;
            assert!(stat.norm() < 1e-5, "rows={rows}: stationarity {}", stat.norm());
            for i in 0..44 {
                let slack = f[i] - (d.row(i) * &sol.x)[0];
                assert!(slack > -1e-6, "rows={rows}: iq {i} violated");
                assert!(sol.lambda_iq[i] > -1e-6, "rows={rows}: dual infeasible at {i}");
            }
        }
    }

    // ─── Interior-point (Mehrotra predictor-corrector) ───────────────

    #[test]
    fn ipm_satisfies_kkt_on_random_problems() {
        let mut rng = Lcg(0xC0FFEE);
        let cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
        for _ in 0..25 {
            let (h, c, ae, be, ai, bi) = random_qp(&mut rng, 10, 3, 12);
            let sol = solve_qp(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg);
            assert_kkt(&h, &c, &ae, &be, &ai, &bi, &sol);
        }
    }

    #[test]
    fn ipm_agrees_with_active_set_on_full_rank_problems() {
        // Full rank ⇒ unique optimum, so the two independently-derived
        // methods (barrier path-following vs active-set) must land on
        // the same x, not just the same objective.
        let mut rng = Lcg(0xFACADE);
        let ipm_cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
        let as_cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        for _ in 0..15 {
            let n = 10;
            let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
            let h = a.transpose() * &a + DMatrix::identity(n, n) * 0.5;
            let c = DVector::from_fn(n, |_, _| rng.next_f64());
            let d = DMatrix::from_fn(2 * n, n, |_, _| rng.next_f64());
            let f = DVector::from_fn(2 * n, |_, _| rng.next_f64().abs() * 0.5 + 0.1);

            let ipm = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &ipm_cfg);
            let act = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &as_cfg);
            assert_eq!(ipm.status, QpStatus::Optimal);
            assert_eq!(act.status, QpStatus::Optimal);
            assert!(
                (&ipm.x - &act.x).norm() < 1e-4,
                "ipm/active-set disagree: {:?} vs {:?}",
                ipm.x,
                act.x
            );
        }
    }

    #[test]
    fn ipm_solves_equality_only_in_one_newton_step() {
        // No inequalities: the KKT system is linear, so a single exact
        // Newton step (the m_iq == 0 fast path) must already be optimal.
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
        let sol = solve_qp(&h, &c, Some(&a), Some(&b), None, None, None, &cfg);
        assert_eq!(sol.status, QpStatus::Optimal);
        // One Newton step to solve, one more loop entry to confirm
        // convergence against the (now-zero) residual.
        assert_eq!(sol.iterations, 2);
        assert!((sol.x[0] - 0.5).abs() < 1e-8 && (sol.x[1] - 0.5).abs() < 1e-8);
    }

    #[test]
    fn ipm_respects_max_iters_on_an_unreachable_tolerance() {
        // A trivially-solvable QP but zero iteration budget must report
        // MaxIterations rather than silently returning x = 0.
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-1.0, -1.0]);
        let d = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let f = DVector::from_vec(vec![1.0]);
        let cfg = QpConfig { solver: QpSolver::Ipm, max_iters: 0, ..Default::default() };
        let sol = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &cfg);
        assert_eq!(sol.status, QpStatus::MaxIterations);
    }

    // ─── IpmMcc (Gondzio's multiple centrality correctors) ───────────

    #[test]
    fn ipm_mcc_satisfies_kkt_on_random_problems() {
        let mut rng = Lcg(0xC0FFEE);
        let cfg = QpConfig { solver: QpSolver::IpmMcc, ..Default::default() };
        for _ in 0..25 {
            let (h, c, ae, be, ai, bi) = random_qp(&mut rng, 10, 3, 12);
            let sol = solve_qp(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg);
            assert_kkt(&h, &c, &ae, &be, &ai, &bi, &sol);
        }
    }

    #[test]
    fn ipm_mcc_agrees_with_plain_ipm_on_full_rank_problems() {
        // Adding centrality correctors changes the path taken, not the
        // problem: both must land on the same unique optimum.
        let mut rng = Lcg(0xFACADE);
        let mcc_cfg = QpConfig { solver: QpSolver::IpmMcc, ..Default::default() };
        let plain_cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
        for _ in 0..15 {
            let n = 10;
            let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
            let h = a.transpose() * &a + DMatrix::identity(n, n) * 0.5;
            let c = DVector::from_fn(n, |_, _| rng.next_f64());
            let d = DMatrix::from_fn(2 * n, n, |_, _| rng.next_f64());
            let f = DVector::from_fn(2 * n, |_, _| rng.next_f64().abs() * 0.5 + 0.1);

            let mcc = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &mcc_cfg);
            let plain = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &plain_cfg);
            assert_eq!(mcc.status, QpStatus::Optimal);
            assert_eq!(plain.status, QpStatus::Optimal);
            assert!(
                (&mcc.x - &plain.x).norm() < 1e-4,
                "ipm/ipm-mcc disagree: {:?} vs {:?}",
                mcc.x,
                plain.x
            );
        }
    }

    #[test]
    fn ipm_mcc_solves_equality_only_in_one_newton_step() {
        // No inequalities: MCC has nothing to correct (m_iq == 0 takes
        // the same fast path as plain Ipm).
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let cfg = QpConfig { solver: QpSolver::IpmMcc, ..Default::default() };
        let sol = solve_qp(&h, &c, Some(&a), Some(&b), None, None, None, &cfg);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_eq!(sol.iterations, 2);
    }

    // ─── ADMM (operator splitting / OSQP algorithm) ──────────────────

    #[test]
    fn admm_satisfies_kkt_on_random_problems() {
        let mut rng = Lcg(0xADD3D);
        let cfg = QpConfig { solver: QpSolver::Admm, max_iters: 5000, ..Default::default() };
        for _ in 0..25 {
            let (h, c, ae, be, ai, bi) = random_qp(&mut rng, 10, 3, 12);
            let sol = solve_qp(&h, &c, Some(&ae), Some(&be), Some(&ai), Some(&bi), None, &cfg);
            assert_kkt(&h, &c, &ae, &be, &ai, &bi, &sol);
        }
    }

    #[test]
    fn admm_agrees_with_active_set_on_full_rank_problems() {
        // Full rank ⇒ unique optimum. ADMM's linear convergence needs a
        // generous iteration budget to reach the same tight tolerance
        // active-set hits in a handful of steps — this is the expected
        // ADMM/active-set trade-off (see solve_qp_admm's docs), not a
        // bug, so the budget here is set accordingly rather than tuned
        // down to "prove" a fast answer.
        let mut rng = Lcg(0xFACADE);
        let admm_cfg = QpConfig { solver: QpSolver::Admm, max_iters: 5000, ..Default::default() };
        let as_cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        for _ in 0..15 {
            let n = 10;
            let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
            let h = a.transpose() * &a + DMatrix::identity(n, n) * 0.5;
            let c = DVector::from_fn(n, |_, _| rng.next_f64());
            let d = DMatrix::from_fn(2 * n, n, |_, _| rng.next_f64());
            let f = DVector::from_fn(2 * n, |_, _| rng.next_f64().abs() * 0.5 + 0.1);

            let admm = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &admm_cfg);
            let act = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &as_cfg);
            assert_eq!(admm.status, QpStatus::Optimal);
            assert_eq!(act.status, QpStatus::Optimal);
            assert!(
                (&admm.x - &act.x).norm() < 1e-4,
                "admm/active-set disagree: {:?} vs {:?}",
                admm.x,
                act.x
            );
        }
    }

    #[test]
    fn admm_handles_equality_only_problems() {
        // No inequalities (m_iq = 0): A = A_eq only, l = u = b_eq — the
        // box projection degenerates to the identity on those rows.
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let cfg = QpConfig { solver: QpSolver::Admm, max_iters: 2000, ..Default::default() };
        let sol = solve_qp(&h, &c, Some(&a), Some(&b), None, None, None, &cfg);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 0.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn admm_handles_unconstrained_problems() {
        // No constraints at all (m = 0): the fast path x = -H⁻¹c.
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        let cfg = QpConfig { solver: QpSolver::Admm, ..Default::default() };
        let sol = solve_qp(&h, &c, None, None, None, None, None, &cfg);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_eq!(sol.iterations, 0);
        assert!((sol.x[0] - 2.0).abs() < 1e-10 && (sol.x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn admm_adaptive_rho_converges_at_default_max_iters() {
        // The exact problem that needed ~770 fixed-ρ iterations (see
        // solve_qp_admm's docs) and used to exceed the crate default
        // max_iters=500 — with adaptive ρ retuning it reaches Optimal
        // well inside the default budget, and agrees with active-set.
        let mut rng = Lcg(0xFACADE);
        let n = 10;
        let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
        let h = a.transpose() * &a + DMatrix::identity(n, n) * 0.5;
        let c = DVector::from_fn(n, |_, _| rng.next_f64());
        let d = DMatrix::from_fn(2 * n, n, |_, _| rng.next_f64());
        let f = DVector::from_fn(2 * n, |_, _| rng.next_f64().abs() * 0.5 + 0.1);

        let cfg = QpConfig { solver: QpSolver::Admm, ..Default::default() };
        let admm = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &cfg);
        assert_eq!(admm.status, QpStatus::Optimal, "adaptive ρ should converge in-budget");

        let as_cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        let act = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &as_cfg);
        assert!(
            (&admm.x - &act.x).norm() < 1e-4,
            "admm/active-set disagree: {}",
            (&admm.x - &act.x).norm()
        );
    }

    #[test]
    fn admm_reports_max_iterations_without_diverging() {
        // A genuinely tiny iteration budget must still report
        // MaxIterations honestly (not silently return an unconverged
        // x as Optimal), while the iterate stays in the right
        // neighbourhood rather than blowing up — ADMM degrades
        // gracefully even when cut off early.
        let mut rng = Lcg(0xFACADE);
        let n = 10;
        let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
        let h = a.transpose() * &a + DMatrix::identity(n, n) * 0.5;
        let c = DVector::from_fn(n, |_, _| rng.next_f64());
        let d = DMatrix::from_fn(2 * n, n, |_, _| rng.next_f64());
        let f = DVector::from_fn(2 * n, |_, _| rng.next_f64().abs() * 0.5 + 0.1);

        let cfg = QpConfig { solver: QpSolver::Admm, max_iters: 5, ..Default::default() };
        let admm = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &cfg);
        assert_eq!(admm.status, QpStatus::MaxIterations);
        assert!(admm.x.iter().all(|v| v.is_finite()), "iterate must not blow up: {:?}", admm.x);

        let as_cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };
        let act = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &as_cfg);
        assert!(
            (&admm.x - &act.x).norm() < 5.0,
            "even a 5-iteration ADMM run should be in the right neighbourhood: {}",
            (&admm.x - &act.x).norm()
        );
    }

    #[test]
    fn admm_adaptive_rho_speeds_up_the_ill_scaled_wbc_problem() {
        // The Go2-scale, mixed-weight problem that needed ~2000 fixed-ρ
        // iterations/tick (measured in qp_warm_bench without adaptive
        // ρ) must now converge well inside the crate default max_iters
        // (500) thanks to retuning away from the badly-guessed ρ=10.
        let n = 42;
        let weights = [1.0_f64, 0.1, 1e-3];
        let mut h = DMatrix::<f64>::identity(n, n) * 1e-8;
        let mut c = DVector::zeros(n);
        for (k, &w) in weights.iter().enumerate() {
            let rows = 6 + k * 3;
            let a = DMatrix::from_fn(rows, n, |i, j| ((i * (7 + k) + j * (3 + k)) as f64 * 0.31).sin());
            let b = DVector::from_fn(rows, |i, _| ((i as f64) * 0.8).cos() * 100.0);
            h += w * a.transpose() * &a;
            c -= w * a.transpose() * &b;
        }
        let d = DMatrix::from_fn(44, n, |i, j| ((i * 5 + j * 3) as f64 * 0.23).cos());
        let f = DVector::from_fn(44, |i, _| 0.5 + 0.1 * ((i as f64) * 0.7).sin().abs());

        let cfg = QpConfig { solver: QpSolver::Admm, ..Default::default() };
        let sol = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &cfg);
        assert_eq!(sol.status, QpStatus::Optimal, "adaptive ρ should tame the ill-scaled problem");
        assert!(sol.iterations < 500, "took {} iterations", sol.iterations);
    }

}
