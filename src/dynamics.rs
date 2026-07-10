//! Formulation-switchable dynamics context: one place that answers
//! "what is `q̈`?", "what is `τ`?" as [`Affine`] expressions of the
//! decision vector, so the same task declarations work in any of the
//! three whole-body-control formulations found in the field:
//!
//! | [`Formulation`] | decision vector | eliminated | lineage |
//! |---|---|---|---|
//! | `Explicit`    | `x = [q̈; f; τ]` | —                | quadruped-gait / this crate's default |
//! | `AccelSpace`  | `x = [q̈; f]`    | `τ` via EoM      | OpenSoT |
//! | `ForceSpace`  | `x = [τ; f]`    | `q̈` via `M⁻¹`   | GID (operational-space inverse dynamics) |
//!
//! All three describe the same physics; they differ in which quantities
//! are optimisation variables and which are affine consequences. Because
//! either elimination is an **affine** map of the remaining variables,
//! a [`Dynamics`] value can hand every task builder the right
//! expression and the stack itself stays formulation-agnostic — build
//! the same stack under two formulations and compare solutions and
//! solve times directly.
//!
//! The floating-base equation of motion with point contacts,
//!
//! ```text
//!   M·q̈ + h = Sᵀ·τ + Jcᵀ·f,       S = [0 | I_na]
//! ```
//!
//! is carried per formulation as:
//!
//! - `Explicit`: all `nv` rows as an equality task
//!   ([`Dynamics::dynamics_task`]) — nothing eliminated.
//! - `AccelSpace`: `τ(x) = M_a·q̈ + h_a − (Jcᵀ)_a·f` (the actuated
//!   rows, rearranged), plus the `n_base` **underactuated rows** as the
//!   equality task. This is OpenSoT's construction.
//! - `ForceSpace`: `q̈(x) = M⁻¹(Sᵀτ + Jcᵀf − h)` — the EoM holds
//!   identically, no equality task remains. The matrix products
//!   `M⁻¹Sᵀ`, `M⁻¹Jcᵀ`, `M⁻¹h` are what GID measures matrix-free by
//!   unit-force propagation (`I = J·M⁻¹·[Sᵀ Jcᵀ]`); here they are
//!   formed explicitly via a Cholesky solve on `M`.

use nalgebra::{DMatrix, DVector};

use crate::affine::{Affine, Var, VarLayout};
use crate::task::Task;
use crate::tasks;

/// Which quantities are decision variables. See the module docs for the
/// mapping to OpenSoT / GID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Formulation {
    /// `x = [q̈; f; τ]` — everything explicit, EoM as an equality task.
    Explicit,
    /// `x = [q̈; f]` — τ eliminated through the actuated EoM rows
    /// (OpenSoT's acceleration-level formulation).
    AccelSpace,
    /// `x = [τ; f]` — q̈ eliminated through `M⁻¹` (GID's force-space /
    /// operational-space inverse-dynamics formulation).
    ForceSpace,
}

/// Physical quantities recovered from a solution vector, whichever
/// formulation produced it. Eliminated quantities are evaluated through
/// their affine expressions, so all three fields are always filled.
#[derive(Clone, Debug)]
pub struct Extracted {
    /// Generalised accelerations (`nv`).
    pub qddot: DVector<f64>,
    /// Stacked contact forces (`3·nc`).
    pub forces: DVector<f64>,
    /// Actuated joint torques (`na`).
    pub tau: DVector<f64>,
}

/// The per-tick dynamics context: owns the [`VarLayout`] for the chosen
/// [`Formulation`] and the affine expressions for `q̈` and `τ`.
///
/// Rebuild it each tick from the current `M`, `h`, `Jc` (they are
/// state-dependent), then feed its expressions to the task builders:
///
/// ```
/// use misa_wbc::{Dynamics, Formulation, tasks};
/// use nalgebra::{DMatrix, DVector};
///
/// let (nv, na) = (8, 2);
/// let mass = DMatrix::<f64>::identity(nv, nv);
/// let h = DVector::zeros(nv);
/// let jc = DMatrix::<f64>::zeros(3, nv);
///
/// let dyn_ = Dynamics::new(Formulation::ForceSpace, &mass, &h, &jc, na);
/// // The same call as in the explicit formulation — qddot() happens to
/// // be an expression of [τ; f] here instead of a raw variable block.
/// let contact = tasks::zero_contact_acceleration(dyn_.qddot(), &jc, &DVector::zeros(3));
/// assert!(dyn_.dynamics_task().is_none()); // EoM holds identically
/// ```
#[derive(Clone, Debug)]
pub struct Dynamics {
    formulation: Formulation,
    layout: VarLayout,
    qddot: Affine,
    tau: Affine,
    physics: Option<Task>,
}

impl Dynamics {
    /// Build the context for one tick.
    ///
    /// - `mass`: `M`, `nv × nv`, symmetric positive-definite (required
    ///   for `ForceSpace`, where it is Cholesky-factorised).
    /// - `nle`: `h` (Coriolis/centrifugal + gravity), length `nv`.
    /// - `j_contact`: stacked linear contact Jacobian, `(3·nc) × nv`.
    /// - `na`: number of actuated joints (the **last** `na` of the `nv`
    ///   generalised velocities, matching
    ///   [`tasks::equation_of_motion`]).
    pub fn new(
        formulation: Formulation,
        mass: &DMatrix<f64>,
        nle: &DVector<f64>,
        j_contact: &DMatrix<f64>,
        na: usize,
    ) -> Self {
        let nv = mass.nrows();
        let nf = j_contact.nrows();
        assert_eq!(mass.shape(), (nv, nv), "dynamics: M must be square");
        assert_eq!(nle.len(), nv, "dynamics: h must have length nv");
        assert_eq!(j_contact.ncols(), nv, "dynamics: Jc must have nv columns");
        assert!(na <= nv, "dynamics: na ({na}) must be ≤ nv ({nv})");
        let n_base = nv - na;

        match formulation {
            Formulation::Explicit => {
                let layout = VarLayout::builder()
                    .add("qddot", nv)
                    .add("f", nf)
                    .add("tau", na)
                    .build();
                let (q, f, t) = (layout.var("qddot"), layout.var("f"), layout.var("tau"));
                let physics = tasks::equation_of_motion(&q, &f, &t, mass, nle, j_contact);
                Dynamics {
                    formulation,
                    qddot: q.affine(),
                    tau: t.affine(),
                    physics: Some(physics),
                    layout,
                }
            }
            Formulation::AccelSpace => {
                let layout = VarLayout::builder().add("qddot", nv).add("f", nf).build();
                let (q, f) = (layout.var("qddot"), layout.var("f"));
                let jc_t = j_contact.transpose(); // nv × nf

                // Actuated (bottom na) rows, rearranged for τ:
                //   τ = M_a·q̈ + h_a − (Jcᵀ)_a·f
                let m_a = mass.rows(n_base, na).into_owned();
                let jct_a = jc_t.rows(n_base, na).into_owned();
                let h_a = nle.rows(n_base, na).into_owned();
                let tau = &(&(&m_a * &q) - &(&jct_a * &f)) + &h_a;

                // Underactuated (top n_base) rows stay as the equality
                // task:  M_u·q̈ + h_u − (Jcᵀ)_u·f = 0.
                let m_u = mass.rows(0, n_base).into_owned();
                let jct_u = jc_t.rows(0, n_base).into_owned();
                let h_u = nle.rows(0, n_base).into_owned();
                let residual_u = &(&(&m_u * &q) - &(&jct_u * &f)) + &h_u;
                let physics = (n_base > 0).then(|| Task::soft_eq(&residual_u));

                Dynamics { formulation, qddot: q.affine(), tau, physics, layout }
            }
            Formulation::ForceSpace => {
                let layout = VarLayout::builder().add("tau", na).add("f", nf).build();
                let (t, f) = (layout.var("tau"), layout.var("f"));

                // q̈ = M⁻¹·Sᵀ·τ + M⁻¹·Jcᵀ·f − M⁻¹·h, all through one
                // Cholesky factorisation of M.
                let chol = mass
                    .clone()
                    .cholesky()
                    .expect("dynamics: ForceSpace needs a symmetric positive-definite M");
                let mut s_t = DMatrix::zeros(nv, na);
                for i in 0..na {
                    s_t[(n_base + i, i)] = 1.0;
                }
                let minv_st = chol.solve(&s_t); // nv × na
                let minv_jct = chol.solve(&j_contact.transpose()); // nv × nf
                let minv_h: DVector<f64> = chol.solve(nle); // nv

                let qddot = &(&(&minv_st * &t) + &(&minv_jct * &f)) - &minv_h;

                Dynamics { formulation, qddot, tau: t.affine(), physics: None, layout }
            }
        }
    }

    pub fn formulation(&self) -> Formulation {
        self.formulation
    }

    /// The decision-vector layout for this formulation.
    pub fn layout(&self) -> &VarLayout {
        &self.layout
    }

    /// The contact-force variable (a raw block in every formulation).
    pub fn forces(&self) -> Var {
        self.layout.var("f")
    }

    /// `q̈` as an expression of the decision vector — a raw variable in
    /// `Explicit`/`AccelSpace`, the `M⁻¹(Sᵀτ + Jcᵀf − h)` map in
    /// `ForceSpace`.
    pub fn qddot(&self) -> &Affine {
        &self.qddot
    }

    /// `τ` as an expression of the decision vector — a raw variable in
    /// `Explicit`/`ForceSpace`, the actuated-EoM map in `AccelSpace`.
    pub fn tau(&self) -> &Affine {
        &self.tau
    }

    /// The dynamics-consistency equality this formulation still needs as
    /// a task (place it at priority 0):
    ///
    /// - `Explicit` → the full `nv`-row EoM,
    /// - `AccelSpace` → the `n_base` underactuated rows,
    /// - `ForceSpace` → `None` (the EoM holds identically).
    pub fn dynamics_task(&self) -> Option<Task> {
        self.physics.clone()
    }

    /// Recover all physical quantities from a solution vector, whichever
    /// formulation produced it.
    pub fn extract(&self, x: &DVector<f64>) -> Extracted {
        Extracted {
            qddot: self.qddot.eval(x),
            forces: self.forces().extract(x),
            tau: self.tau.eval(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small non-trivial consistent system: SPD M with coupling,
    /// nonzero h, a contact Jacobian sensing base + joints.
    fn toy() -> (usize, usize, DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
        let (nv, na) = (8usize, 2usize);
        // M = I + 0.1·L·Lᵀ  (SPD, deterministic coupling)
        let l = DMatrix::from_fn(nv, 3, |i, j| ((i + 2 * j) as f64 * 0.37).sin());
        let mass = DMatrix::<f64>::identity(nv, nv) + 0.1 * (&l * l.transpose());
        let h = DVector::from_fn(nv, |i, _| 0.3 * (i as f64 + 1.0).cos());
        let mut jc = DMatrix::zeros(3, nv);
        for i in 0..3 {
            jc[(i, i)] = 1.0;
        }
        jc[(0, 6)] = 0.2;
        jc[(1, 7)] = -0.1;
        (nv, na, mass, h, jc)
    }

    fn probe(n: usize) -> DVector<f64> {
        DVector::from_fn(n, |i, _| ((i as f64) * 0.71 - 1.3).sin())
    }

    /// In every formulation, the expressions must satisfy the EoM
    /// identity  M·q̈(x) + h − Sᵀ·τ(x) − Jcᵀ·f(x) = 0  for any x that
    /// satisfies the formulation's dynamics task (identically for
    /// ForceSpace / AccelSpace-actuated-rows by construction).
    #[test]
    fn expressions_satisfy_eom_identity() {
        let (nv, na, mass, h, jc) = toy();
        let n_base = nv - na;
        let mut s_t = DMatrix::zeros(nv, na);
        for i in 0..na {
            s_t[(n_base + i, i)] = 1.0;
        }

        // ForceSpace: identity must hold for EVERY x (q̈ is defined by it).
        let d = Dynamics::new(Formulation::ForceSpace, &mass, &h, &jc, na);
        let x = probe(d.layout().n_decision());
        let e = d.extract(&x);
        let r = &mass * &e.qddot + &h - &s_t * &e.tau - jc.transpose() * &e.forces;
        assert!(r.norm() < 1e-10, "ForceSpace EoM identity violated: {}", r.norm());

        // AccelSpace: the actuated rows must hold for every x (τ is
        // defined by them); the base rows are the remaining task.
        let d = Dynamics::new(Formulation::AccelSpace, &mass, &h, &jc, na);
        let x = probe(d.layout().n_decision());
        let e = d.extract(&x);
        let r = &mass * &e.qddot + &h - &s_t * &e.tau - jc.transpose() * &e.forces;
        assert!(
            r.rows(n_base, na).norm() < 1e-10,
            "AccelSpace actuated-row identity violated"
        );
        let task = d.dynamics_task().expect("base rows remain");
        assert_eq!(task.n_eq(), n_base);
        // The task's residual equals the base rows of the EoM residual.
        let task_res = &task.a * &x - &task.b;
        assert!((task_res - r.rows(0, n_base)).norm() < 1e-10);
    }

    /// Explicit formulation reproduces the existing equation_of_motion
    /// task and raw-variable expressions.
    #[test]
    fn explicit_matches_task_catalogue() {
        let (nv, na, mass, h, jc) = toy();
        let d = Dynamics::new(Formulation::Explicit, &mass, &h, &jc, na);
        assert_eq!(d.layout().n_decision(), nv + 3 + na);
        let task = d.dynamics_task().expect("explicit keeps the full EoM");
        assert_eq!(task.n_eq(), nv);

        let x = probe(d.layout().n_decision());
        let e = d.extract(&x);
        // Raw blocks: extraction is just slicing.
        assert_eq!(e.qddot, x.rows(0, nv).into_owned());
        assert_eq!(e.tau, x.rows(nv + 3, na).into_owned());
    }

    /// The same physical (q̈, f, τ) triple maps to consistent task
    /// residuals across formulations: a Cartesian task built from each
    /// formulation's qddot() expression evaluates to the same value.
    #[test]
    fn cartesian_task_agrees_across_formulations() {
        let (nv, na, mass, h, jc) = toy();
        let j_task = DMatrix::from_fn(3, nv, |i, j| ((i * 5 + j) as f64 * 0.23).cos());
        let dj_v = DVector::from_vec(vec![0.1, -0.2, 0.3]);
        let aref = DVector::from_vec(vec![1.0, 0.5, -0.7]);

        // Pick a physical state: choose x_force in ForceSpace, extract
        // the physical triple, then embed it in the other formulations'
        // decision vectors.
        let df = Dynamics::new(Formulation::ForceSpace, &mass, &h, &jc, na);
        let xf = probe(df.layout().n_decision());
        let e = df.extract(&xf);

        let t_force = crate::tasks::cartesian_acceleration(df.qddot(), &j_task, &dj_v, &aref);
        let rf = &t_force.a * &xf - &t_force.b;

        // Explicit: x = [q̈; f; τ].
        let de = Dynamics::new(Formulation::Explicit, &mass, &h, &jc, na);
        let mut xe = DVector::zeros(de.layout().n_decision());
        xe.rows_mut(0, nv).copy_from(&e.qddot);
        xe.rows_mut(nv, 3).copy_from(&e.forces);
        xe.rows_mut(nv + 3, na).copy_from(&e.tau);
        let t_exp = crate::tasks::cartesian_acceleration(de.qddot(), &j_task, &dj_v, &aref);
        let re = &t_exp.a * &xe - &t_exp.b;

        assert!((rf - re).norm() < 1e-10, "task residuals differ across formulations");
    }
}
