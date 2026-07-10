//! A catalogue of ready-made whole-body-control tasks, built from the
//! matrices a rigid-body-dynamics engine produces (`M`, `h`, `J`,
//! `J̇·v`) over an [`Affine`](crate::Affine) variable layout.
//!
//! Every function here is a pure builder: hand it the relevant
//! [`Var`]s and matrices for the current tick and it returns
//! a [`Task`]. Nothing is stateful and nothing knows about a specific
//! robot — a quadruped host chooses which contacts are stance and slots
//! the right rows in; an arm host uses the same `track` /
//! `cartesian_acceleration` with its own Jacobian.
//!
//! The catalogue mirrors OpenSoT's task / constraint families
//! (`tasks::acceleration`, `constraints::force`) but collapses the ones
//! that share a shape into a few generic primitives:
//!
//! | this module | covers (OpenSoT / quadruped-gait) |
//! |---|---|
//! | [`equation_of_motion`] | DynamicFeasibility / floating-base EoM |
//! | [`cartesian_acceleration`] | acceleration::Cartesian / CoM / Contact |
//! | [`track`] | acceleration::Postural, swing-leg, force/τ regularisation |
//! | [`friction_pyramid`] | constraints::force::FrictionCone |
//! | [`patch_contact`] | GID SetContactSupport / OpenSoT CoP+WrenchLimits |
//! | [`centroidal_momentum`] | GID Momentum unit / OpenSoT CoM+AngularMomentum |
//! | [`box_bound`] | TorqueLimits / WrenchLimits (symmetric) |
//!
//! Decision-vector layout is the caller's: these builders only touch the
//! variables passed in, so `x = [q̈; f; τ]` (τ explicit, the
//! quadruped-gait convention) and `x = [q̈; f]` (τ eliminated, the
//! OpenSoT convention) are both expressible.

use nalgebra::{DMatrix, DVector};

use crate::affine::{AsAffine, Var};
use crate::task::Task;

/// Soft-equality tracking: drive an affine expression to a reference,
/// `expr ≈ reference`, as a least-squares cost. The generic backbone of
/// postural / swing-leg / Cartesian / regularisation tasks.
///
/// `reference` length must equal `expr.out_size()`.
pub fn track(expr: &impl AsAffine, reference: &DVector<f64>) -> Task {
    Task::soft_eq(&(&expr.as_affine() - reference))
}

/// Regularise a variable toward a target value (`var ≈ target`) — a
/// thin, well-named wrapper over [`track`] for force / torque / posture
/// regularisation. `target` length must equal `var.size()`.
pub fn regularize(var: &impl AsAffine, target: &DVector<f64>) -> Task {
    track(var, target)
}

/// Symmetric box bound `−max ≤ var ≤ max` as a hard inequality. Covers
/// actuator torque limits and (symmetric) wrench limits. All entries of
/// `max` should be ≥ 0; `max` length must equal `var.size()`.
pub fn box_bound(var: &impl AsAffine, max: &DVector<f64>) -> Task {
    assert_eq!(
        var.out_size(),
        max.len(),
        "box_bound: expression size ({}) must equal max len ({})",
        var.out_size(),
        max.len(),
    );
    let neg = -max;
    Task::in_range(&neg, &var.as_affine(), max)
}

/// Cartesian (task-space) acceleration tracking:
/// `J·q̈ + J̇·v ≈ accel_ref`, i.e. drive the operational-space
/// acceleration to `accel_ref`. Feed a PD+feed-forward command in as
/// `accel_ref = ẍ_ref + Kd·(ẋ_ref − ẋ) + Kp·(x_ref − x)` (the caller
/// forms it; misa-wbc stays gain-agnostic).
///
/// - `qddot`: the joint-acceleration variable.
/// - `j`: task Jacobian, `m × nv` (`m` = task dim: 6 for a pose, 3 for a
///   point / CoM).
/// - `dj_v`: the bias `J̇·v`, length `m`.
/// - `accel_ref`: desired task acceleration, length `m`.
pub fn cartesian_acceleration(
    qddot: &impl AsAffine,
    j: &DMatrix<f64>,
    dj_v: &DVector<f64>,
    accel_ref: &DVector<f64>,
) -> Task {
    assert_eq!(j.ncols(), qddot.out_size(), "cartesian: J cols must equal qddot size");
    assert_eq!(j.nrows(), dj_v.len(), "cartesian: J rows must equal dj_v len");
    assert_eq!(j.nrows(), accel_ref.len(), "cartesian: J rows must equal accel_ref len");
    // expr = J·q̈ + J̇·v ;  track toward accel_ref.
    let expr = &(j * &qddot.as_affine()) + dj_v;
    track(&expr, accel_ref)
}

/// Rigid-contact no-motion constraint: a stance point holds still, so
/// its Cartesian acceleration is zero, `J·q̈ + J̇·v = 0`. The zero-
/// reference special case of [`cartesian_acceleration`], emitted as a
/// (soft) equality — place it at priority 0 for a hard contact.
pub fn zero_contact_acceleration(
    qddot: &impl AsAffine,
    j: &DMatrix<f64>,
    dj_v: &DVector<f64>,
) -> Task {
    cartesian_acceleration(qddot, j, dj_v, &DVector::zeros(j.nrows()))
}

/// The floating-base equation of motion as an equality task, in the
/// **τ-explicit** convention `x = [… q̈ … f … τ …]`:
///
/// ```text
///   M·q̈  −  Jcᵀ·f  −  Sᵀ·τ  =  −h
/// ```
///
/// where `S = [0 | I_na]` selects the actuated joints (so `Sᵀ·τ` has
/// zeros in the `n_base = nv − na` floating-base rows). All `nv` rows are
/// emitted; the floating-base rows enforce underactuation (no τ), the
/// actuated rows relate τ to `q̈` and `f`. Place at priority 0.
///
/// - `qddot` (size `nv`), `forces` (size `3·nc` — stacked linear contact
///   forces), `tau` (size `na`).
/// - `mass`: `M`, `nv × nv`. `nle`: `h`, length `nv`.
/// - `j_contact`: the stacked **linear** contact Jacobian, `(3·nc) × nv`.
///
/// A swing foot is handled by the caller pinning its force block to zero
/// (e.g. [`regularize`] the swing sub-force to 0 at priority 0, or a
/// friction task that excludes it) — this builder itself treats all
/// contact columns uniformly.
pub fn equation_of_motion(
    qddot: &Var,
    forces: &Var,
    tau: &Var,
    mass: &DMatrix<f64>,
    nle: &DVector<f64>,
    j_contact: &DMatrix<f64>,
) -> Task {
    let nv = qddot.size();
    let na = tau.size();
    assert_eq!(mass.shape(), (nv, nv), "eom: M must be nv × nv");
    assert_eq!(nle.len(), nv, "eom: h must have length nv");
    assert_eq!(j_contact.ncols(), nv, "eom: j_contact must have nv columns");
    assert_eq!(j_contact.nrows(), forces.size(), "eom: j_contact rows must equal forces size");
    assert!(na <= nv, "eom: na ({na}) must be ≤ nv ({nv})");

    let n_base = nv - na;
    // Sᵀ: nv × na, identity in the bottom na rows (the actuated joints).
    let mut s_t = DMatrix::zeros(nv, na);
    for i in 0..na {
        s_t[(n_base + i, i)] = 1.0;
    }
    let jc_t = j_contact.transpose(); // nv × 3nc

    // residual  e = M·q̈ − Jcᵀ·f − Sᵀ·τ + h  →  soft_eq drives it to 0.
    let m_qddot = mass * qddot;
    let jct_f = &jc_t * forces;
    let st_tau = &s_t * tau;
    let e = &(&(&m_qddot - &jct_f) - &st_tau) + nle;
    Task::soft_eq(&e)
}

/// Linearised Coulomb friction pyramid for one **point** contact,
/// `C·f ≤ 0` (5 rows), assuming the surface normal is the contact
/// frame's +z (the common flat-ground / point-foot case):
///
/// ```text
///   −fz            ≤ 0     (unilateral: push only)
///    fx − μ·fz     ≤ 0
///   −fx − μ·fz     ≤ 0
///    fy − μ·fz     ≤ 0
///   −fy − μ·fz     ≤ 0
/// ```
///
/// `force` is the 3-vector force variable (or a per-contact sub-`Var`)
/// for the contact. For a rotated contact normal, pre-multiply the
/// caller's force by the world→contact rotation, or use a rotated `C`
/// (a `world_to_contact` overload lands with the model integration).
pub fn friction_pyramid(force: &impl AsAffine, mu: f64) -> Task {
    assert_eq!(force.out_size(), 3, "friction_pyramid: force expression must be 3-D");
    #[rustfmt::skip]
    let c = DMatrix::from_row_slice(5, 3, &[
        0.0,  0.0, -1.0,
        1.0,  0.0, -mu,
       -1.0,  0.0, -mu,
        0.0,  1.0, -mu,
        0.0, -1.0, -mu,
    ]);
    let expr = &c * &force.as_affine();
    Task::le(&expr, &DVector::zeros(5))
}

/// Centroidal momentum-rate tracking:
/// `A_G·q̈ + Ȧ_G·v ≈ h_rate_ref` — drive the whole-body (centroidal)
/// momentum rate `ḣ = [ḣ_ang; ḣ_lin]` to a reference. THE balance
/// primitive: GID's Momentum operation unit, OpenSoT's CoM/angular-
/// momentum tasks.
///
/// - `qddot`: the joint-acceleration expression (any formulation).
/// - `cmm`: the Centroidal Momentum Matrix `A_G`, `6 × nv`
///   (misarta: `compute_centroidal_momentum_matrix`).
/// - `dcmm_v`: the bias `Ȧ_G·v`, length 6
///   (misarta: `compute_cmm_dot_times_v`).
/// - `h_rate_ref`: desired momentum rate, length 6 (`[ang; lin]`, the
///   ecosystem row convention). E.g. for CoM stabilisation feed the
///   linear rows `m·(ẍ_com)_ref` from [`crate::refgen`] and zero (or a
///   damping law) on the angular rows.
///
/// Row-selection (linear-only / angular-only) is the caller slicing
/// `cmm` / `dcmm_v` / the reference before the call.
pub fn centroidal_momentum(
    qddot: &impl AsAffine,
    cmm: &DMatrix<f64>,
    dcmm_v: &DVector<f64>,
    h_rate_ref: &DVector<f64>,
) -> Task {
    assert_eq!(cmm.ncols(), qddot.out_size(), "momentum: CMM cols must equal qddot size");
    assert_eq!(cmm.nrows(), dcmm_v.len(), "momentum: CMM rows must equal dcmm_v len");
    assert_eq!(cmm.nrows(), h_rate_ref.len(), "momentum: CMM rows must equal ref len");
    // Same shape as cartesian_acceleration with J = A_G — named so the
    // balance intent reads at call sites.
    cartesian_acceleration(qddot, cmm, dcmm_v, h_rate_ref)
}

/// Parameters of a rectangular **surface (patch) contact** — the full
/// GID `SetContactSupport` set: Coulomb friction, centre-of-pressure
/// (CoP/ZMP) box, torsional friction and unilaterality, all coupled to
/// the normal force.
#[derive(Clone, Copy, Debug)]
pub struct ContactPatch {
    /// Coulomb friction coefficient μ (tangential ≤ μ·fz).
    pub mu: f64,
    /// Half-lengths (`Lx`, `Ly`) of the support rectangle: the CoP must
    /// satisfy `|my| ≤ Lx·fz`, `|mx| ≤ Ly·fz`.
    pub cop_half: (f64, f64),
    /// Torsional friction coefficient (`|mz| ≤ μ_t·fz`).
    pub mu_torsion: f64,
    /// Upper bound on the normal force (`fz ≤ f_max`).
    pub f_max: f64,
}

/// The 12 linear inequality rows of a rectangular patch contact,
/// `C·w ≤ f`, over a 6-D contact **wrench** expression `w = [m; f]`
/// (moment rows 0–2, force rows 3–5 — the dual of this ecosystem's
/// `[angular; linear]` twist row convention, so `Jᵀ·w` works directly
/// with a misarta 6-D contact Jacobian). Assumes the surface normal is
/// the contact frame's +z:
///
/// ```text
///   −fz ≤ 0,             fz ≤ f_max          (unilateral + cap)
///   ±fx − μ·fz    ≤ 0,   ±fy − μ·fz    ≤ 0   (friction pyramid)
///   ±mx − Ly·fz   ≤ 0,   ±my − Lx·fz   ≤ 0   (CoP inside the patch)
///   ±mz − μt·fz   ≤ 0                        (torsional friction)
/// ```
///
/// This is GID's `Set*ContactSupport` constraint set (its `Multiplier`
/// coupling rows) as one task. For a 3-D point contact use
/// [`friction_pyramid`] instead.
pub fn patch_contact(wrench: &impl AsAffine, patch: &ContactPatch) -> Task {
    assert_eq!(wrench.out_size(), 6, "patch_contact: wrench must be 6-D [m; f]");
    let (mx, my, mz, fx, fy, fz) = (0, 1, 2, 3, 4, 5);
    let (lx, ly) = patch.cop_half;
    let mut c = DMatrix::zeros(12, 6);
    let mut f = DVector::zeros(12);
    // unilateral + cap
    c[(0, fz)] = -1.0;
    c[(1, fz)] = 1.0;
    f[1] = patch.f_max;
    // friction pyramid
    for (row, axis, sign) in [(2, fx, 1.0), (3, fx, -1.0), (4, fy, 1.0), (5, fy, -1.0)] {
        c[(row, axis)] = sign;
        c[(row, fz)] = -patch.mu;
    }
    // CoP box:  |mx| ≤ Ly·fz,  |my| ≤ Lx·fz
    for (row, axis, sign, half) in
        [(6, mx, 1.0, ly), (7, mx, -1.0, ly), (8, my, 1.0, lx), (9, my, -1.0, lx)]
    {
        c[(row, axis)] = sign;
        c[(row, fz)] = -half;
    }
    // torsional friction
    c[(10, mz)] = 1.0;
    c[(10, fz)] = -patch.mu_torsion;
    c[(11, mz)] = -1.0;
    c[(11, fz)] = -patch.mu_torsion;

    let expr = &c * &wrench.as_affine();
    Task::le(&expr, &f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarLayout;

    fn layout(nv: usize, nc: usize, na: usize) -> VarLayout {
        VarLayout::builder()
            .add("qddot", nv)
            .add("f", 3 * nc)
            .add("tau", na)
            .build()
    }

    #[test]
    fn track_and_regularize() {
        let l = layout(2, 0, 2);
        let tau = l.var("tau");
        let t = regularize(&tau, &DVector::from_vec(vec![1.0, -2.0]));
        // A selects the τ block; b = target.
        assert_eq!(t.n_eq(), 2);
        // At x with τ = target the residual is zero.
        let mut x = DVector::zeros(l.n_decision());
        x[l.var("tau").offset()] = 1.0;
        x[l.var("tau").offset() + 1] = -2.0;
        assert!((&t.a * &x - &t.b).norm() < 1e-12);
    }

    #[test]
    fn box_bound_is_symmetric_pair() {
        let l = layout(0, 0, 2);
        let tau = l.var("tau");
        let t = box_bound(&tau, &DVector::from_vec(vec![3.0, 5.0]));
        // 2·na inequality rows:  τ ≤ max  and  −τ ≤ max.
        assert_eq!(t.n_iq(), 4);
        // Row set encodes ±I with f = [3,5,3,5].
        let x = DVector::from_vec(vec![3.0, 5.0]);
        let lhs = &t.d * &x; // [τ ; −τ]
        for i in 0..4 {
            assert!(lhs[i] <= t.f[i] + 1e-12, "row {i} violates");
        }
    }

    #[test]
    fn cartesian_acceleration_residual() {
        // 1-D toy: J = [1 0], dj_v = 0.5, accel_ref = 2.0.  q̈=[a,b].
        let l = layout(2, 0, 0);
        let q = l.var("qddot");
        let j = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let dj_v = DVector::from_vec(vec![0.5]);
        let aref = DVector::from_vec(vec![2.0]);
        let t = cartesian_acceleration(&q, &j, &dj_v, &aref);
        // residual = J·q̈ + dj_v − aref = a + 0.5 − 2 = a − 1.5.
        // As Task: a·x = b  →  [1 0]·x = 1.5.
        assert_eq!(t.n_eq(), 1);
        let x = DVector::from_vec(vec![1.5, 9.9]);
        assert!((&t.a * &x - &t.b).norm() < 1e-12);
    }

    #[test]
    fn friction_pyramid_rows() {
        let l = layout(0, 1, 0);
        let f = l.var("f");
        let t = friction_pyramid(&f, 0.5);
        assert_eq!(t.n_iq(), 5);
        // A force inside the cone (fz large, fx/fy small) satisfies C·f ≤ 0.
        let x = DVector::from_vec(vec![0.1, -0.1, 1.0]);
        let lhs = &t.d * &x;
        for i in 0..5 {
            assert!(lhs[i] <= 1e-12, "inside-cone force violates row {i}: {}", lhs[i]);
        }
        // A pulling force (fz < 0) violates the unilateral row.
        let pull = DVector::from_vec(vec![0.0, 0.0, -1.0]);
        assert!((&t.d * &pull)[0] > 0.0, "pull should violate unilateral row");
    }

    #[test]
    fn patch_contact_rows() {
        let l = VarLayout::builder().add("w", 6).build();
        let w = l.var("w");
        let patch = ContactPatch { mu: 0.5, cop_half: (0.10, 0.05), mu_torsion: 0.02, f_max: 200.0 };
        let t = patch_contact(&w, &patch);
        assert_eq!(t.n_iq(), 12);

        // A wrench well inside every cone: fz = 100, small everything else.
        //   w = [mx, my, mz, fx, fy, fz]
        let ok = DVector::from_vec(vec![1.0, 2.0, 0.5, 10.0, -10.0, 100.0]);
        let m = &t.f - &t.d * &ok;
        assert!(m.min() > 0.0, "inside wrench should have positive margin: {}", m.min());

        // Violations, one row family at a time.
        let pull = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.0]);
        assert!((&t.f - &t.d * &pull).min() < 0.0, "pulling must violate");
        let cap = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 300.0]);
        assert!((&t.f - &t.d * &cap).min() < 0.0, "f_max must bind");
        let slip = DVector::from_vec(vec![0.0, 0.0, 0.0, 80.0, 0.0, 100.0]);
        assert!((&t.f - &t.d * &slip).min() < 0.0, "friction must bind (80 > 0.5·100)");
        let cop = DVector::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0, 100.0]);
        assert!((&t.f - &t.d * &cop).min() < 0.0, "CoP must bind (10 > 0.05·100)");
        let twist = DVector::from_vec(vec![0.0, 0.0, 5.0, 0.0, 0.0, 100.0]);
        assert!((&t.f - &t.d * &twist).min() < 0.0, "torsion must bind (5 > 0.02·100)");
    }

    #[test]
    fn centroidal_momentum_is_cartesian_with_cmm() {
        let l = layout(4, 0, 0);
        let q = l.var("qddot");
        let cmm = DMatrix::from_fn(6, 4, |i, j| ((i + j) as f64 * 0.31).sin());
        let dcmm_v = DVector::from_fn(6, |i, _| i as f64 * 0.1);
        let href = DVector::from_fn(6, |i, _| 1.0 - i as f64 * 0.2);
        let t = centroidal_momentum(&q, &cmm, &dcmm_v, &href);
        assert_eq!(t.n_eq(), 6);
        // Residual definition: A_G·q̈ = href − dcmm_v.
        let x = DVector::from_vec(vec![0.3, -0.4, 0.5, 0.9]);
        let want = &cmm * &x - (&href - &dcmm_v);
        assert!(((&t.a * &x - &t.b) - want).norm() < 1e-12);
    }

    /// EoM residual matches the hand-assembled `M·q̈ − Jcᵀ·f − Sᵀ·τ + h`
    /// (ported from quadruped-gait's `matches_eom_residual_definition`).
    #[test]
    fn equation_of_motion_matches_hand_assembly() {
        let (nv, nc, na) = (9, 2, 3);
        let l = layout(nv, nc, na);
        let (q, f, tau) = (l.var("qddot"), l.var("f"), l.var("tau"));

        let mass = DMatrix::<f64>::identity(nv, nv);
        let nle = DVector::from_fn(nv, |i, _| i as f64 * 0.1);
        let jc = DMatrix::from_fn(3 * nc, nv, |i, j| ((i * 3 + j) as f64).sin());

        let t = equation_of_motion(&q, &f, &tau, &mass, &nle, &jc);
        assert_eq!(t.n_eq(), nv);

        // Hand assembly: A = [M | −Jcᵀ | −S], b = −h, S bottom-na identity.
        let n = l.n_decision();
        let mut a = DMatrix::zeros(nv, n);
        a.view_mut((0, q.offset()), (nv, nv)).copy_from(&mass);
        a.view_mut((0, f.offset()), (nv, 3 * nc)).copy_from(&(-jc.transpose()));
        let n_base = nv - na;
        for i in 0..na {
            a[(n_base + i, tau.offset() + i)] = -1.0;
        }
        let b = -&nle;
        assert!((&t.a - &a).norm() < 1e-12, "A differs");
        assert!((&t.b - &b).norm() < 1e-12, "b differs");
    }
}
