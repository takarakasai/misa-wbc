//! Reference generators: turn `(target, state, gains)` into the
//! acceleration reference a task tracks — GID's `SetAccel` / `SetVeloc`
//! / `SetValue` / `SetImpedance` ladder as free functions.
//!
//! The task builders (e.g. [`crate::tasks::cartesian_acceleration`])
//! deliberately take a plain acceleration vector so the core stays
//! gain-agnostic; these helpers are the standard ways to produce that
//! vector. All gains are per-axis (`DVector`), matching diagonal
//! `Kp`/`Kd`; for a scalar gain pass `DVector::repeat(n, k)`.
//!
//! | this module | GID equivalent | law |
//! |---|---|---|
//! | (pass `a_ref` directly) | `SetAccel` | feed-forward |
//! | [`vel`] | `SetVeloc` | `(v_ref − v)/T` |
//! | [`pd`] | `SetValue` (cascaded PD) | `kp∘(x_ref−x) + kd∘(v_ref−v)` |
//! | [`impedance`] | `SetImpedance` | `k∘(x_ref−x) − d∘v` (= [`pd`] with `v_ref = 0`) |

use nalgebra::DVector;

/// P-on-velocity: reach `v_ref` over time-constant `t` — the reference
/// acceleration `(v_ref − v) / t`. GID's `SetVeloc`.
pub fn vel(v_ref: &DVector<f64>, v: &DVector<f64>, t: f64) -> DVector<f64> {
    assert_eq!(v_ref.len(), v.len(), "refgen::vel: dimension mismatch");
    assert!(t > 0.0, "refgen::vel: time constant must be > 0");
    (v_ref - v) / t
}

/// PD tracking with feed-forward:
/// `a = a_ff + kp∘(x_ref − x) + kd∘(v_ref − v)`, element-wise gains.
/// The standard operational-space servo (GID's `SetValue`, OpenSoT's
/// Cartesian `lambda` gains, the classic `ẍ_cmd`).
#[allow(clippy::too_many_arguments)]
pub fn pd_ff(
    x_ref: &DVector<f64>,
    x: &DVector<f64>,
    v_ref: &DVector<f64>,
    v: &DVector<f64>,
    kp: &DVector<f64>,
    kd: &DVector<f64>,
    a_ff: &DVector<f64>,
) -> DVector<f64> {
    let n = x.len();
    assert!(
        [x_ref.len(), v_ref.len(), v.len(), kp.len(), kd.len(), a_ff.len()]
            .iter()
            .all(|&l| l == n),
        "refgen::pd_ff: dimension mismatch"
    );
    a_ff + kp.component_mul(&(x_ref - x)) + kd.component_mul(&(v_ref - v))
}

/// PD tracking without feed-forward:
/// `a = kp∘(x_ref − x) + kd∘(v_ref − v)`.
pub fn pd(
    x_ref: &DVector<f64>,
    x: &DVector<f64>,
    v_ref: &DVector<f64>,
    v: &DVector<f64>,
    kp: &DVector<f64>,
    kd: &DVector<f64>,
) -> DVector<f64> {
    pd_ff(x_ref, x, v_ref, v, kp, kd, &DVector::zeros(x.len()))
}

/// Virtual spring-damper toward `x_ref` at rest:
/// `a = k∘(x_ref − x) − d∘v`. GID's `SetImpedance` (equivalently
/// [`pd`] with `v_ref = 0`).
pub fn impedance(
    x_ref: &DVector<f64>,
    x: &DVector<f64>,
    v: &DVector<f64>,
    k: &DVector<f64>,
    d: &DVector<f64>,
) -> DVector<f64> {
    pd(x_ref, x, &DVector::zeros(v.len()), v, k, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_laws() {
        let x_ref = DVector::from_vec(vec![1.0, 2.0]);
        let x = DVector::from_vec(vec![0.5, 2.5]);
        let v = DVector::from_vec(vec![0.1, -0.2]);
        let zeros = DVector::zeros(2);
        let kp = DVector::from_vec(vec![100.0, 100.0]);
        let kd = DVector::from_vec(vec![20.0, 20.0]);

        // vel: (v_ref − v)/T
        let a = vel(&zeros, &v, 0.5);
        assert!((a[0] - (-0.2)).abs() < 1e-12 && (a[1] - 0.4).abs() < 1e-12);

        // pd: kp·err + kd·(−v)
        let a = pd(&x_ref, &x, &zeros, &v, &kp, &kd);
        assert!((a[0] - (100.0 * 0.5 - 20.0 * 0.1)).abs() < 1e-12);
        assert!((a[1] - (100.0 * -0.5 + 20.0 * 0.2)).abs() < 1e-12);

        // impedance == pd with v_ref = 0
        let b = impedance(&x_ref, &x, &v, &kp, &kd);
        assert!((a - b).norm() < 1e-12);
    }
}
