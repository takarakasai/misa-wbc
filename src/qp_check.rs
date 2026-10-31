//! Independent residuals for the original, unregularized dense QP.
use crate::qp::QpSolution;
use nalgebra::{DMatrix, DVector};
impl QpSolution {
    /// Strict completion status. Clarabel's relaxed AlmostSolved outcome is
    /// retained for compatibility by the solver but is not benchmark-qualified.
    pub fn formally_solved(&self) -> bool {
        self.status == crate::qp::QpStatus::Optimal
            && self
                .diagnostics
                .as_ref()
                .is_none_or(|d| d.raw_status != "AlmostSolved")
    }
}

#[derive(Debug, Clone)]
pub struct Residuals {
    pub finite: bool,
    pub primal_raw: f64,
    pub primal_scaled: f64,
    pub stationarity_raw: f64,
    pub stationarity_scaled: f64,
    pub complementarity_raw: f64,
    pub complementarity_scaled: f64,
    pub dual_sign: f64,
}
impl Residuals {
    pub fn accepted(&self, tol: f64) -> bool {
        self.finite
            && self.primal_scaled <= tol
            && self.stationarity_scaled <= tol
            && self.complementarity_scaled <= tol
            && self.dual_sign <= tol
    }
}
fn inf(v: &DVector<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}
/// Row primal scale: max(1, |b_i|, sum_j |A_ij x_j|).
/// Stationarity scale: max(1, ||Hx||inf, ||c||inf, ||Aeq'lambda||inf, ||Aiq'lambda||inf).
/// Complementarity scale: max(1, |lambda_i| max(|b_i|, |A_i x|)).
pub fn check(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    eq: Option<(&DMatrix<f64>, &DVector<f64>)>,
    iq: Option<(&DMatrix<f64>, &DVector<f64>)>,
    s: &QpSolution,
) -> Residuals {
    let mut r = Residuals {
        finite: true,
        primal_raw: 0.,
        primal_scaled: 0.,
        stationarity_raw: 0.,
        stationarity_scaled: 0.,
        complementarity_raw: 0.,
        complementarity_scaled: 0.,
        dual_sign: 0.,
    };
    r.finite = h
        .iter()
        .chain(c.iter())
        .chain(s.x.iter())
        .chain(s.lambda_eq.iter())
        .chain(s.lambda_iq.iter())
        .all(|x| x.is_finite());
    r.finite &= s.x.len() == c.len()
        && h.shape() == (c.len(), c.len())
        && eq.map_or(s.lambda_eq.is_empty(), |(a, b)| {
            a.ncols() == c.len() && a.nrows() == b.len() && b.len() == s.lambda_eq.len()
        })
        && iq.map_or(s.lambda_iq.is_empty(), |(a, b)| {
            a.ncols() == c.len() && a.nrows() == b.len() && b.len() == s.lambda_iq.len()
        });
    if !r.finite {
        return r;
    }
    let hx = h * &s.x;
    let mut station = &hx + c;
    let mut scale = 1_f64.max(inf(&hx)).max(inf(c));
    for (pair, lambda, equality) in [(eq, &s.lambda_eq, true), (iq, &s.lambda_iq, false)] {
        if let Some((a, b)) = pair {
            if !a.iter().chain(b.iter()).all(|x| x.is_finite()) {
                r.finite = false;
                return r;
            }
            let ax = a * &s.x;
            let term = a.transpose() * lambda;
            scale = scale.max(inf(&term));
            r.finite &= ax.iter().chain(term.iter()).all(|v| v.is_finite());
            station += term;
            for i in 0..b.len() {
                let d = ax[i] - b[i];
                let violation = if equality { d.abs() } else { d.max(0.) };
                let row_scale = (0..a.ncols())
                    .map(|j| (a[(i, j)] * s.x[j]).abs())
                    .sum::<f64>()
                    .max(b[i].abs())
                    .max(1.);
                r.finite &= row_scale.is_finite() && d.is_finite();
                r.primal_raw = r.primal_raw.max(violation);
                r.primal_scaled = r.primal_scaled.max(violation / row_scale);
                if !equality {
                    let comp = (lambda[i] * d).abs();
                    r.complementarity_raw = r.complementarity_raw.max(comp);
                    r.complementarity_scaled = r
                        .complementarity_scaled
                        .max(comp / (lambda[i].abs() * b[i].abs().max(ax[i].abs())).max(1.));
                    r.dual_sign = r.dual_sign.max(-lambda[i]);
                }
            }
        }
    }
    r.finite &= hx.iter().chain(station.iter()).all(|v| v.is_finite()) && scale.is_finite();
    r.stationarity_raw = inf(&station);
    r.stationarity_scaled = r.stationarity_raw / scale;
    r.finite &= [
        r.primal_raw,
        r.primal_scaled,
        r.stationarity_raw,
        r.stationarity_scaled,
        r.complementarity_raw,
        r.complementarity_scaled,
        r.dual_sign,
    ]
    .iter()
    .all(|x| x.is_finite());
    r
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::{QpConfig, solve_qp};
    #[test]
    fn relaxed_status_is_not_formal_success_even_with_exact_residuals() {
        let h = DMatrix::identity(1, 1);
        let c = DVector::from_element(1, -1.);
        let mut sol = solve_qp(&h, &c, None, None, None, None, None, &QpConfig::default());
        assert!(sol.formally_solved());
        sol.diagnostics = Some(crate::qp::QpDiagnostics {
            raw_status: "AlmostSolved".into(),
            ..Default::default()
        });
        assert!(check(&h, &c, None, None, &sol).accepted(1e-6));
        assert!(!sol.formally_solved());
    }

    #[test]
    fn known_solution_and_nonfinite() {
        let h = DMatrix::identity(1, 1);
        let c = DVector::from_element(1, -2.);
        let a = DMatrix::identity(1, 1);
        let b = DVector::from_element(1, 1.);
        let mut s = solve_qp(
            &h,
            &c,
            None,
            None,
            Some(&a),
            Some(&b),
            None,
            &QpConfig::default(),
        );
        assert!(check(&h, &c, None, Some((&a, &b)), &s).accepted(1e-6));
        s.x[0] = f64::NAN;
        assert!(!check(&h, &c, None, Some((&a, &b)), &s).accepted(1e-6));
    }
    #[test]
    fn infeasible_and_wrong_dual_rejected() {
        let h = DMatrix::identity(1, 1);
        let c = DVector::from_element(1, -2.);
        let a = DMatrix::identity(1, 1);
        let b = DVector::from_element(1, 1.);
        let mut s = solve_qp(
            &h,
            &c,
            None,
            None,
            Some(&a),
            Some(&b),
            None,
            &QpConfig::default(),
        );
        s.lambda_iq[0] = -1.;
        assert!(!check(&h, &c, None, Some((&a, &b)), &s).accepted(1e-6));
        s.x[0] = 2.;
        assert!(!check(&h, &c, None, Some((&a, &b)), &s).accepted(1e-6));
    }
}

#[cfg(test)]
mod degenerate_tests {
    use super::*;
    use crate::qp::{QpConfig, QpSolver, solve_qp};
    #[test]
    fn dependent_rows_and_zero_load() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-1., -1.]);
        let a = DMatrix::from_row_slice(3, 2, &[1., 0., 2., 0., 0., 1.]);
        let b = DVector::zeros(3);
        let cfg = QpConfig {
            solver: QpSolver::Clarabel,
            feasibility_tol: 1e-9,
            ..Default::default()
        };
        let sol = solve_qp(&h, &c, None, None, Some(&a), Some(&b), None, &cfg);
        assert!(check(&h, &c, None, Some((&a, &b)), &sol).accepted(1e-6));
    }
    #[test]
    fn impossible_constraints_are_not_accepted() {
        let h = DMatrix::identity(1, 1);
        let c = DVector::zeros(1);
        let a = DMatrix::from_row_slice(2, 1, &[1., -1.]);
        let b = DVector::from_vec(vec![0., -1.]);
        let sol = solve_qp(
            &h,
            &c,
            None,
            None,
            Some(&a),
            Some(&b),
            None,
            &QpConfig::default(),
        );
        assert!(!check(&h, &c, None, Some((&a, &b)), &sol).accepted(1e-6));
    }
}
