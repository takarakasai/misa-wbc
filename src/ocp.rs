//! Uncondensed finite-horizon OCP with costs 0.5 x'Qx + 0.5 u'Ru + u'Sx + q'x + r'u.
use nalgebra::{DMatrix, DVector};
#[derive(Clone, Debug)]
pub struct OcpStage {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub bias: DVector<f64>,
    pub q_matrix: DMatrix<f64>,
    pub r_matrix: DMatrix<f64>,
    pub s_matrix: DMatrix<f64>,
    pub q: DVector<f64>,
    pub r: DVector<f64>,
    /// General constraints lower <= C x + D u <= upper.
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub lower: DVector<f64>,
    pub upper: DVector<f64>,
    pub constant: f64,
}
#[derive(Clone, Debug)]
pub struct OcpProblem {
    pub initial: DVector<f64>,
    pub stages: Vec<OcpStage>,
    // Terminal cost is zero: physical terminal cost is folded into the last stage.
}

/// Solve using HPIPM's uncondensed OCP interface. Returns controls stacked by
/// stage, and upper-minus-lower multipliers in stage general-constraint order.
/// The current interface performs a cold solve (no factorization reuse).
#[cfg(feature = "hpipm")]
pub fn solve_ocp(problem: &OcpProblem, config: &crate::qp::QpConfig) -> crate::qp::QpSolution {
    use crate::qp::{QpDiagnostics, QpSolution, QpStatus};
    #[repr(C)]
    struct NativeStage {
        a: *const f64,
        b: *const f64,
        bias: *const f64,
        qmat: *const f64,
        rmat: *const f64,
        smat: *const f64,
        q: *const f64,
        r: *const f64,
        c: *const f64,
        d: *const f64,
        lower: *const f64,
        upper: *const f64,
        ng: i32,
    }
    unsafe extern "C" {
        fn misa_native_ocp(
            n: i32,
            nx: i32,
            nu: i32,
            initial: *const f64,
            stages: *const NativeStage,
            max_iter: i32,
            ftol: f64,
            otol: f64,
            controls: *mut f64,
            multipliers: *mut f64,
            iterations: *mut i32,
            raw: *mut i32,
            times: *mut u64,
        ) -> i32;
    }
    assert!(!problem.stages.is_empty());
    let nx = problem.initial.len();
    let nu = problem.stages[0].r.len();
    let n = problem.stages.len();
    let finite = problem.initial.iter().all(|v| v.is_finite())
        && problem.stages.iter().all(|s| {
            s.a.iter()
                .chain(s.b.iter())
                .chain(s.bias.iter())
                .chain(s.q_matrix.iter())
                .chain(s.r_matrix.iter())
                .chain(s.s_matrix.iter())
                .chain(s.q.iter())
                .chain(s.r.iter())
                .chain(s.c.iter())
                .chain(s.d.iter())
                .all(|v| v.is_finite())
                && s.lower.iter().chain(s.upper.iter()).all(|v| !v.is_nan())
        });
    if !finite
        || nx == 0
        || nu == 0
        || config.max_iters == 0
        || !config.feasibility_tol.is_finite()
        || config.feasibility_tol <= 0.
        || !config.optimality_tol.is_finite()
        || config.optimality_tol <= 0.
    {
        return QpSolution {
            x: DVector::zeros(n * nu),
            objective: f64::NAN,
            lambda_eq: DVector::zeros(0),
            lambda_iq: DVector::zeros(problem.stages.iter().map(|s| s.upper.len()).sum()),
            status: QpStatus::NumericalFailure,
            iterations: 0,
            diagnostics: Some(QpDiagnostics {
                raw_status: "InvalidInput".into(),
                ..Default::default()
            }),
        };
    }

    let start = std::time::Instant::now();
    let stages: Vec<_> = problem
        .stages
        .iter()
        .map(|s| {
            let ng = s.upper.len();
            assert_eq!(s.a.shape(), (nx, nx));
            assert_eq!(s.b.shape(), (nx, nu));
            assert_eq!(s.bias.len(), nx);
            assert_eq!(s.q_matrix.shape(), (nx, nx));
            assert_eq!(s.r_matrix.shape(), (nu, nu));
            assert_eq!(s.s_matrix.shape(), (nu, nx));
            assert_eq!(s.q.len(), nx);
            assert_eq!(s.r.len(), nu);
            assert_eq!(s.c.shape(), (ng, nx));
            assert_eq!(s.d.shape(), (ng, nu));
            assert_eq!(s.lower.len(), ng);
            NativeStage {
                a: s.a.as_ptr(),
                b: s.b.as_ptr(),
                bias: s.bias.as_ptr(),
                qmat: s.q_matrix.as_ptr(),
                rmat: s.r_matrix.as_ptr(),
                smat: s.s_matrix.as_ptr(),
                q: s.q.as_ptr(),
                r: s.r.as_ptr(),
                c: s.c.as_ptr(),
                d: s.d.as_ptr(),
                lower: s.lower.as_ptr(),
                upper: s.upper.as_ptr(),
                ng: ng.try_into().unwrap(),
            }
        })
        .collect();
    let mut x = DVector::zeros(n * nu);
    let mut lambda = DVector::zeros(problem.stages.iter().map(|s| s.upper.len()).sum());
    let mut iterations = 0;
    let mut raw = 0;
    let mut times = [0u64; 4];
    let conversion_ns = start.elapsed().as_nanos() as u64;
    let code = unsafe {
        misa_native_ocp(
            n.try_into().unwrap(),
            nx.try_into().unwrap(),
            nu.try_into().unwrap(),
            problem.initial.as_ptr(),
            stages.as_ptr(),
            config.max_iters.try_into().unwrap(),
            config.feasibility_tol,
            config.optimality_tol,
            x.as_mut_ptr(),
            lambda.as_mut_ptr(),
            &mut iterations,
            &mut raw,
            times.as_mut_ptr(),
        )
    };
    let mut state = problem.initial.clone();
    let mut objective = 0.;
    for (k, s) in problem.stages.iter().enumerate() {
        let u = x.rows(k * nu, nu).into_owned();
        objective += 0.5 * state.dot(&(&s.q_matrix * &state))
            + 0.5 * u.dot(&(&s.r_matrix * &u))
            + u.dot(&(&s.s_matrix * &state))
            + s.q.dot(&state)
            + s.r.dot(&u)
            + s.constant;
        state = &s.a * state + &s.b * u + &s.bias;
    }
    QpSolution {
        x,
        objective,
        lambda_eq: DVector::zeros(0),
        lambda_iq: lambda,
        status: match code {
            0 => QpStatus::Optimal,
            1 => QpStatus::MaxIterations,
            _ => QpStatus::NumericalFailure,
        },
        iterations: iterations.max(0) as usize,
        diagnostics: Some(QpDiagnostics {
            raw_status: format!("HpipmOcp:{raw}"),
            conversion_ns,
            setup_ns: times[1],
            update_ns: 0,
            solve_ns: times[3],
            warm_start_used: false,
        }),
    }
}
