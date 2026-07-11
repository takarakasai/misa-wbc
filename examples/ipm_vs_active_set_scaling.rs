//! Interior-point vs active-set: how each scales with problem size
//! (run with `--release`).
//!
//! Both are dense, from-scratch, self-contained implementations in
//! this crate (`qp.rs`) — the comparison isolates the *algorithmic*
//! difference (barrier path-following vs simplex-style vertex hopping)
//! from any warm-start or implementation-maturity effect (Clarabel is
//! excluded here on purpose; see `qp_warm_bench` for that comparison).
//!
//! For a sweep of `(n, m_iq)` — decision variables × inequality rows,
//! at a fixed ratio `m_iq ≈ 1.1·n` typical of a WBC tick — solves a
//! fresh random QP (cold start, no workspace) with each backend and
//! reports median solve time and iteration count.
//!
//! Textbook expectation: **IPM iteration count grows slowly with size**
//! (theoretically O(√m_iq) worst-case per Newton/barrier-reduction
//! phase; empirically often near-constant for well-conditioned random
//! QPs), because every iteration considers *all* constraints via one
//! Newton step. **Active-set iteration count tends to grow with the
//! number of constraints that must change status** between the start
//! and the optimum — for a cold start on a random problem, that is
//! often close to the number of active constraints at the solution.
//! This is the standard qualitative case for "IPM scales better,
//! active-set wins when warm-started" — the numbers below quantify it
//! for these two implementations specifically.

use std::time::Instant;

use misa_wbc::qp::{solve_qp, QpConfig, QpSolver, QpStatus};
use nalgebra::{DMatrix, DVector};

/// Deterministic LCG so the sweep is reproducible without a `rand` dep.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A well-conditioned random inequality-constrained QP at size `n`,
/// with `m_iq ≈ 1.1·n` rows (loosely feasible so the active set at the
/// optimum is a genuine subset, not everything or nothing).
fn random_qp(rng: &mut Lcg, n: usize) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
    let m_iq = (n * 11) / 10;
    let a = DMatrix::from_fn(n, n, |_, _| rng.next_f64());
    let h = &a * a.transpose() + DMatrix::identity(n, n);
    let c = DVector::from_fn(n, |_, _| rng.next_f64());
    let d = DMatrix::from_fn(m_iq, n, |_, _| rng.next_f64());
    let f = DVector::from_fn(m_iq, |_, _| rng.next_f64().abs() * 0.5 + 0.1);
    (h, c, d, f)
}

fn main() {
    const SIZES: [usize; 6] = [5, 10, 20, 40, 80, 160];
    const REPEATS: usize = 30;

    println!("IPM vs ActiveSet scaling (cold start, dense, no warm-start on either side).\n");
    println!(
        "| {:>5} | {:>6} | {:>14} | {:>14} | {:>14} | {:>14} |",
        "n", "m_iq", "IPM med [ms]", "IPM iters", "AS med [ms]", "AS iters"
    );
    println!("|{}|", "-".repeat(84));

    let ipm_cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
    let as_cfg = QpConfig { solver: QpSolver::ActiveSet, ..Default::default() };

    for &n in &SIZES {
        let mut rng = Lcg(0xC0FFEE_u64.wrapping_add(n as u64));
        let mut ipm_times = Vec::with_capacity(REPEATS);
        let mut as_times = Vec::with_capacity(REPEATS);
        let mut ipm_iters = Vec::with_capacity(REPEATS);
        let mut as_iters = Vec::with_capacity(REPEATS);
        let mut m_iq = 0;

        for _ in 0..REPEATS {
            let (h, c, d, f) = random_qp(&mut rng, n);
            m_iq = d.nrows();

            let t0 = Instant::now();
            let sol_ipm = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &ipm_cfg);
            ipm_times.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(sol_ipm.status, QpStatus::Optimal, "IPM failed at n={n}");
            ipm_iters.push(sol_ipm.iterations);

            let t0 = Instant::now();
            let sol_as = solve_qp(&h, &c, None, None, Some(&d), Some(&f), None, &as_cfg);
            as_times.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(sol_as.status, QpStatus::Optimal, "ActiveSet failed at n={n}");
            as_iters.push(sol_as.iterations);
        }

        ipm_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        as_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ipm_iters.sort_unstable();
        as_iters.sort_unstable();

        println!(
            "| {:>5} | {:>6} | {:>14.4} | {:>14} | {:>14.4} | {:>14} |",
            n,
            m_iq,
            ipm_times[REPEATS / 2],
            ipm_iters[REPEATS / 2],
            as_times[REPEATS / 2],
            as_iters[REPEATS / 2],
        );
    }

    println!(
        "\nmed = median over {REPEATS} independent random QPs per size. \
         iters = active-set / interior-point iteration count (not wall \
         time) — the size-scaling signal free of per-iteration cost \
         differences."
    );
}
