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
//!
//! **Does the iteration-count advantage ever flip the wall-clock
//! ranking?** Extended to n=5120 (m_iq≈5632) looking for exactly that
//! crossover (`ref/wbc_comparison.md` §5f had flagged it as an open
//! question). Answer, measured: the AS/IPM time ratio climbs steadily
//! — 0.22 at n=160, 0.54 at n=1280, 0.66 at n=5120 — but the climb is
//! *decelerating* (Δ +0.19, +0.07, +0.05 per doubling), consistent
//! with an asymptote below 1.0 rather than an eventual crossover. Both
//! backends are, in this dense implementation, ultimately O(n³) work
//! (IPM: near-constant iterations × an O(n³) refactorisation each;
//! active-set: near-linear iterations × an O(n²) incremental update
//! each ≈ O(n³) total) — active-set's smaller constant appears to win
//! at every size tested, not just small ones. Active-set's iteration
//! count also needs `max_iters` to scale with `n` past a few hundred
//! (it is genuinely doing more combinatorial work, not degrading) —
//! `as_max_iters_for` below sizes it at `3n`.

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

/// Fewer repeats at larger `n` — the per-solve cost grows as O(n³) for
/// both backends (dense Cholesky / dense Schur complement), so a fixed
/// repeat count would make the sweep take minutes at n=1280+.
fn repeats_for(n: usize) -> usize {
    match n {
        0..=160 => 30,
        161..=320 => 15,
        321..=640 => 6,
        641..=1280 => 3,
        1281..=2560 => 2,
        _ => 1,
    }
}

/// Active-set's iteration count grows roughly linearly with `n` (see
/// the module docs), so the crate default `max_iters = 500` is
/// eventually too small — it must scale with `n`, not stay fixed,
/// or the sweep just measures "does it hit the cap" instead of "how
/// long does it take to converge".
fn as_max_iters_for(n: usize) -> usize {
    (n * 3).max(500)
}

fn main() {
    // Extended past the original 5..160 sweep to look for the crossover
    // point where IPM's near-flat iteration count should eventually
    // outrun active-set's near-linear growth in WALL TIME, not just
    // iteration count — see ref/wbc_comparison.md §5f for the original
    // (unresolved) observation that "the iteration-count advantage
    // would need to compound over a much larger n to flip the
    // wall-clock ranking."
    const SIZES: [usize; 11] = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];

    println!("IPM vs ActiveSet scaling (cold start, dense, no warm-start on either side).\n");
    println!(
        "| {:>5} | {:>6} | {:>7} | {:>14} | {:>14} | {:>14} | {:>14} | {:>10} |",
        "n", "m_iq", "repeats", "IPM med [ms]", "IPM iters", "AS med [ms]", "AS iters", "AS/IPM"
    );
    println!("|{}|", "-".repeat(103));

    for &n in &SIZES {
        let ipm_cfg =
            QpConfig { solver: QpSolver::Ipm, max_iters: as_max_iters_for(n), ..Default::default() };
        let as_cfg = QpConfig {
            solver: QpSolver::ActiveSet,
            max_iters: as_max_iters_for(n),
            ..Default::default()
        };
        let repeats = repeats_for(n);
        let mut rng = Lcg(0xC0FFEE_u64.wrapping_add(n as u64));
        let mut ipm_times = Vec::with_capacity(repeats);
        let mut as_times = Vec::with_capacity(repeats);
        let mut ipm_iters = Vec::with_capacity(repeats);
        let mut as_iters = Vec::with_capacity(repeats);
        let mut m_iq = 0;

        for _ in 0..repeats {
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

        let ipm_med = ipm_times[repeats / 2];
        let as_med = as_times[repeats / 2];

        println!(
            "| {:>5} | {:>6} | {:>7} | {:>14.4} | {:>14} | {:>14.4} | {:>14} | {:>9.2}x |",
            n,
            m_iq,
            repeats,
            ipm_med,
            ipm_iters[repeats / 2],
            as_med,
            as_iters[repeats / 2],
            as_med / ipm_med,
        );
    }

    println!(
        "\nmed = median over the size-dependent repeat count above. \
         iters = active-set / interior-point iteration count (not wall \
         time) — the size-scaling signal free of per-iteration cost \
         differences. AS/IPM > 1 means IPM is faster in wall time \
         (the crossover this sweep is looking for)."
    );
}
