//! Integration check: a small WBC problem assembled two ways — with the
//! `affine` vocabulary and by hand-indexing the decision vector — must
//! produce byte-identical `Task` matrices and the same HoQp solution.
//!
//! This pins the Phase-1 promise: writing tasks symbolically over
//! `VarLayout` variables is a zero-overhead reformulation of the raw
//! matrix assembly the quadruped WBC uses today.


// Under --no-default-features only the non-Clarabel tests remain;
// the shared fixtures/imports above them would trip -D warnings.
#![cfg_attr(not(feature = "clarabel"), allow(unused))]
use misa_wbc::{solve, SolveConfig, Task, VarLayout};
use nalgebra::{DMatrix, DVector};

/// Toy layout: x = [ qddot(2) ; f(2) ]  (n = 4).
/// - EoM (equality, priority 0):  A_eom·qddot + B_eom·f = c_eom
/// - friction-ish (inequality):   f ≤ f_max
/// - track (soft equality, prio 1): qddot ≈ qddot_ref
fn problem() -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>, DVector<f64>) {
    let a_eom = DMatrix::from_row_slice(2, 2, &[1.0, 0.2, 0.0, 1.0]);
    let b_eom = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
    let c_eom = DVector::from_vec(vec![0.3, -0.4]);
    let f_max = DVector::from_vec(vec![5.0, 5.0]);
    let qddot_ref = DVector::from_vec(vec![0.1, -0.2]);
    (a_eom, b_eom, c_eom, f_max, qddot_ref)
}

fn assemble_by_hand() -> Vec<Task> {
    let (a_eom, b_eom, c_eom, f_max, qddot_ref) = problem();
    let n = 4;
    // priority 0: EoM equality [A_eom | B_eom]·x = c_eom  +  f ≤ f_max
    let mut eom_a = DMatrix::zeros(2, n);
    eom_a.view_mut((0, 0), (2, 2)).copy_from(&a_eom);
    eom_a.view_mut((0, 2), (2, 2)).copy_from(&b_eom);
    let eom = Task::equality(eom_a, c_eom);

    // f ≤ f_max :  [0 0 | I]·x ≤ f_max
    let mut fmax_d = DMatrix::zeros(2, n);
    fmax_d.view_mut((0, 2), (2, 2)).copy_from(&DMatrix::<f64>::identity(2, 2));
    let fmax = Task::inequality(fmax_d, f_max);

    // priority 1: qddot ≈ qddot_ref :  [I | 0]·x = qddot_ref
    let mut track_a = DMatrix::zeros(2, n);
    track_a.view_mut((0, 0), (2, 2)).copy_from(&DMatrix::<f64>::identity(2, 2));
    let track = Task::equality(track_a, qddot_ref);

    vec![eom + fmax, track]
}

fn assemble_with_affine() -> Vec<Task> {
    let (a_eom, b_eom, c_eom, f_max, qddot_ref) = problem();
    let vars = VarLayout::builder().add("qddot", 2).add("f", 2).build();
    let qddot = vars.var("qddot");
    let f = vars.var("f");

    // EoM residual  e = A_eom·qddot + B_eom·f − c_eom  → 0
    let e = &(&(&a_eom * &qddot) + &(&b_eom * &f)) - &c_eom;
    let eom = Task::soft_eq(&e);

    // f ≤ f_max
    let fmax = Task::le(&f.affine(), &f_max);

    // qddot ≈ qddot_ref  →  residual (qddot − qddot_ref)
    let track_res = &qddot.affine() - &qddot_ref;
    let track = Task::soft_eq(&track_res);

    vec![eom + fmax, track]
}

#[test]
fn affine_and_hand_assembly_are_identical() {
    let hand = assemble_by_hand();
    let aff = assemble_with_affine();
    assert_eq!(hand.len(), aff.len());
    for (h, a) in hand.iter().zip(aff.iter()) {
        assert!((h.a.clone() - a.a.clone()).norm() < 1e-12, "A differs");
        assert!((h.b.clone() - a.b.clone()).norm() < 1e-12, "b differs");
        assert!((h.d.clone() - a.d.clone()).norm() < 1e-12, "D differs");
        assert!((h.f.clone() - a.f.clone()).norm() < 1e-12, "f differs");
    }
}

#[cfg(feature = "clarabel")]
#[test]
fn affine_and_hand_assembly_solve_identically() {
    let cfg = SolveConfig::default();
    let x_hand = solve(&assemble_by_hand(), &cfg).unwrap().x;
    let x_aff = solve(&assemble_with_affine(), &cfg).unwrap().x;
    assert!(
        (&x_hand - &x_aff).norm() < 1e-9,
        "solutions differ: {:?} vs {:?}",
        x_hand,
        x_aff,
    );
    // Sanity: the EoM equality is satisfied and f respects its cap.
    let (a_eom, b_eom, c_eom, f_max, _) = problem();
    let qddot = x_aff.rows(0, 2).into_owned();
    let f = x_aff.rows(2, 2).into_owned();
    assert!((&a_eom * &qddot + &b_eom * &f - &c_eom).norm() < 1e-6, "EoM violated");
    assert!(f[0] <= f_max[0] + 1e-6 && f[1] <= f_max[1] + 1e-6, "f cap violated");
}
