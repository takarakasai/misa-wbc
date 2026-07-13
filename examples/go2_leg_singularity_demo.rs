//! Singularity-approach stability comparison, Go2-leg edition: the same
//! study as `panda_singularity_demo.rs`, run against a real quadruped
//! leg instead of an arm, to connect the Panda benchmark back to
//! misa-wbc's actual target (quadruped-gait's Go2 WBC).
//!
//! Loads `go2.misa` (fixed-base as-is -- no floating joint, so this is
//! effectively "one Go2 mounted on a test stand", exactly like the
//! Panda demo's "arm on a table") and drives the front-right foot
//! (`FR_foot_fixed`) along a fixed radial line from the hip, out to
//! near the leg's full straight-line reach (thigh 0.213 m + calf
//! 0.213 m =~ 0.426 m) -- the classical 3-DOF leg singularity (fully
//! extended knee), not the Panda's redundant-arm reach boundary.
//!
//! Unlike the Panda (7 DOF driving a 6-D pose task), a leg has only 3
//! actuated DOF (hip/thigh/calf) -- nowhere near enough for a 6-D pose
//! task, so this demo tracks **position only** (the linear 3 rows of
//! the foot Jacobian), giving a genuinely *square* 3x3 task Jacobian at
//! the singularity rather than Panda's overdetermined 6x9. The other
//! three legs are along for the ride (part of the same qddot/tau
//! vector) but untasked, held near rest by the priority-2 regularizer.
//!
//! ```text
//! cargo run --release --example go2_leg_singularity_demo -- <formulation> <backend> [damped]
//! (same argument semantics as panda_singularity_demo.rs)
//! ```
//!
//! CSV layout: `tick,t,q0..11,ee_xyz,ref_xyz,link_poses...,` then the
//! same six trailing diagnostic columns as the Panda demo
//! (`sigma_min,tau_norm,tau_max_abs,qddot_norm,status_code,degraded_level`).

use misarta::fk::forward_kinematics;
use misarta::jacobian::{compute_jacobian_dot_times_v, compute_joint_jacobian};
use misarta::se3;

use misa_wbc::{refgen, tasks, Dynamics, Formulation, QpSolver, SolveConfig, Solver};
use nalgebra::{DMatrix, DVector, Vector3};

const DT: f64 = 0.004; // 250 Hz
const DURATION_S: f64 = 5.0;
const REACH_CENTER: f64 = 0.395;
const REACH_AMP: f64 = 0.06; // sweeps reach in [0.335, 0.455] m from the hip -- 0.426 m is full straight-leg reach; the far target is deliberately unreachable so the CBF (D6) has to actually brake against it
const PERIOD_S: f64 = 2.5; // two full in/out cycles over DURATION_S

fn parse_formulation(s: &str) -> Formulation {
    match s {
        "explicit" => Formulation::Explicit,
        "accel" => Formulation::AccelSpace,
        "force" => Formulation::ForceSpace,
        other => panic!("unknown formulation '{other}' (expected explicit|accel|force)"),
    }
}

fn parse_backend(s: &str) -> QpSolver {
    match s {
        "activeset" => QpSolver::ActiveSet,
        "ipm" => QpSolver::Ipm,
        "admm" => QpSolver::Admm,
        "clarabel" => QpSolver::Clarabel,
        other => panic!("unknown backend '{other}' (expected activeset|ipm|admm|clarabel)"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let formulation = parse_formulation(args.get(1).map(|s| s.as_str()).unwrap_or("explicit"));
    let backend = parse_backend(args.get(2).map(|s| s.as_str()).unwrap_or("activeset"));
    let damped = args.get(3).map(|s| s.as_str()) == Some("damped");
    eprintln!("formulation={:?} backend={:?} damped={damped}", formulation, backend);

    let path = format!(
        "{}/../articara/models/unitree_go2/go2.misa",
        env!("CARGO_MANIFEST_DIR")
    );
    let parsed = misarta::native::load(std::path::Path::new(&path)).expect("load go2.misa");
    let (model, vis, _col) = misarta::native::build_model(&parsed.file).expect("build go2 model");
    let (nv, nq) = (model.nv, model.nq);
    assert_eq!(nq, nv, "fixed-base quadruped (no floating joint) expected");
    let na = nv;

    // Mesh manifest (written once): unlike the Panda demo, go2.misa
    // already carries real visual meshes (unitree's own .obj files),
    // each with a non-trivial per-mesh placement (some visuals are
    // rotated -- see e.g. FR_hip's meshes, rpy=[pi,0,0]) relative to
    // their parent joint. Dump parent joint index + resolved mesh path
    // + placement (translation, row-major rotation) so the renderer
    // can compose world_T_mesh = oMi[parent_joint] * placement without
    // re-deriving any of this from the .misa TOML itself.
    {
        let mesh_dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        let manifest_path = format!("{}/examples/models/go2_mesh_manifest.csv", env!("CARGO_MANIFEST_DIR"));
        let mut manifest = String::from("parent_joint,mesh_path,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n");
        for obj in &vis.objects {
            if let Some(mesh_path) = &obj.mesh_path {
                let resolved = mesh_dir.join(mesh_path);
                let t = se3::translation(&obj.placement);
                let r = se3::rotation_matrix(&obj.placement);
                manifest.push_str(&format!(
                    "{},{},{:.6},{:.6},{:.6}",
                    obj.parent_joint,
                    resolved.display(),
                    t.x, t.y, t.z
                ));
                for row in 0..3 {
                    for col in 0..3 {
                        manifest.push_str(&format!(",{:.6}", r[(row, col)]));
                    }
                }
                manifest.push('\n');
            }
        }
        std::fs::write(&manifest_path, manifest).expect("write mesh manifest");
        eprintln!("wrote mesh manifest to {manifest_path}");
    }

    let ee_idx = model
        .joints
        .iter()
        .position(|j| j.name == "FR_foot_fixed")
        .expect("FR_foot_fixed in model");
    let hip_idx = model
        .joints
        .iter()
        .position(|j| j.name == "FR_hip_joint")
        .expect("FR_hip_joint in model");

    // Resting pose: a generic half-crouched stance, all four legs
    // (only FR is tasked; the rest just sit here under regularization).
    let mut q = vec![0.0_f64; nq];
    let mut v = vec![0.0_f64; nv];
    for (i, j) in model.joints.iter().enumerate() {
        if i == 0 {
            continue; // universe/base root, no q entry
        }
        if j.name.ends_with("_thigh_joint") {
            q[i - 1] = 0.9;
        } else if j.name.ends_with("_calf_joint") {
            q[i - 1] = -1.8;
        }
    }

    // Per-joint torque limits (go2.misa: hip/thigh 23.7 Nm, calf 45.43 Nm).
    let mut tau_max = DVector::from_element(na, 23.7);
    for (i, j) in model.joints.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if j.name.ends_with("_calf_joint") {
            tau_max[i - 1] = 45.43;
        }
    }

    // Per-joint position limits (go2.misa): without these, the leg has
    // nothing stopping it from "escaping" the singularity by bending
    // the knee back past its real mechanical range to reach the same
    // Cartesian target through an unphysical, well-conditioned
    // configuration instead of the true one -- measured directly while
    // tuning this demo (calf reached -0.48 rad against a real limit of
    // -0.84). Wired as tasks::joint_limit_cbf (D6) at priority 0, the
    // same primitive validated in tests/joint_limit_cbf_stack.rs.
    let mut q_min = DVector::from_element(na, -1.0472); // hip default
    let mut q_max = DVector::from_element(na, 1.0472);
    for (i, j) in model.joints.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if j.name.ends_with("_thigh_joint") {
            q_min[i - 1] = -1.5708;
            q_max[i - 1] = 3.4907;
        } else if j.name.ends_with("_calf_joint") {
            q_min[i - 1] = -2.7227;
            q_max[i - 1] = -0.83776;
        }
    }
    let joint_limits = tasks::JointLimitCbf {
        q_min,
        q_max,
        v_max: DVector::from_element(na, 20.0), // not in go2.misa (unset sentinel) -- generous, non-binding default
        a_max: DVector::from_element(na, 100.0),
        alpha1: DVector::from_element(na, 8.0),
        alpha2: DVector::from_element(na, 8.0),
        alpha3: DVector::from_element(na, 8.0),
    };

    let kp = DVector::from_element(3, 400.0);
    let kd = DVector::from_element(3, 40.0);

    // Fixed radial direction from the hip (FR_hip_joint's origin -- a
    // pure abduction rotation about X there, so its own translation is
    // q-independent) through the resting foot position.
    let data0 = forward_kinematics(&model, &q);
    let hip_origin = se3::translation(&data0.oMi[hip_idx]);
    let ee0 = se3::translation(&data0.oMi[ee_idx]);
    let dir = (ee0 - hip_origin).normalize();

    let cfg = SolveConfig { backend, ..SolveConfig::default() };
    let mut solver = Solver::new();

    {
        let topo_path = format!("{}/examples/models/go2_topology.csv", env!("CARGO_MANIFEST_DIR"));
        let mut topo = String::from("idx,name,parent\n");
        for (i, j) in model.joints.iter().enumerate().skip(1) {
            topo.push_str(&format!("{i},{},{}\n", j.name, j.parent));
        }
        std::fs::write(&topo_path, topo).expect("write topology sidecar");
        eprintln!("wrote topology to {topo_path}");
    }

    let n_ticks = (DURATION_S / DT) as usize;
    for tick in 0..n_ticks {
        let t = tick as f64 * DT;
        let omega = 2.0 * std::f64::consts::PI / PERIOD_S;
        let phase = omega * t;

        let reach = REACH_CENTER + REACH_AMP * phase.cos();
        let reach_dot = -REACH_AMP * omega * phase.sin();
        let x_ref = hip_origin + reach * dir;
        let v_ref = reach_dot * dir;

        let mass = misarta::crba::crba(&model, &q);
        let h = misarta::rnea::nonlinear_effects(&model, &q, &v);
        let j_foot6 = compute_joint_jacobian(&model, &q, ee_idx);
        let dj_v6 = compute_jacobian_dot_times_v(&model, &q, &v, ee_idx);
        // Position-only: a 3-DOF leg has no business being handed a
        // 6-D pose task (see module docs) -- slice to the linear rows,
        // giving a genuinely square 3x3 task Jacobian at the singularity.
        let j_foot = j_foot6.rows(3, 3).into_owned();
        let dj_v = DVector::from_column_slice(dj_v6.rows(3, 3).into_owned().as_slice());

        let svd = nalgebra::linalg::SVD::new(j_foot.clone(), false, false);
        let sigma_min = svd.singular_values.iter().cloned().fold(f64::INFINITY, f64::min);

        let data = forward_kinematics(&model, &q);
        let ee_pos = se3::translation(&data.oMi[ee_idx]);
        let ee_lin_vel = {
            let jv = &j_foot * DVector::from_column_slice(&v);
            Vector3::new(jv[0], jv[1], jv[2])
        };

        let a_ref = refgen::pd(
            &DVector::from_column_slice(x_ref.as_slice()),
            &DVector::from_column_slice(ee_pos.as_slice()),
            &DVector::from_column_slice(v_ref.as_slice()),
            &DVector::from_column_slice(ee_lin_vel.as_slice()),
            &kp,
            &kd,
        );

        let d = Dynamics::new(formulation, &mass, &h, &DMatrix::zeros(0, nv), na);
        let mut p0 = tasks::box_bound(d.tau(), &tau_max)
            + tasks::joint_limit_cbf(
                d.qddot(),
                &DVector::from_column_slice(&q),
                &DVector::from_column_slice(&v),
                &joint_limits,
            );
        if let Some(phys) = d.dynamics_task() {
            p0 = phys + p0;
        }
        let p1 = if damped {
            let mut dcfg = tasks::SingularityDamping::default();
            if let Ok(l) = std::env::var("LAMBDA_MAX_SQ") {
                dcfg.lambda_max_sq = l.parse().expect("LAMBDA_MAX_SQ must be a float");
            }
            tasks::cartesian_acceleration_damped(d.qddot(), &j_foot, &dj_v, &a_ref, &dcfg)
        } else {
            tasks::cartesian_acceleration(d.qddot(), &j_foot, &dj_v, &a_ref)
        };
        let p2 = tasks::track(d.qddot(), &DVector::zeros(nv))
            + tasks::track(d.tau(), &DVector::zeros(na)).weight(0.01);

        let sol = solver.solve(&[p0, p1, p2], &cfg).expect("solve");
        let e = d.extract(&sol.x);

        let (status_code, degraded_level) = match &sol.status {
            misa_wbc::SolveStatus::Optimal => (0, -1i64),
            misa_wbc::SolveStatus::Degraded { level, .. } => (1, *level as i64),
        };
        let tau_norm = e.tau.norm();
        let tau_max_abs = e.tau.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let qddot_norm = e.qddot.norm();

        for i in 0..nv {
            v[i] += e.qddot[i] * DT;
            q[i] += v[i] * DT;
        }

        if q.iter().chain(v.iter()).any(|x| !x.is_finite() || x.abs() > 1e3) {
            eprintln!("STATE DIVERGED at tick {tick} (t={t:.4}s) -- stopping early, trace truncated here");
            break;
        }

        print!("{tick},{t:.4}");
        for &qi in &q {
            print!(",{qi:.5}");
        }
        print!(",{:.5},{:.5},{:.5}", ee_pos.x, ee_pos.y, ee_pos.z);
        print!(",{:.5},{:.5},{:.5}", x_ref.x, x_ref.y, x_ref.z);
        for j in 0..model.joints.len() {
            let p = se3::translation(&data.oMi[j]);
            let r = se3::rotation_matrix(&data.oMi[j]);
            print!(",{:.5},{:.5},{:.5}", p.x, p.y, p.z);
            for row in 0..3 {
                for col in 0..3 {
                    print!(",{:.6}", r[(row, col)]);
                }
            }
        }
        print!(",{sigma_min:.6},{tau_norm:.5},{tau_max_abs:.5},{qddot_norm:.5},{status_code},{degraded_level}");
        println!();
    }
}
