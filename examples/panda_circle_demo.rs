//! Closed-loop WBC simulation: the Panda's end-effector traces a
//! circle under `Dynamics` + `tasks::cartesian_acceleration` +
//! `refgen::pd`, solved every tick with `Solver` (warm-started
//! ActiveSet — the crate's recommended real-time backend). Writes a
//! CSV trace (joint angles + every link's world position) for
//! external rendering into a video.
//!
//! Not a benchmark (see `panda_bench.rs` for that) — this is the
//! "does the whole stack actually make the robot do something
//! sensible" demo: `cargo run --release --example panda_circle_demo`.

use misarta::fk::forward_kinematics;
use misarta::jacobian::{compute_jacobian_dot_times_v, compute_joint_jacobian};
use misarta::se3;

use misa_wbc::{refgen, tasks, Dynamics, Formulation, SolveConfig, Solver};
use nalgebra::{DMatrix, DVector, Vector3};

const DT: f64 = 0.004; // 250 Hz
const DURATION_S: f64 = 6.0;
const RADIUS: f64 = 0.18;
const PERIOD_S: f64 = 3.0; // one full circle every 3s

fn main() {
    let path = format!("{}/examples/models/panda.urdf", env!("CARGO_MANIFEST_DIR"));
    let imported = misarta_formats::urdf::import(std::path::Path::new(&path)).expect("import panda.urdf");
    let (model, _vis, _col) = misarta::native::build_model(&imported.file).expect("build panda model");
    let (nv, nq) = (model.nv, model.nq);
    assert_eq!(nq, nv, "fixed-base arm expected");
    let na = nv;

    let ee_idx = model
        .joints
        .iter()
        .position(|j| j.name == "panda_joint7")
        .expect("panda_joint7 in model");

    // Ready pose, at rest.
    let mut q = vec![0.0_f64; nq];
    for (i, &qi) in [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8].iter().enumerate() {
        q[i] = qi;
    }
    let mut v = vec![0.0_f64; nv];

    let mut tau_max = DVector::from_element(na, 20.0);
    for (i, &t) in [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0].iter().enumerate() {
        tau_max[i] = t;
    }
    let kp = DVector::from_element(6, 400.0);
    let kd = DVector::from_element(6, 40.0);

    // Circle centre = the EE's starting position, in the plane
    // perpendicular to the initial approach direction (world XZ, a
    // vertical circle facing the viewer).
    let q0: Vec<f64> = q.clone();
    let data0 = forward_kinematics(&model, &q0);
    let centre = se3::translation(&data0.oMi[ee_idx]);

    let cfg = SolveConfig::default(); // ActiveSet + NullSpace
    let mut solver = Solver::new();

    // Topology sidecar (written once): joint index, name, parent index
    // — so the plotting script can draw the actual kinematic tree
    // instead of assuming a serial chain.
    {
        let topo_path = format!("{}/examples/models/panda_topology.csv", env!("CARGO_MANIFEST_DIR"));
        let mut topo = String::from("idx,name,parent\n");
        for (i, j) in model.joints.iter().enumerate().skip(1) {
            topo.push_str(&format!("{i},{},{}\n", j.name, j.parent));
        }
        std::fs::write(&topo_path, topo).expect("write topology sidecar");
        eprintln!("wrote topology to {topo_path}");
    }

    println!("tick,t,q0,q1,q2,q3,q4,q5,q6,ee_x,ee_y,ee_z,ref_x,ref_y,ref_z,link_xyz...");
    // Header row above is documentation; actual CSV has no header line
    // (link count varies only by model, kept simple for the plotting script).
    let n_ticks = (DURATION_S / DT) as usize;
    for tick in 0..n_ticks {
        let t = tick as f64 * DT;
        let omega = 2.0 * std::f64::consts::PI / PERIOD_S;
        let phase = omega * t;

        // Reference: a vertical circle in the world X-Z plane, centred
        // on the starting EE position.
        let x_ref = Vector3::new(
            centre.x,
            centre.y + RADIUS * phase.cos() - RADIUS,
            centre.z + RADIUS * phase.sin(),
        );
        let v_ref = Vector3::new(0.0, -RADIUS * omega * phase.sin(), RADIUS * omega * phase.cos());

        let mass = misarta::crba::crba(&model, &q);
        let h = misarta::rnea::nonlinear_effects(&model, &q, &v);
        let j_ee = compute_joint_jacobian(&model, &q, ee_idx);
        let dj_v6 = compute_jacobian_dot_times_v(&model, &q, &v, ee_idx);
        let dj_v = DVector::from_column_slice(dj_v6.as_slice());

        let data = forward_kinematics(&model, &q);
        let ee_pos = se3::translation(&data.oMi[ee_idx]);
        let ee_lin_vel = {
            // Linear velocity of the EE origin: bottom 3 rows of J·v.
            let jv = &j_ee * DVector::from_column_slice(&v);
            Vector3::new(jv[3], jv[4], jv[5])
        };

        let a_lin = refgen::pd(
            &DVector::from_column_slice(x_ref.as_slice()),
            &DVector::from_column_slice(ee_pos.as_slice()),
            &DVector::from_column_slice(v_ref.as_slice()),
            &DVector::from_column_slice(ee_lin_vel.as_slice()),
            &kp.rows(3, 3).into_owned(),
            &kd.rows(3, 3).into_owned(),
        );
        // 6-D reference: angular rows held at 0 (keep EE orientation level).
        let mut a_ref = DVector::zeros(6);
        a_ref.rows_mut(3, 3).copy_from(&a_lin);

        let d = Dynamics::new(Formulation::Explicit, &mass, &h, &DMatrix::zeros(0, nv), na);
        let mut p0 = tasks::box_bound(d.tau(), &tau_max);
        if let Some(phys) = d.dynamics_task() {
            p0 = phys + p0;
        }
        let p1 = tasks::cartesian_acceleration(d.qddot(), &j_ee, &dj_v, &a_ref);
        let p2 = tasks::track(d.qddot(), &DVector::zeros(nv))
            + tasks::track(d.tau(), &DVector::zeros(na)).weight(0.01);

        let sol = solver.solve(&[p0, p1, p2], &cfg).expect("solve");
        let e = d.extract(&sol.x);

        // Semi-implicit Euler integration.
        for i in 0..nv {
            v[i] += e.qddot[i] * DT;
            q[i] += v[i] * DT;
        }

        // Numerical-divergence guard -- see panda_singularity_demo.rs
        // for why this matters (a solver can report Optimal every tick
        // while the integrated state quietly runs away).
        if q.iter().chain(v.iter()).any(|x| !x.is_finite() || x.abs() > 1e3) {
            eprintln!("STATE DIVERGED at tick {tick} (t={t:.4}s) -- stopping early, trace truncated here");
            break;
        }

        // Emit CSV row: tick,t,q...,ee_xyz,ref_xyz, then every link's
        // full world pose (translation + row-major rotation matrix),
        // joint 0 (universe root, = panda_link0) included so mesh
        // rendering can place every link without special-casing it.
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
        println!();
    }
}
