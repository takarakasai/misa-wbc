//! misa-wbc の Python バインディング。
//!
//! 現状は**接触力の配分**だけを公開する。go2_rl の sim2sim ハーネスが
//! 支持レンチ前置き（`τ = −Jᵀf`）のために手書きの最小ノルム解を持っていて、
//! **摩擦錐と片側拘束を解いた後にクリップしていた**（それでは拘束を満たさない）。
//! misa-wbc なら QP の中で硬い不等式として扱える。
//!
//! misa-wbc は「行列を渡す」境界なので、この層もモデルを知らない。呼ぶ側が
//! レンチ写像 `A` と目標レンチを作る。

use misa_wbc::affine::{Affine, VarLayout};
use misa_wbc::solve::{solve, SolveConfig};
use misa_wbc::task::Task;
use misa_wbc::tasks;
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// 1 脚ぶんの力を選ぶ補助（`VarLayout` は脚ごとに 1 ブロック宣言する）。
fn leg_var(layout: &VarLayout, leg: usize) -> misa_wbc::affine::Var {
    layout.var(&format!("f{leg}"))
}

/// 目標レンチを接地脚へ配る（摩擦錐と片側拘束を硬い拘束として解く）。
///
/// `wrench_map`: 6 × 3n のレンチ写像（脚ごとに `[I; [r]×]` を並べたもの）。
/// `wrench`: 目標の胴体レンチ 6（`[力; モーメント]`）。
/// `stance`: 接地しているかの真偽 n 個。遊脚の力は 0 に拘束する。
/// `mu`: 摩擦係数（角錐近似）。`fz_max`: 1 脚あたりの垂直力の上限。
/// `reg`: 力の正則化（小さいほどレンチ追従を優先）。
///
/// 戻り値は n × 3 の接触力（平坦化した 3n）。
#[pyfunction]
#[pyo3(signature = (wrench_map, wrench, stance, mu=0.6, fz_max=1.0e4, reg=1.0e-3))]
pub fn distribute_contact_forces<'py>(
    py: Python<'py>,
    wrench_map: PyReadonlyArray2<f64>,
    wrench: PyReadonlyArray1<f64>,
    stance: Vec<bool>,
    mu: f64,
    fz_max: f64,
    reg: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = wrench_map.as_array();
    let n = stance.len();
    if a.shape() != [6, 3 * n] {
        return Err(PyValueError::new_err(format!(
            "wrench_map は (6, {}) であること: {:?}",
            3 * n,
            a.shape()
        )));
    }
    let w = wrench.as_slice()?;
    if w.len() != 6 {
        return Err(PyValueError::new_err("wrench は長さ 6"));
    }

    let mut b = VarLayout::builder();
    for leg in 0..n {
        b = b.add(format!("f{leg}"), 3);
    }
    let layout = b.build();

    let a_mat = DMatrix::from_fn(6, 3 * n, |i, j| a[[i, j]]);
    let w_vec = DVector::from_row_slice(w);

    // 優先度 1（硬い拘束）: 摩擦錐、垂直力の上限、遊脚の力ゼロ。
    let mut hard = Task::empty(layout.n_decision());
    for (leg, &in_stance) in stance.iter().enumerate() {
        let fi = leg_var(&layout, leg);
        if in_stance {
            hard = hard + tasks::friction_pyramid(&fi, mu);
            // fz <= fz_max（下限 0 は摩擦錐の 1 行目が持っている）
            let sel = DMatrix::from_row_slice(1, 3, &[0.0, 0.0, 1.0]);
            hard = hard + Task::le(&(&sel * &fi.affine()), &DVector::from_element(1, fz_max));
        } else {
            // 遊脚: f = 0 を硬く
            hard = hard + Task::le(&fi.affine(), &DVector::zeros(3))
                        + Task::ge(&fi.affine(), &DVector::zeros(3));
        }
    }

    // 優先度 2: 目標レンチの追従。優先度 3: 力の正則化。
    let f_all: Affine = {
        let mut m = DMatrix::zeros(3 * n, layout.n_decision());
        for i in 0..3 * n {
            m[(i, i)] = 1.0;
        }
        Affine::new(m, DVector::zeros(3 * n))
    };
    let track = tasks::track(&(&a_mat * &f_all), &w_vec);
    let regular = tasks::regularize(&f_all, &DVector::zeros(3 * n));

    let cfg = SolveConfig::default();
    let sol = solve(&[hard, track, regular.weight(reg)], &cfg)
        .map_err(|e| PyValueError::new_err(format!("WBC solve failed: {e:?}")))?;
    let x = sol.x;
    Ok(PyArray1::from_vec_bound(py, x.as_slice().to_vec()))
}

#[pymodule]
fn _misa_wbc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distribute_contact_forces, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
