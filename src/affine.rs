//! Affine optimization variables — the OpenSoT-style vocabulary that
//! lets tasks be written symbolically in terms of named variables.
//!
//! A whole-body-control QP has a stacked decision vector, e.g.
//! `x = [q̈ ; f ; τ]`. Writing tasks by hand-indexing into `x` is
//! error-prone and re-derives the layout at every call site. Instead:
//!
//! - [`VarLayout`] declares the layout once (`("qddot", nv)`,
//!   `("f", 3·nc)`, …) and hands out a [`Var`] selector per variable.
//! - [`Affine`] models an affine map `y = M·x + q` from the **global**
//!   `x` to a **local** quantity `y`. Operators (`&M * &var`,
//!   `&affine + &affine`, `&affine ± &vec`, `-&affine`) compose these,
//!   so a task residual reads like the math:
//!
//! ```text
//! // floating-base EoM residual  e = M_u·q̈ − Jc_uᵀ·f + h_u
//! let e = &m_u * &qddot - &jc_u_t * &f + &h_u;   // Affine
//! let eom = Task::soft_eq(&e);                    // drives ‖e‖² → 0
//! ```
//!
//! This mirrors OpenSoT's `AffineHelper` (`y = M·x + q`, dense). The
//! representation is a dense `DMatrix` sized `out × n_decision`; at WBC
//! scale (a few dozen to a couple hundred decision variables) the zero
//! blocks are cheap and the matrix-product composition keeps the
//! operators trivial.

use nalgebra::{DMatrix, DVector};

// ─── VarLayout ──────────────────────────────────────────────────────────────

/// A declared layout of the global decision vector `x` as a sequence of
/// named variable blocks. Build with [`VarLayout::builder`].
#[derive(Clone, Debug)]
pub struct VarLayout {
    /// `(name, offset, size)` in declaration order.
    entries: Vec<(String, usize, usize)>,
    n_decision: usize,
}

/// Builder for [`VarLayout`]. Blocks are laid out contiguously in the
/// order they are [`add`](VarLayoutBuilder::add)ed.
#[derive(Clone, Debug, Default)]
pub struct VarLayoutBuilder {
    entries: Vec<(String, usize)>,
}

impl VarLayout {
    pub fn builder() -> VarLayoutBuilder {
        VarLayoutBuilder::default()
    }

    /// Total number of decision variables (sum of all block sizes).
    pub fn n_decision(&self) -> usize {
        self.n_decision
    }

    /// The selector [`Var`] for a declared variable. Panics if `name`
    /// was never added — use [`try_var`](Self::try_var) to handle that.
    pub fn var(&self, name: &str) -> Var {
        self.try_var(name)
            .unwrap_or_else(|| panic!("VarLayout: unknown variable '{name}'"))
    }

    /// The selector [`Var`], or `None` if `name` isn't in the layout.
    pub fn try_var(&self, name: &str) -> Option<Var> {
        self.entries
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|&(_, offset, size)| Var {
                offset,
                size,
                n_decision: self.n_decision,
            })
    }

    /// Declared variable names, in layout order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _, _)| n.as_str())
    }
}

impl VarLayoutBuilder {
    /// Append a variable block of `size` decision variables. Zero-size
    /// blocks are allowed (e.g. `nc == 0`) and simply occupy no range.
    pub fn add(mut self, name: impl Into<String>, size: usize) -> Self {
        self.entries.push((name.into(), size));
        self
    }

    pub fn build(self) -> VarLayout {
        let mut offset = 0;
        let mut entries = Vec::with_capacity(self.entries.len());
        for (name, size) in self.entries {
            entries.push((name, offset, size));
            offset += size;
        }
        VarLayout {
            entries,
            n_decision: offset,
        }
    }
}

// ─── Var ────────────────────────────────────────────────────────────────────

/// A selector for one variable block inside the global decision vector.
/// Multiply a coefficient matrix by it (`&coef * &var`) to build an
/// [`Affine`], or [`extract`](Var::extract) its slice out of a solution.
#[derive(Clone, Copy, Debug)]
pub struct Var {
    offset: usize,
    size: usize,
    n_decision: usize,
}

impl Var {
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn n_decision(&self) -> usize {
        self.n_decision
    }

    /// Promote to a full [`Affine`] `y = S·x` where `S` is the selection
    /// matrix picking this block (`M = S`, `q = 0`).
    pub fn affine(&self) -> Affine {
        let mut m = DMatrix::zeros(self.size, self.n_decision);
        for i in 0..self.size {
            m[(i, self.offset + i)] = 1.0;
        }
        Affine {
            m,
            q: DVector::zeros(self.size),
        }
    }

    /// Pull this variable's slice out of a global solution vector.
    pub fn extract(&self, x: &DVector<f64>) -> DVector<f64> {
        x.rows(self.offset, self.size).into_owned()
    }
}

/// Anything that can stand in for an affine expression of the decision
/// vector — a raw [`Var`] block or a composed [`Affine`]. Task builders
/// accept `&impl AsAffine` so the same call works with a plain variable
/// (`x = [q̈; f; τ]`, the explicit formulation) or an eliminated
/// quantity (`q̈` as `M⁻¹(Sᵀτ + Jcᵀf − h)` in the force-space
/// formulation) without a separate `_expr` API.
pub trait AsAffine {
    /// The expression as a concrete [`Affine`] map.
    fn as_affine(&self) -> Affine;
    /// Output dimension of the expression.
    fn out_size(&self) -> usize;
}

impl AsAffine for Var {
    fn as_affine(&self) -> Affine {
        self.affine()
    }
    fn out_size(&self) -> usize {
        self.size
    }
}

impl AsAffine for Affine {
    fn as_affine(&self) -> Affine {
        self.clone()
    }
    fn out_size(&self) -> usize {
        Affine::out_size(self)
    }
}

// ─── Affine ─────────────────────────────────────────────────────────────────

/// An affine map `y = M·x + q` from the global decision vector `x` to a
/// local quantity `y`. `M` is `out_size × n_decision` (dense).
#[derive(Clone, Debug)]
pub struct Affine {
    m: DMatrix<f64>,
    q: DVector<f64>,
}

impl Affine {
    /// Construct directly from `M` and `q` (must agree: `M.nrows() ==
    /// q.len()`).
    pub fn new(m: DMatrix<f64>, q: DVector<f64>) -> Self {
        assert_eq!(
            m.nrows(),
            q.len(),
            "Affine::new: M rows ({}) must equal q len ({})",
            m.nrows(),
            q.len(),
        );
        Self { m, q }
    }

    /// A constant affine `y = c` (i.e. `M = 0`).
    pub fn constant(c: DVector<f64>, n_decision: usize) -> Self {
        Self {
            m: DMatrix::zeros(c.len(), n_decision),
            q: c,
        }
    }

    /// A zero affine `y = 0` of the given output size.
    pub fn zeros(out_size: usize, n_decision: usize) -> Self {
        Self {
            m: DMatrix::zeros(out_size, n_decision),
            q: DVector::zeros(out_size),
        }
    }

    pub fn out_size(&self) -> usize {
        self.m.nrows()
    }
    pub fn n_decision(&self) -> usize {
        self.m.ncols()
    }
    pub fn m(&self) -> &DMatrix<f64> {
        &self.m
    }
    pub fn q(&self) -> &DVector<f64> {
        &self.q
    }

    /// Evaluate `y = M·x + q` at a concrete decision vector.
    pub fn eval(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.m * x + &self.q
    }
}

impl From<&Var> for Affine {
    fn from(v: &Var) -> Self {
        v.affine()
    }
}

impl From<Var> for Affine {
    fn from(v: Var) -> Self {
        v.affine()
    }
}

// ─── Operators ──────────────────────────────────────────────────────────────
//
// Enough to write task residuals fluently. `&coef * &var` and
// `&coef * &affine` inject a coefficient matrix; `±` between affines and
// with constant vectors composes them. All check dimension agreement.

/// `coef · var` → affine. `coef` is `out × var.size`; the result places
/// `coef`'s columns into the var's block of the global layout.
impl std::ops::Mul<&Var> for &DMatrix<f64> {
    type Output = Affine;
    fn mul(self, var: &Var) -> Affine {
        assert_eq!(
            self.ncols(),
            var.size,
            "matrix * var: matrix cols ({}) must equal var size ({})",
            self.ncols(),
            var.size,
        );
        let mut m = DMatrix::zeros(self.nrows(), var.n_decision);
        m.view_mut((0, var.offset), (self.nrows(), var.size))
            .copy_from(self);
        Affine {
            m,
            q: DVector::zeros(self.nrows()),
        }
    }
}

/// `coef · affine` → affine (`M' = coef·M`, `q' = coef·q`).
impl std::ops::Mul<&Affine> for &DMatrix<f64> {
    type Output = Affine;
    fn mul(self, a: &Affine) -> Affine {
        assert_eq!(
            self.ncols(),
            a.m.nrows(),
            "matrix * affine: matrix cols ({}) must equal affine out_size ({})",
            self.ncols(),
            a.m.nrows(),
        );
        Affine {
            m: self * &a.m,
            q: self * &a.q,
        }
    }
}

fn assert_compat(a: &Affine, b: &Affine, op: &str) {
    assert_eq!(
        a.m.ncols(),
        b.m.ncols(),
        "Affine {op}: n_decision mismatch ({} vs {})",
        a.m.ncols(),
        b.m.ncols(),
    );
    assert_eq!(
        a.m.nrows(),
        b.m.nrows(),
        "Affine {op}: out_size mismatch ({} vs {})",
        a.m.nrows(),
        b.m.nrows(),
    );
}

impl std::ops::Add<&Affine> for &Affine {
    type Output = Affine;
    fn add(self, rhs: &Affine) -> Affine {
        assert_compat(self, rhs, "+");
        Affine {
            m: &self.m + &rhs.m,
            q: &self.q + &rhs.q,
        }
    }
}

impl std::ops::Sub<&Affine> for &Affine {
    type Output = Affine;
    fn sub(self, rhs: &Affine) -> Affine {
        assert_compat(self, rhs, "-");
        Affine {
            m: &self.m - &rhs.m,
            q: &self.q - &rhs.q,
        }
    }
}

/// `affine + const_vector` shifts `q`.
impl std::ops::Add<&DVector<f64>> for &Affine {
    type Output = Affine;
    fn add(self, rhs: &DVector<f64>) -> Affine {
        assert_eq!(
            self.m.nrows(),
            rhs.len(),
            "Affine + vec: out_size ({}) must equal vec len ({})",
            self.m.nrows(),
            rhs.len(),
        );
        Affine {
            m: self.m.clone(),
            q: &self.q + rhs,
        }
    }
}

impl std::ops::Sub<&DVector<f64>> for &Affine {
    type Output = Affine;
    fn sub(self, rhs: &DVector<f64>) -> Affine {
        assert_eq!(
            self.m.nrows(),
            rhs.len(),
            "Affine - vec: out_size ({}) must equal vec len ({})",
            self.m.nrows(),
            rhs.len(),
        );
        Affine {
            m: self.m.clone(),
            q: &self.q - rhs,
        }
    }
}

impl std::ops::Neg for &Affine {
    type Output = Affine;
    fn neg(self) -> Affine {
        Affine {
            m: -&self.m,
            q: -&self.q,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> VarLayout {
        // x = [ qddot(2) ; f(3) ]   -> n_decision = 5
        VarLayout::builder().add("qddot", 2).add("f", 3).build()
    }

    #[test]
    fn layout_offsets_and_sizes() {
        let l = layout();
        assert_eq!(l.n_decision(), 5);
        assert_eq!(l.var("qddot").offset(), 0);
        assert_eq!(l.var("qddot").size(), 2);
        assert_eq!(l.var("f").offset(), 2);
        assert_eq!(l.var("f").size(), 3);
        assert!(l.try_var("nope").is_none());
    }

    #[test]
    fn var_affine_is_selection_matrix() {
        let l = layout();
        let f = l.var("f").affine();
        assert_eq!(f.out_size(), 3);
        assert_eq!(f.n_decision(), 5);
        // selects columns 2,3,4
        let x = DVector::from_vec(vec![10.0, 20.0, 1.0, 2.0, 3.0]);
        assert_eq!(f.eval(&x), DVector::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn matrix_times_var_places_block() {
        let l = layout();
        let qddot = l.var("qddot");
        // coef 2x2 acting on qddot -> affine 2x5 with coef in cols 0..2
        let coef = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let a = &coef * &qddot;
        let x = DVector::from_vec(vec![5.0, 6.0, 0.0, 0.0, 0.0]);
        // [1*5+2*6 ; 3*5+4*6] = [17 ; 39]
        assert_eq!(a.eval(&x), DVector::from_vec(vec![17.0, 39.0]));
    }

    /// Compose a floating-base-EoM-shaped residual with the operators and
    /// check it equals the hand-assembled `M·x + q`.
    #[test]
    fn composed_residual_matches_hand_assembly() {
        let l = layout();
        let qddot = l.var("qddot");
        let f = l.var("f");

        // e = Mu·qddot − Jcuᵀ·f + hu     (out_size 2)
        let m_u = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let jc_u_t = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let h_u = DVector::from_vec(vec![0.5, -0.5]);

        let e = &(&(&m_u * &qddot) - &(&jc_u_t * &f)) + &h_u;
        assert_eq!(e.out_size(), 2);
        assert_eq!(e.n_decision(), 5);

        // hand assembly: M = [Mu | −Jcuᵀ], q = hu
        let x = DVector::from_vec(vec![1.0, 2.0, 0.1, 0.2, 0.3]);
        let mut m = DMatrix::zeros(2, 5);
        m.view_mut((0, 0), (2, 2)).copy_from(&m_u);
        m.view_mut((0, 2), (2, 3)).copy_from(&(-&jc_u_t));
        let expected = &m * &x + &h_u;
        assert!((e.eval(&x) - expected).norm() < 1e-12);
    }

    #[test]
    fn neg_and_const_and_zeros() {
        let l = layout();
        let f = l.var("f").affine();
        let n = (-&f).eval(&DVector::from_vec(vec![0.0, 0.0, 1.0, 2.0, 3.0]));
        assert_eq!(n, DVector::from_vec(vec![-1.0, -2.0, -3.0]));

        let c = Affine::constant(DVector::from_vec(vec![7.0, 8.0]), 5);
        assert_eq!(
            c.eval(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0])),
            DVector::from_vec(vec![7.0, 8.0])
        );
        assert_eq!(Affine::zeros(3, 5).out_size(), 3);
    }
}
