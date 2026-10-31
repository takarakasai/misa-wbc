# Native QP adapters

Run `bash native/setup.sh` (from this crate's root). This fetches
fixed Git revisions and builds OSQP, HPIPM and BLASFEO beneath
`/tmp/articara-native-qp`. Override with `MISA_QP_NATIVE_ROOT` both when running
setup and Cargo. Requires Git, CMake, a C++17 compiler and Eigen 3 headers
(`/usr/include/eigen3`; validated with system Eigen 3.4.0). Setup does not modify system installations. Dependency
licenses remain in each checkout; deployment must retain their notices
(including LGPL obligations for qpOASES).

Features are `proxqp`, `eiquadprog`, `qpoases`, `osqp`, `hpipm`, and the aggregate
`native-all`. No native dependency is built or linked with default features.
Example validation from the root:

```
cargo test -p misa-wbc --lib --features misa-wbc/native-all native_tests
```

The bridge preserves column-major Rust matrices, transforms sign conventions
for Eiquadprog/qpOASES/HPIPM, and returns original backend codes. Nonfinite QP
inputs are rejected before FFI. Rust dimensions are checked before pointer use;
C++ exceptions become numerical failure. OSQP uses fixed full upper-triangle P
and dense A CSC structure; it does not measure a tuned sparse application layout.

`QpWorkspace` retains ProxQP update state, qpOASES SQProblem hotstart state, and
OSQP matrix/factorization state. Matrix-value updates are supported. `clear()`
destroys the native context. Dimension/backend changes rebuild it. Callers must
clear when contact IDs or row meanings change despite identical dimensions.
Cloning a workspace clones the primal/dual seeds but creates a fresh native
context. Eiquadprog and HPIPM currently perform cold solves. HPIPM's separate
`ocp::solve_ocp` directly uses its OCP interface and classical Riccati recursion;
it also performs cold solves.

Timing diagnostics separate conversion, setup, update and solve. qpOASES
`init`/`hotstart` combine internal setup/update with solving, so that combined
native call is recorded as solve; it cannot be decomposed further by this API.
OSQP matrix updates can refactorize; those costs are in update. The first solve
in a warm replay is necessarily cold. ProxQP explicitly checks the duality gap as well as primal/stationary residuals;
this is essential when an active bound moves during warm replay.
ProxQP is built without SIMD optional
headers, BLASFEO uses GENERIC architecture, and this baseline must not be
presented as hardware-tuned best performance.

Pinned revisions are recorded explicitly in `setup.sh`. Each native backend is
independently optional, and selecting a backend without its feature fails with
an explicit configuration error. `QpSolution::diagnostics` is additive; older
internal backends leave it absent. Clarabel records its exact status, including
`AlmostSolved`, while its legacy coarse status mapping is preserved.
