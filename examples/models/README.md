# Benchmark models

- `panda.urdf` — Franka Emika Panda, from
  [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3)
  `examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf`
  (zlib license; inertial parameters derived from the Apache-2.0
  `franka_description`). Kinematics + inertials only are used here —
  the referenced meshes are not included.
- `panda_topology.csv` — joint index/name/parent sidecar, regenerated
  each run by `panda_circle_demo`/`panda_singularity_demo` (not meant
  to be hand-edited; kept here only because the demos write it as a
  side effect for `render_panda_vtk.py` to read).
- Visual mesh rendering (`../render_panda_vtk.py`) draws the real
  Panda shape by sourcing STL meshes from a *different* local
  catkin package (`mc_models/panda_description`) that happens to
  describe the same robot — not vendored in this repo. See the
  script's docstring for how the two URDFs were cross-checked before
  use, and `--mesh-dir` to point at your own copy.
- `go2_topology.csv` — same idea as `panda_topology.csv`, regenerated
  by `go2_leg_singularity_demo`.
- `go2_mesh_manifest.csv` — regenerated each run: for every visual
  `GeometryObject` in the model (`build_model`'s `vis` return value),
  its parent joint index, resolved `.obj` path, and static placement
  (translation + rotation) relative to that joint. Unlike Panda's
  meshes (zero visual offset, verified by hand), several of Go2's own
  meshes carry a real per-mesh rotation (e.g. `FR_hip`'s, `rpy=[pi,0,0]`),
  so this is read straight from `GeometryObject::placement` rather than
  assumed — see `../render_go2_vtk.py`.
- The Go2 model itself (`go2.misa`) is **not** in this repo —
  `go2_leg_singularity_demo.rs` loads it via a relative path into the
  sibling `articara` checkout (`../../articara/models/unitree_go2/go2.misa`,
  a submodule shared with `go2-gait-runner`), including its real
  `.obj` meshes (`assets/`) — no external mesh source needed, unlike
  Panda. Only meaningful to run with that sibling repo present.
