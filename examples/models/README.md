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
