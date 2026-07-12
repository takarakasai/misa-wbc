#!/usr/bin/env python3
"""Render panda_circle_demo's CSV trace (now carrying full per-link SE3
poses, not just positions) into an MP4, using the real Panda visual
meshes (sourced locally from an existing catkin package,
mc_models/panda_description) instead of a stick-figure skeleton.

Mesh link names line up 1:1 with our own panda.urdf's joint chain
(verified against both URDFs: identical joint origins, zero visual
offsets) -- link j's mesh is placed at the FK pose misa-wbc computed
for oMi[j], j=0..7 (panda_link0..panda_link7). No hand/finger meshes
were available in the found package, so the wrist onward is rendered
as simple primitives.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import vtk

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_TOPO_CSV = EXAMPLES_DIR / "models" / "panda_topology.csv"
# Visual meshes are NOT vendored in this repo -- panda.urdf (bullet3-sourced)
# references package://meshes/... paths whose .obj files were never fetched.
# These come from a *different* local catkin package (mc_models/
# panda_description) that happens to describe the same Franka Panda with
# real STL/DAE meshes. Verified against our own panda.urdf before use: same
# joint origins, zero visual offsets, so mesh link j's local frame lines up
# with oMi[j] from our own FK. Point --mesh-dir at your own copy if this
# path doesn't exist on your machine.
DEFAULT_MESH_DIR = Path("/home/kasai/work/hanamo/silence_meta/dep/mc_models/panda_description/meshes")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--trace", required=True, help="CSV trace from panda_circle_demo or panda_singularity_demo")
ap.add_argument("--out", required=True, help="output .mp4 path")
ap.add_argument("--frames-dir", required=True, help="scratch dir for intermediate PNG frames")
ap.add_argument("--topo", default=str(DEFAULT_TOPO_CSV))
ap.add_argument("--mesh-dir", default=str(DEFAULT_MESH_DIR))
ap.add_argument("--title", default="misa-wbc -- Panda WBC circle tracking")
ap.add_argument("--ref-closed", action="store_true", help="reference path is a closed loop (circle demo)")
args = ap.parse_args()

TRACE_CSV = Path(args.trace)
FRAMES_DIR = Path(args.frames_dir)
OUT_MP4 = Path(args.out)
TOPO_CSV = Path(args.topo)
MESH_DIR = Path(args.mesh_dir)
TITLE = args.title

FPS = 50
STRIDE = 2
TRAIL_LEN = 60  # ticks of EE trail to show, fading

# ---- column layout (see panda_circle_demo.rs) ----
IDX_T = 1
N_Q = 9
IDX_EE = 2 + N_Q       # = 11
IDX_REF = IDX_EE + 3   # = 14
IDX_LINKS0 = IDX_REF + 3  # = 17
LINK_STRIDE = 12  # tx,ty,tz, r00..r22 (row-major)

def link_pose(row, j):
    base = IDX_LINKS0 + j * LINK_STRIDE
    t = row[base:base + 3]
    r = np.array(row[base + 3:base + 12]).reshape(3, 3)
    return t, r

def vtk_matrix_from_pose(t, r):
    m = vtk.vtkMatrix4x4()
    m.Identity()
    for i in range(3):
        for jc in range(3):
            m.SetElement(i, jc, r[i, jc])
        m.SetElement(i, 3, t[i])
    return m

# ---- load trace ----
rows = []
with open(TRACE_CSV) as f:
    f.readline()
    for line in f:
        rows.append([float(v) for v in line.strip().split(",")])
rows = np.array(rows)
n_frames_total = len(rows)

# joint idx -> child link mesh basename, for the 8 links we have real
# meshes for (panda_link0..panda_link7 == oMi[0..7]).
MESH_LINKS = {j: f"panda_link{j}" for j in range(8)}

renderer = vtk.vtkRenderer()
renderer.SetBackground(0x0e / 255, 0x14 / 255, 0x20 / 255)

render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1000, 1000)
render_window.SetMultiSamples(8)

# ---- one actor per real link mesh, transform updated per-frame ----
link_actors = {}
for j, name in MESH_LINKS.items():
    stl_path = MESH_DIR / f"{name}_v.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ConsistencyOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.92, 0.93, 0.95)
    actor.GetProperty().SetAmbient(0.28)
    actor.GetProperty().SetDiffuse(0.72)
    actor.GetProperty().SetSpecular(0.25)
    actor.GetProperty().SetSpecularPower(18)
    renderer.AddActor(actor)
    link_actors[j] = actor
    print(f"loaded {name}_v.stl", file=sys.stderr)

# ---- wrist/hand stand-in (no mesh available past link7): a small dark
# capsule-ish stack of primitives, driven by oMi[9] (hand) and
# oMi[10]/oMi[11] (fingers), so the arm doesn't look severed. ----
def make_box_actor(color, size):
    src = vtk.vtkCubeSource()
    src.SetXLength(size[0])
    src.SetYLength(size[1])
    src.SetZLength(size[2])
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    renderer.AddActor(actor)
    return actor

hand_actor = make_box_actor((0.15, 0.16, 0.18), (0.09, 0.09, 0.07))
lfinger_actor = make_box_actor((0.1, 0.1, 0.12), (0.02, 0.018, 0.045))
rfinger_actor = make_box_actor((0.1, 0.1, 0.12), (0.02, 0.018, 0.045))

# ---- reference circle: static tube over the full path ----
def make_polyline_tube(points, color, radius, alpha=1.0, closed=False):
    pts = vtk.vtkPoints()
    for p in points:
        pts.InsertNextPoint(*p)
    n = len(points)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n + (1 if closed else 0))
    for i in range(n):
        lines.InsertCellPoint(i)
    if closed:
        lines.InsertCellPoint(0)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(10)
    tube.CappingOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(alpha)
    renderer.AddActor(actor)
    return actor, poly, tube

ref_points = list(rows[:, IDX_REF:IDX_REF + 3])
make_polyline_tube(ref_points, (0.16, 0.47, 0.84), 0.003, alpha=0.55, closed=args.ref_closed)

# ---- EE trail: rebuilt every frame ----
trail_pts = vtk.vtkPoints()
trail_lines = vtk.vtkCellArray()
trail_poly = vtk.vtkPolyData()
trail_poly.SetPoints(trail_pts)
trail_poly.SetLines(trail_lines)
trail_tube = vtk.vtkTubeFilter()
trail_tube.SetInputData(trail_poly)
trail_tube.SetRadius(0.0035)
trail_tube.SetNumberOfSides(8)
trail_mapper = vtk.vtkPolyDataMapper()
trail_mapper.SetInputConnection(trail_tube.GetOutputPort())
trail_actor = vtk.vtkActor()
trail_actor.SetMapper(trail_mapper)
trail_actor.GetProperty().SetColor(0.93, 0.63, 0.0)
renderer.AddActor(trail_actor)

def update_trail(tick):
    start = max(0, tick - TRAIL_LEN * STRIDE)
    sel = rows[start:tick + 1:STRIDE, IDX_EE:IDX_EE + 3]
    trail_pts.Reset()
    trail_lines.Reset()
    for p in sel:
        trail_pts.InsertNextPoint(*p)
    n = len(sel)
    if n > 1:
        trail_lines.InsertNextCell(n)
        for i in range(n):
            trail_lines.InsertCellPoint(i)
    trail_pts.Modified()
    trail_poly.Modified()

# ---- EE marker (current) ----
ee_sphere = vtk.vtkSphereSource()
ee_sphere.SetRadius(0.014)
ee_sphere.SetThetaResolution(20)
ee_sphere.SetPhiResolution(20)
ee_mapper = vtk.vtkPolyDataMapper()
ee_mapper.SetInputConnection(ee_sphere.GetOutputPort())
ee_actor = vtk.vtkActor()
ee_actor.SetMapper(ee_mapper)
ee_actor.GetProperty().SetColor(0.05, 0.72, 0.05)
ee_actor.GetProperty().SetAmbient(0.5)
renderer.AddActor(ee_actor)

# ---- ground grid (subtle, for spatial reference) ----
grid = vtk.vtkPlaneSource()
grid.SetOrigin(-0.6, -0.6, 0.0)
grid.SetPoint1(0.6, -0.6, 0.0)
grid.SetPoint2(-0.6, 0.6, 0.0)
grid.SetXResolution(12)
grid.SetYResolution(12)
grid_mapper = vtk.vtkPolyDataMapper()
grid_mapper.SetInputConnection(grid.GetOutputPort())
grid_actor = vtk.vtkActor()
grid_actor.SetMapper(grid_mapper)
grid_actor.GetProperty().SetRepresentationToWireframe()
grid_actor.GetProperty().SetColor(0.25, 0.3, 0.38)
grid_actor.GetProperty().SetOpacity(0.35)
grid_actor.GetProperty().SetLineWidth(1)
renderer.AddActor(grid_actor)

# ---- lighting ----
key_light = vtk.vtkLight()
key_light.SetPosition(1.2, -1.0, 1.8)
key_light.SetFocalPoint(0.3, 0, 0.4)
key_light.SetIntensity(0.9)
key_light.SetColor(1.0, 1.0, 0.98)
renderer.AddLight(key_light)

fill_light = vtk.vtkLight()
fill_light.SetPosition(-1.0, 1.2, 0.6)
fill_light.SetFocalPoint(0.3, 0, 0.4)
fill_light.SetIntensity(0.35)
fill_light.SetColor(0.75, 0.82, 1.0)
renderer.AddLight(fill_light)

# ---- title text overlay ----
text_actor = vtk.vtkTextActor()
text_actor.SetPosition(24, 950)
text_actor.GetTextProperty().SetFontSize(22)
text_actor.GetTextProperty().SetColor(0.9, 0.92, 0.94)
text_actor.GetTextProperty().SetFontFamilyToCourier()
renderer.AddActor2D(text_actor)

# ---- camera: centred on the reachable workspace, similar framing to
# the earlier matplotlib view (elev~14deg, azim~-60deg). ----
all_xyz = np.vstack([
    np.array([link_pose(rows[i], j)[0] for i in range(0, n_frames_total, 25) for j in range(8)]),
    rows[:, IDX_EE:IDX_EE + 3],
])
center = all_xyz.mean(axis=0)
extent = (all_xyz.max(axis=0) - all_xyz.min(axis=0)).max()
elev = np.radians(16)
azim = np.radians(-55)
distance = extent * 1.9 + 0.6
direction = np.array([
    np.cos(elev) * np.cos(azim),
    np.cos(elev) * np.sin(azim),
    np.sin(elev),
])
cam_pos = center + distance * direction

camera = renderer.GetActiveCamera()
camera.SetFocalPoint(*center)
camera.SetPosition(*cam_pos)
camera.SetViewUp(0, 0, 1)
camera.SetViewAngle(35)

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window)
w2i.SetInputBufferTypeToRGB()
writer = vtk.vtkPNGWriter()

FRAMES_DIR.mkdir(exist_ok=True)
for f in FRAMES_DIR.glob("*.png"):
    f.unlink()

frame_indices = list(range(0, n_frames_total, STRIDE))
print(f"Rendering {len(frame_indices)} frames...", file=sys.stderr)
for fi, tick in enumerate(frame_indices):
    row = rows[tick]
    for j, actor in link_actors.items():
        t, r = link_pose(row, j)
        actor.SetUserMatrix(vtk_matrix_from_pose(t, r))
    hand_t, hand_r = link_pose(row, 9)
    hand_actor.SetUserMatrix(vtk_matrix_from_pose(hand_t, hand_r))
    lf_t, lf_r = link_pose(row, 10)
    lfinger_actor.SetUserMatrix(vtk_matrix_from_pose(lf_t, lf_r))
    rf_t, rf_r = link_pose(row, 11)
    rfinger_actor.SetUserMatrix(vtk_matrix_from_pose(rf_t, rf_r))

    update_trail(tick)
    ee_actor.SetPosition(*row[IDX_EE:IDX_EE + 3])
    text_actor.SetInput(f"{TITLE}   t = {row[IDX_T]:5.2f}s")

    render_window.Render()
    w2i.Modified()
    w2i.Update()
    writer.SetFileName(str(FRAMES_DIR / f"f{fi:05d}.png"))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    if fi % 30 == 0:
        print(f"  frame {fi}/{len(frame_indices)}", file=sys.stderr)

print("Encoding mp4...", file=sys.stderr)
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(FPS),
    "-i", str(FRAMES_DIR / "f%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    str(OUT_MP4),
], check=True)
print(f"Done: {OUT_MP4}", file=sys.stderr)
