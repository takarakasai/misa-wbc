#!/usr/bin/env python3
"""Render go2_leg_singularity_demo's CSV trace into an MP4 using Go2's
own real visual meshes (go2.misa already vendors them, unlike Panda) --
VTK offscreen rendering, mirroring render_panda_vtk.py's approach but
sourcing (parent_joint, mesh_path, placement) from a mesh manifest
sidecar written by the demo itself instead of assuming any particular
robot's link names.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import vtk

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--trace", required=True)
ap.add_argument("--manifest", required=True, help="mesh manifest CSV (parent_joint,mesh_path,placement)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames-dir", required=True)
ap.add_argument("--title", default="misa-wbc -- Go2 leg singularity approach")
ap.add_argument("--n-q", type=int, default=12)
args = ap.parse_args()

TRACE_CSV = Path(args.trace)
MANIFEST_CSV = Path(args.manifest)
OUT_MP4 = Path(args.out)
FRAMES_DIR = Path(args.frames_dir)
TITLE = args.title

FPS = 50
STRIDE = 2
TRAIL_LEN = 60

# ---- column layout (see go2_leg_singularity_demo.rs) ----
IDX_T = 1
N_Q = args.n_q
IDX_EE = 2 + N_Q
IDX_REF = IDX_EE + 3
IDX_LINKS0 = IDX_REF + 3
LINK_STRIDE = 12


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


def compose(t1, r1, t2, r2):
    """world_T_mesh = (t1,r1) * (t2,r2)."""
    r = r1 @ r2
    t = r1 @ t2 + t1
    return t, r


rows = []
with open(TRACE_CSV) as f:
    for line in f:
        rows.append([float(v) for v in line.strip().split(",")])
rows = np.array(rows)
n_frames_total = len(rows)

# ---- mesh manifest: parent_joint index, resolved path, static placement ----
mesh_entries = []
with open(MANIFEST_CSV) as f:
    for row in csv.DictReader(f):
        parent = int(row["parent_joint"])
        path = row["mesh_path"]
        t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
        r = np.array([[float(row[f"r{i}{j}"]) for j in range(3)] for i in range(3)])
        mesh_entries.append((parent, path, t, r))

renderer = vtk.vtkRenderer()
renderer.SetBackground(0x0e / 255, 0x14 / 255, 0x20 / 255)

render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1000, 1000)
render_window.SetMultiSamples(8)

mesh_actors = []  # (parent_joint, placement_t, placement_r, actor)
for parent, path, t, r in mesh_entries:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ConsistencyOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.62, 0.63, 0.66)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.75)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(12)
    renderer.AddActor(actor)
    mesh_actors.append((parent, t, r, actor))
print(f"loaded {len(mesh_actors)} mesh objects", file=sys.stderr)


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
make_polyline_tube(ref_points, (0.16, 0.47, 0.84), 0.003, alpha=0.55, closed=False)

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


ee_sphere = vtk.vtkSphereSource()
ee_sphere.SetRadius(0.012)
ee_sphere.SetThetaResolution(20)
ee_sphere.SetPhiResolution(20)
ee_mapper = vtk.vtkPolyDataMapper()
ee_mapper.SetInputConnection(ee_sphere.GetOutputPort())
ee_actor = vtk.vtkActor()
ee_actor.SetMapper(ee_mapper)
ee_actor.GetProperty().SetColor(0.05, 0.72, 0.05)
ee_actor.GetProperty().SetAmbient(0.5)
renderer.AddActor(ee_actor)

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

key_light = vtk.vtkLight()
key_light.SetPosition(1.0, -1.2, 1.4)
key_light.SetFocalPoint(0.0, 0, 0.1)
key_light.SetIntensity(0.9)
key_light.SetColor(1.0, 1.0, 0.98)
renderer.AddLight(key_light)

fill_light = vtk.vtkLight()
fill_light.SetPosition(-1.0, 1.2, 0.8)
fill_light.SetFocalPoint(0.0, 0, 0.1)
fill_light.SetIntensity(0.35)
fill_light.SetColor(0.75, 0.82, 1.0)
renderer.AddLight(fill_light)

text_actor = vtk.vtkTextActor()
text_actor.SetPosition(24, 950)
text_actor.GetTextProperty().SetFontSize(20)
text_actor.GetTextProperty().SetColor(0.9, 0.92, 0.94)
text_actor.GetTextProperty().SetFontFamilyToCourier()
renderer.AddActor2D(text_actor)

# ---- camera framing ----
all_xyz = np.vstack([
    rows[:, IDX_EE:IDX_EE + 3],
    rows[:, IDX_REF:IDX_REF + 3],
])
# Include the base's own extent by sampling every mesh's placement translation.
mesh_pts = np.array([t for _, t, _, _ in mesh_actors])
center = np.vstack([all_xyz, mesh_pts]).mean(axis=0)
extent = max((all_xyz.max(axis=0) - all_xyz.min(axis=0)).max(), 0.6)
elev = np.radians(20)
azim = np.radians(-50)
distance = extent * 2.4 + 0.5
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

FRAMES_DIR.mkdir(exist_ok=True, parents=True)
for f in FRAMES_DIR.glob("*.png"):
    f.unlink()

frame_indices = list(range(0, n_frames_total, STRIDE))
print(f"Rendering {len(frame_indices)} frames...", file=sys.stderr)
for fi, tick in enumerate(frame_indices):
    row = rows[tick]
    for parent, pt, pr, actor in mesh_actors:
        jt, jr = link_pose(row, parent)
        wt, wr = compose(jt, jr, pt, pr)
        actor.SetUserMatrix(vtk_matrix_from_pose(wt, wr))

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
