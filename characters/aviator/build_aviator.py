"""Procedurally builds "Wren", a sky-pirate aviator character, in Blender.

Run headless:
    blender --background --python build_aviator.py -- [--render out.png] [--view full|portrait|back|face]
                                                      [--samples N] [--res WxH] [--save file.blend]

Z is up, the character faces -Y, units are meters (about 1.78 m tall).
Everything is generated from code: the body is a Skin-modifier armature graph,
the head is a sphere sculpted with analytic gaussian features, clothing and
props are snapped onto the body with BVH ray casts.
"""
import bpy
import bmesh
import math
import random
import sys
from mathutils import Vector, Matrix, Euler
from mathutils.bvhtree import BVHTree

random.seed(7)

# ---------------------------------------------------------------- utilities

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def active():
    return bpy.context.active_object


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def add_subsurf(obj, levels=2, render=None):
    m = obj.modifiers.new("Subsurf", 'SUBSURF')
    m.levels = levels
    m.render_levels = render if render is not None else levels
    return m


def bevel(obj, width, segments=2):
    b = obj.modifiers.new("Bevel", 'BEVEL')
    b.width = width
    b.segments = segments
    return b


def mat_to(obj, mat):
    obj.data.materials.append(mat)
    return obj


def mirror_nodes(nodes):
    """Duplicate every node whose name ends in _L into an _R twin with x flipped."""
    out = dict(nodes)
    for k, (co, r) in nodes.items():
        if k.endswith("_L"):
            out[k[:-2] + "_R"] = ((-co[0], co[1], co[2]), r)
    return out


def mirror_edges(edges):
    out = list(edges)
    for a, b in edges:
        if a.endswith("_L") or b.endswith("_L"):
            ra = a[:-2] + "_R" if a.endswith("_L") else a
            rb = b[:-2] + "_R" if b.endswith("_L") else b
            out.append((ra, rb))
    return out


def skin_object(name, nodes, edges, root, subsurf=2, mat=None, branch_smoothing=0.6):
    """Organic tube mesh from a node graph via the Skin modifier.

    nodes: name -> ((x, y, z), radius or (rx, ry))
    """
    names = list(nodes.keys())
    idx = {n: i for i, n in enumerate(names)}
    me = bpy.data.meshes.new(name)
    me.from_pydata([nodes[n][0] for n in names], [(idx[a], idx[b]) for a, b in edges], [])
    obj = link(bpy.data.objects.new(name, me))
    mod = obj.modifiers.new("Skin", 'SKIN')
    mod.branch_smoothing = branch_smoothing
    mod.use_smooth_shade = True
    sv = me.skin_vertices[0].data
    for n in names:
        r = nodes[n][1]
        if not isinstance(r, (tuple, list)):
            r = (r, r)
        sv[idx[n]].radius = tuple(r)
        sv[idx[n]].use_root = (n == root)
    if subsurf:
        add_subsurf(obj, subsurf)
    if mat:
        mat_to(obj, mat)
    return obj


def apply_modifiers(obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if obj.type != 'MESH':
        bpy.ops.object.convert(target='MESH')
    else:
        for m in list(obj.modifiers):
            bpy.ops.object.modifier_apply(modifier=m.name)
    obj.select_set(False)
    return obj


def gauss(x, mu, s):
    return math.exp(-((x - mu) ** 2) / (2 * s * s))


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def bez_curve(name, pts, radius, mat=None, radii=None, tilts=None, resolution=3, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = resolution
    cu.use_fill_caps = True
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        bp = sp.bezier_points[i]
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
        if radii:
            bp.radius = radii[i]
        if tilts:
            bp.tilt = tilts[i]
    sp.use_cyclic_u = cyclic
    obj = link(bpy.data.objects.new(name, cu))
    if mat:
        cu.materials.append(mat)
    return obj


def profile(name, pts, smooth_profile=False):
    """Closed 2D cross-section used as a curve bevel object."""
    cu = bpy.data.curves.new(name, 'CURVE')
    if smooth_profile:
        sp = cu.splines.new('BEZIER')
        sp.bezier_points.add(len(pts) - 1)
        for i, (x, y) in enumerate(pts):
            bp = sp.bezier_points[i]
            bp.co = (x, y, 0)
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
    else:
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for i, (x, y) in enumerate(pts):
            sp.points[i].co = (x, y, 0, 1)
    sp.use_cyclic_u = True
    obj = link(bpy.data.objects.new(name, cu))
    obj.hide_render = obj.hide_viewport = True
    return obj


def strap_curve(name, pts, width, thick, mat, cyclic=False, tilt=0.0):
    obj = bez_curve(name, pts, 0.0, mat, cyclic=cyclic)
    obj.data.bevel_mode = 'OBJECT'
    obj.data.bevel_object = profile(name + "_prof", [(-thick, -width), (thick, -width), (thick, width), (-thick, width)])
    obj.data.twist_mode = 'MINIMUM'
    for bp in obj.data.splines[0].bezier_points:
        bp.tilt = tilt
    return obj


class Surface:
    """Ray-cast helper over an object's evaluated (modifier-applied) surface in world space."""

    def __init__(self, *objs):
        dg = bpy.context.evaluated_depsgraph_get()
        verts, polys = [], []
        for obj in objs:
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            mw = obj.matrix_world
            off = len(verts)
            verts += [mw @ v.co for v in me.vertices]
            polys += [[off + i for i in p.vertices] for p in me.polygons]
            ev.to_mesh_clear()
        self.bvh = BVHTree.FromPolygons(verts, polys)

    def hit(self, origin, direction):
        loc, nor, _, _ = self.bvh.ray_cast(Vector(origin), Vector(direction).normalized())
        return loc, nor

    def onto(self, point, axis_center, offset=0.0):
        """Shoot from far outside toward an axis (point projected to a vertical axis) and return the surface point."""
        p = Vector(point)
        c = Vector((axis_center[0], axis_center[1], p.z))
        d = (p - c)
        d.z = 0
        d.normalize()
        loc, nor = self.hit(c + d * 1.0, -d)
        if loc is None:
            return p, d
        return loc + nor * offset, nor

    def front(self, x, z, offset=0.0, origin_y=-1.0):
        loc, nor = self.hit((x, origin_y, z), (0, 1, 0))
        if loc is None or nor.y > 0:
            return None, None
        return loc + nor * offset, nor


# ---------------------------------------------------------------- materials

def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    return m, nt, bsdf


def add_bump(nt, bsdf, scale, strength, detail=8.0, distance=0.01, kind="noise"):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if kind == "voronoi":
        tex = nt.nodes.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = scale
        out = tex.outputs["Distance"]
    elif kind == "wave":
        tex = nt.nodes.new("ShaderNodeTexWave")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Distortion"].default_value = 2.0
        tex.inputs["Detail"].default_value = 2.0
        out = tex.outputs["Fac"]
    else:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = detail
        out = tex.outputs["Fac"]
    nt.links.new(tc.outputs["Object"], tex.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    nt.links.new(out, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return tex


def noise_ramp(nt, c1, c2, scale, lo=0.35, hi=0.65):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = lo
    ramp.color_ramp.elements[0].color = c1
    ramp.color_ramp.elements[1].position = hi
    ramp.color_ramp.elements[1].color = c2
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]


def ellipse_mask(nt, cx, cz, rx, rz, soft=0.35, mirror=False):
    """1 inside an ellipse in the object's XZ plane, fading to 0 at its edge."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    xin = sep.outputs["X"]
    if mirror:
        ab = nt.nodes.new("ShaderNodeMath"); ab.operation = 'ABSOLUTE'
        nt.links.new(xin, ab.inputs[0]); xin = ab.outputs[0]

    def norm_sq(inp, c, r):
        sub = nt.nodes.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'
        nt.links.new(inp, sub.inputs[0]); sub.inputs[1].default_value = c
        div = nt.nodes.new("ShaderNodeMath"); div.operation = 'DIVIDE'
        nt.links.new(sub.outputs[0], div.inputs[0]); div.inputs[1].default_value = r
        sq = nt.nodes.new("ShaderNodeMath"); sq.operation = 'POWER'
        nt.links.new(div.outputs[0], sq.inputs[0]); sq.inputs[1].default_value = 2.0
        return sq.outputs[0]
    add = nt.nodes.new("ShaderNodeMath"); add.operation = 'ADD'
    nt.links.new(norm_sq(xin, cx, rx), add.inputs[0])
    nt.links.new(norm_sq(sep.outputs["Z"], cz, rz), add.inputs[1])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 1.0 - soft
    mr.inputs["From Max"].default_value = 1.0
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    mr.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(add.outputs[0], mr.inputs["Value"])
    return mr.outputs["Result"]


def mix_color(nt, fac, a, b):
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = 'RGBA'
    nt.links.new(fac, mx.inputs["Factor"])
    for sock, val in ((mx.inputs["A"], a), (mx.inputs["B"], b)):
        if isinstance(val, tuple):
            sock.default_value = val
        else:
            nt.links.new(val, sock)
    return mx.outputs["Result"]


SKIN_A = (0.42, 0.20, 0.11, 1)
SKIN_B = (0.50, 0.25, 0.14, 1)


def skin_bsdf(b):
    b.inputs["Roughness"].default_value = 0.58
    b.inputs["Subsurface Weight"].default_value = 0.4
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.25)
    b.inputs["Subsurface Scale"].default_value = 0.01
    b.inputs["Specular IOR Level"].default_value = 0.38


def make_materials():
    M = {}

    m, nt, b = node_mat("Skin")
    nt.links.new(noise_ramp(nt, SKIN_A, SKIN_B, 18.0), b.inputs["Base Color"])
    skin_bsdf(b)
    add_bump(nt, b, 900.0, 0.06, distance=0.002)
    M["skin"] = m

    # face: same skin plus lips, cheek flush and a faint shadow of stubble on the jaw
    m, nt, b = node_mat("FaceSkin")
    col = noise_ramp(nt, SKIN_A, SKIN_B, 18.0)
    cheeks = ellipse_mask(nt, 0.045, -0.025, 0.022, 0.020, soft=0.9, mirror=True)
    cm = nt.nodes.new("ShaderNodeMath"); cm.operation = 'MULTIPLY'; cm.inputs[1].default_value = 0.35
    nt.links.new(cheeks, cm.inputs[0])
    col = mix_color(nt, cm.outputs[0], col, (0.55, 0.22, 0.15, 1))
    # freckles: voronoi dots across the nose bridge and upper cheeks
    fz = ellipse_mask(nt, 0.0, -0.012, 0.060, 0.022, soft=0.8)
    vtc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 300.0
    vor.inputs["Randomness"].default_value = 1.0
    nt.links.new(vtc.outputs["Object"], vor.inputs["Vector"])
    dots = nt.nodes.new("ShaderNodeMapRange")
    dots.inputs["From Min"].default_value = 0.10
    dots.inputs["From Max"].default_value = 0.22
    dots.inputs["To Min"].default_value = 1.0
    dots.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], dots.inputs["Value"])
    sel = nt.nodes.new("ShaderNodeMath"); sel.operation = 'GREATER_THAN'
    sel.inputs[1].default_value = 0.55
    nt.links.new(vor.outputs["Color"], sel.inputs[0])
    fm = nt.nodes.new("ShaderNodeMath"); fm.operation = 'MULTIPLY'
    nt.links.new(dots.outputs["Result"], fm.inputs[0]); nt.links.new(sel.outputs[0], fm.inputs[1])
    fm2 = nt.nodes.new("ShaderNodeMath"); fm2.operation = 'MULTIPLY'
    nt.links.new(fm.outputs[0], fm2.inputs[0]); nt.links.new(fz, fm2.inputs[1])
    fm3 = nt.nodes.new("ShaderNodeMath"); fm3.operation = 'MULTIPLY'
    fm3.inputs[1].default_value = 0.8
    nt.links.new(fm2.outputs[0], fm3.inputs[0])
    col = mix_color(nt, fm3.outputs[0], col, (0.26, 0.10, 0.05, 1))
    lips = ellipse_mask(nt, 0.0, -0.0695, 0.0215, 0.0115, soft=0.45)
    col = mix_color(nt, lips, col, (0.42, 0.15, 0.12, 1))
    nt.links.new(col, b.inputs["Base Color"])
    skin_bsdf(b)
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.45
    rr.inputs["To Max"].default_value = 0.28
    nt.links.new(lips, rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    add_bump(nt, b, 1400.0, 0.05, distance=0.001)
    M["face"] = m

    m, nt, b = node_mat("Nail")
    b.inputs["Base Color"].default_value = (0.58, 0.36, 0.30, 1)
    b.inputs["Roughness"].default_value = 0.25
    M["nail"] = m

    m, nt, b = node_mat("Leather")
    nt.links.new(noise_ramp(nt, (0.075, 0.030, 0.012, 1), (0.16, 0.070, 0.030, 1), 5.0), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Coat Weight"].default_value = 0.12
    add_bump(nt, b, 260.0, 0.25, distance=0.003)
    M["leather"] = m

    m, nt, b = node_mat("DarkLeather")
    nt.links.new(noise_ramp(nt, (0.018, 0.011, 0.008, 1), (0.045, 0.028, 0.018, 1), 5.0), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.40
    b.inputs["Coat Weight"].default_value = 0.3
    add_bump(nt, b, 180.0, 0.2, distance=0.003)
    M["dark_leather"] = m

    m, nt, b = node_mat("Shearling")
    nt.links.new(noise_ramp(nt, (0.42, 0.33, 0.22, 1), (0.60, 0.50, 0.36, 1), 300.0, 0.3, 0.7), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.95
    b.inputs["Sheen Weight"].default_value = 1.0
    add_bump(nt, b, 500.0, 1.0, distance=0.004, kind="voronoi")
    M["shearling"] = m

    m, nt, b = node_mat("Pants")
    nt.links.new(noise_ramp(nt, (0.040, 0.048, 0.028, 1), (0.060, 0.068, 0.040, 1), 9.0), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.9
    b.inputs["Sheen Weight"].default_value = 0.5
    add_bump(nt, b, 1500.0, 0.3, distance=0.001)
    M["pants"] = m

    m, nt, b = node_mat("Shirt")
    b.inputs["Base Color"].default_value = (0.24, 0.21, 0.16, 1)
    b.inputs["Roughness"].default_value = 0.85
    b.inputs["Sheen Weight"].default_value = 0.5
    add_bump(nt, b, 900.0, 0.2, distance=0.0015)
    M["shirt"] = m

    m, nt, b = node_mat("Scarf")
    nt.links.new(noise_ramp(nt, (0.26, 0.018, 0.012, 1), (0.40, 0.035, 0.020, 1), 7.0), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.9
    b.inputs["Sheen Weight"].default_value = 1.0
    b.inputs["Sheen Tint"].default_value = (1.0, 0.55, 0.45, 1)
    add_bump(nt, b, 700.0, 0.3, distance=0.002)
    M["scarf"] = m

    m, nt, b = node_mat("Brass")
    b.inputs["Base Color"].default_value = (0.75, 0.50, 0.20, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.30
    add_bump(nt, b, 300.0, 0.15, distance=0.001)
    M["brass"] = m

    m, nt, b = node_mat("Steel")
    b.inputs["Base Color"].default_value = (0.50, 0.50, 0.52, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.35
    M["steel"] = m

    m, nt, b = node_mat("Glass")
    b.inputs["Base Color"].default_value = (0.35, 0.55, 0.50, 1)
    b.inputs["Roughness"].default_value = 0.03
    b.inputs["Transmission Weight"].default_value = 0.85
    b.inputs["IOR"].default_value = 1.5
    M["glass"] = m

    m, nt, b = node_mat("Hair")
    nt.links.new(noise_ramp(nt, (0.020, 0.010, 0.006, 1), (0.065, 0.032, 0.016, 1), 60.0), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.42
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.3
    M["hair"] = m

    m, nt, b = node_mat("EyeWhite")
    b.inputs["Base Color"].default_value = (0.70, 0.64, 0.60, 1)
    b.inputs["Roughness"].default_value = 0.25
    b.inputs["Subsurface Weight"].default_value = 0.3
    b.inputs["Subsurface Scale"].default_value = 0.003
    M["eye_white"] = m

    # iris: rings in object space around the eye's -Y axis, with fibre noise
    m, nt, b = node_mat("Iris")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = 'LENGTH'
    nt.links.new(comb.outputs[0], ln.inputs[0])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1500.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    madd = nt.nodes.new("ShaderNodeMath"); madd.operation = 'MULTIPLY_ADD'
    madd.inputs[1].default_value = 0.0006
    nt.links.new(noise.outputs["Fac"], madd.inputs[0])
    nt.links.new(ln.outputs["Value"], madd.inputs[2])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.004, 0.004, 0.004, 1)
    cr.elements[1].position = 0.0017
    cr.elements[1].color = (0.004, 0.004, 0.004, 1)
    for pos, c in ((0.0021, (0.30, 0.19, 0.05, 1)), (0.0032, (0.12, 0.17, 0.07, 1)),
                   (0.0050, (0.07, 0.12, 0.06, 1)), (0.0057, (0.012, 0.018, 0.012, 1)),
                   (0.0064, (0.62, 0.55, 0.50, 1)), (0.0110, (0.66, 0.56, 0.52, 1))):
        e = cr.elements.new(pos); e.color = c
    nt.links.new(madd.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.3
    M["iris"] = m

    m, nt, b = node_mat("Cornea")
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.376
    M["cornea"] = m

    m, nt, b = node_mat("Ground")
    b.inputs["Base Color"].default_value = (0.10, 0.095, 0.09, 1)
    b.inputs["Roughness"].default_value = 0.7
    M["ground"] = m
    return M


# ---------------------------------------------------------------- body

HEAD_CENTER = Vector((0.0, -0.008, 1.668))
HEAD_AXES = (0.080, 0.098, 0.117)
EYE_X, EYE_Z = 0.0318, 0.0118      # eye centers in head-local coordinates


def build_body(M):
    """Torso, legs and arms as one skin-modifier mesh (left side authored, mirrored)."""
    n = {
        "pelvis":   ((0.0, 0.004, 0.96), (0.165, 0.110)),
        "waist":    ((0.0, 0.000, 1.07), (0.140, 0.095)),
        "belly":    ((0.0, -0.006, 1.17), (0.150, 0.100)),
        "chest":    ((0.0, -0.012, 1.30), (0.172, 0.112)),
        "uchest":   ((0.0, -0.002, 1.40), (0.165, 0.102)),
        "neck0":    ((0.0, 0.006, 1.49), 0.062),
        "neck1":    ((0.0, 0.004, 1.585), 0.056),
        "hip_L":    ((0.092, 0.004, 0.91), 0.098),
        "thigh_L":  ((0.098, -0.004, 0.72), 0.080),
        "knee_L":   ((0.102, -0.012, 0.51), 0.056),
        "calf_L":   ((0.104, 0.008, 0.37), 0.060),
        "ankle_L":  ((0.106, 0.018, 0.090), 0.037),
        "heel_L":   ((0.106, 0.042, 0.040), 0.034),
        "ball_L":   ((0.110, -0.088, 0.030), 0.036),
        "toe_L":    ((0.112, -0.138, 0.026), 0.028),
        "clav_L":   ((0.115, 0.002, 1.462), 0.058),
        "shoulder_L": ((0.200, 0.008, 1.432), 0.062),
        "bicep_L":  ((0.240, 0.016, 1.300), 0.052),
        "elbow_L":  ((0.268, 0.030, 1.165), 0.040),
        "forearm_L": ((0.290, 0.010, 1.055), 0.044),
        "wrist_L":  ((0.312, -0.022, 0.935), 0.028),
    }
    e = [("pelvis", "waist"), ("waist", "belly"), ("belly", "chest"), ("chest", "uchest"),
         ("uchest", "neck0"), ("neck0", "neck1"),
         ("pelvis", "hip_L"), ("hip_L", "thigh_L"), ("thigh_L", "knee_L"), ("knee_L", "calf_L"),
         ("calf_L", "ankle_L"), ("ankle_L", "heel_L"), ("ankle_L", "ball_L"), ("ball_L", "toe_L"),
         ("uchest", "clav_L"), ("clav_L", "shoulder_L"), ("shoulder_L", "bicep_L"),
         ("bicep_L", "elbow_L"), ("elbow_L", "forearm_L"), ("forearm_L", "wrist_L")]
    nodes = mirror_nodes(n)
    body = skin_object("Body", nodes, mirror_edges(e), "pelvis", mat=M["skin"])
    return body, nodes


def build_hand(M, side, nodes):
    """Relaxed hand hanging beside the thigh, palm facing the body, fingers gently curled."""
    s = 1 if side == "L" else -1
    w = Vector(nodes["wrist_L"][0])

    def P(x, y, z):
        return (s * (w.x + x), w.y + y, w.z + z)
    h = {
        "wrist": (P(0, 0, 0), (0.026, 0.020)),
        "palm0": (P(0.004, -0.004, -0.030), (0.016, 0.030)),
        "palm1": (P(0.007, -0.002, -0.062), (0.015, 0.038)),
        "thumb0": (P(-0.006, -0.022, -0.026), 0.015),
        "thumb1": (P(-0.012, -0.040, -0.046), 0.0118),
        "thumb2": (P(-0.017, -0.050, -0.064), 0.0100),
        "thumb3": (P(-0.022, -0.054, -0.080), 0.0088),
    }
    e = [("wrist", "palm0"), ("palm0", "palm1"), ("palm0", "thumb0"), ("thumb0", "thumb1"),
         ("thumb1", "thumb2"), ("thumb2", "thumb3")]
    fingers = [("index", -0.031, (0.038, 0.024, 0.019), 0.0086, 10),
               ("middle", -0.011, (0.042, 0.027, 0.020), 0.0090, 16),
               ("ring",   0.009, (0.039, 0.025, 0.019), 0.0084, 22),
               ("pinky",  0.027, (0.031, 0.019, 0.016), 0.0074, 30)]
    tips = []
    for fname, fy, segs, r, curl in fingers:
        knuck_z = -0.082 + (0.005 if fname in ("index", "pinky") else 0) + (0.004 if fname == "pinky" else 0)
        k = f"{fname}0"
        h[k] = (P(0.009, fy, knuck_z), r * 1.08)
        e.append(("palm1", k))
        prev, z, x = k, knuck_z, 0.009
        ang = 0.0
        for i, L in enumerate(segs):
            ang += math.radians(curl + 10 * i)
            z -= L * math.cos(ang)
            x -= L * math.sin(ang)
            name = f"{fname}{i+1}"
            h[name] = (P(x, fy * (1.0 + 0.04 * (i + 1)), z), r * (0.94 - 0.08 * i))
            e.append((prev, name))
            prev = name
        tips.append((fname, Vector(h[prev][0]), ang))
    obj = skin_object(f"Hand_{side}", h, e, "wrist", mat=M["skin"], branch_smoothing=0.25)
    # fingernails on the back (outer) side of each fingertip
    for fname, tip, ang in tips:
        out = Vector((s * math.cos(ang), 0, math.sin(ang)))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8,
                                             location=tip + out * 0.0062 + Vector((0, 0, 0.004)))
        nail = active()
        nail.name = f"Nail_{fname}_{side}"
        nail.scale = (0.0018, 0.0060, 0.0085)
        nail.rotation_euler = (0, -s * ang, 0)
        smooth(nail)
        mat_to(nail, M["nail"])
    return obj


# ---------------------------------------------------------------- head

def head_shape(d):
    """Map a unit-sphere direction to the sculpted head surface (local coords, face toward -Y)."""
    a, b, c = HEAD_AXES
    dx, dy, dz = d
    x, y, z = a * dx, b * dy, c * dz
    if dy > 0:
        y *= 1.07                                     # occipital bulge
    if dz < 0.15:                                     # jaw/chin taper
        t = min(1.0, (0.15 - dz) / 1.15)
        x *= 1.0 - 0.38 * t ** 1.5
        if dy > 0:
            y *= 1.0 - 0.50 * t ** 1.1
    # temples slightly flattened
    x *= 1.0 - 0.05 * gauss(dz, 0.35, 0.2) * gauss(dy, -0.4, 0.3)
    front = max(0.0, -dy)
    if front > 0:
        y *= 1.0 - 0.12 * front ** 3                  # flatter face plane
        y -= 0.005 * front ** 2 * gauss(z, -0.07, 0.04)  # muzzle forward
    f = front ** 2
    ax = abs(x)
    disp = 0.0
    disp += 0.021 * gauss(x, 0, 0.0085) * gauss(z, -0.030, 0.0110)     # nose tip
    disp += 0.017 * gauss(x, 0, 0.0064) * gauss(z, -0.002, 0.016)     # nose bridge
    disp += 0.011 * gauss(ax, 0.0140, 0.0060) * gauss(z, -0.039, 0.0065)  # nostril wings
    disp -= 0.010 * gauss(x, 0, 0.0050) * gauss(z, -0.049, 0.0030)    # under nose
    disp += 0.011 * gauss(ax, 0.030, 0.020) * gauss(z, 0.034, 0.008)  # brow ridge
    disp -= 0.018 * gauss(ax, 0.031, 0.0135) * gauss(z, 0.012, 0.0110)  # eye sockets
    disp += 0.009 * gauss(ax, 0.052, 0.016) * gauss(z, -0.012, 0.015)  # cheekbones
    disp += 0.004 * gauss(x, 0, 0.0045) * gauss(z, -0.056, 0.006)     # philtrum
    disp += 0.0085 * gauss(x, 0, 0.0175) * gauss(z, -0.063, 0.0055)   # upper lip
    disp += 0.0095 * gauss(x, 0, 0.0155) * gauss(z, -0.077, 0.0058)   # lower lip
    disp -= 0.0060 * gauss(x, 0, 0.021) * gauss(z, -0.0695, 0.0016)   # mouth line
    disp -= 0.0035 * gauss(ax, 0.025, 0.004) * gauss(z, -0.068, 0.006)  # mouth corners
    disp -= 0.0060 * gauss(x, 0, 0.013) * gauss(z, -0.089, 0.0045)    # mentolabial fold
    disp += 0.0160 * gauss(x, 0, 0.018) * gauss(z, -0.103, 0.011)     # chin
    disp -= 0.0040 * gauss(ax, 0.040, 0.010) * gauss(z, -0.045, 0.012)  # nasolabial softness
    # eyelids: soft bulges above and below the eyeball that carve an almond-shaped opening
    ex = ax - EYE_X
    lid_w = gauss(ex, 0.0, 0.0105) * (1.0 + 0.25 * gauss(ex, -0.006, 0.004))
    upper_z = EYE_Z + 0.0068 - 0.0055 * (ex / 0.012) ** 2 + 0.0008 * (ex / 0.012)
    lower_z = EYE_Z - 0.0070 + 0.0042 * (ex / 0.012) ** 2 + 0.0010 * (ex / 0.012)
    lid = 0.0058 * lid_w * gauss(z, upper_z + 0.0028, 0.0034) * smoothstep(upper_z - 0.004, upper_z + 0.0005, z)
    lid += 0.0042 * lid_w * gauss(z, lower_z - 0.0022, 0.0030) * smoothstep(lower_z + 0.004, lower_z - 0.0005, z)
    lid -= 0.0018 * gauss(ex, 0.0, 0.012) * gauss(z, EYE_Z + 0.0150, 0.0022)   # upper lid crease
    disp += lid
    y -= disp * f
    return Vector((x, y, z))


def build_head(M):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=192, ring_count=112, radius=1.0)
    head = active()
    head.name = "Head"
    for v in head.data.vertices:
        v.co = head_shape(v.co.copy())
    smooth(head)
    head.location = HEAD_CENTER
    mat_to(head, M["face"])
    return head


def build_eyes(M, head, head_surf):
    parts = []
    holes = []
    R = 0.0122
    RL = R * 1.025                     # eyelid shell radius, hugging the eyeball
    AP_W, AP_TOP, AP_BOT = 0.93, 0.50, -0.50   # almond opening on the unit lid sphere
    for s in (1, -1):
        sx = s * EYE_X
        sz = EYE_Z
        # sink the eyeball so the head surface cuts it in a ~2 cm wide opening
        OPEN = 0.0110
        ys = []
        for dx, dz in ((OPEN, 0), (-OPEN, 0)):
            hit, _ = head_surf.hit(HEAD_CENTER + Vector((sx + dx, -0.5, sz + dz)), (0, 1, 0))
            ys.append(hit.y)
        rim_y = sum(ys) / len(ys)
        center = Vector((HEAD_CENTER.x + sx, rim_y + math.sqrt(R * R - OPEN * OPEN) * 0.9, HEAD_CENTER.z + sz))
        holes.append(center - HEAD_CENTER)
        tag = 'L' if s > 0 else 'R'
        rot = Euler((math.radians(-3), 0, math.radians(-6 * s + 4)))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=64, ring_count=32, location=center)
        eye = active()
        eye.name = f"Eye_{tag}"
        smooth(eye)
        mat_to(eye, M["iris"])     # one shader: pupil -> iris -> limbal ring -> sclera
        eye.rotation_euler = rot
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R * 1.02, segments=64, ring_count=32, location=center)
        cor = active()
        cor.name = f"Cornea_{tag}"
        for v in cor.data.vertices:
            if v.co.y < -R * 0.72:
                k = min(1.0, (-v.co.y - R * 0.72) / (R * 0.3))
                v.co.y -= 0.0011 * math.sin(k * math.pi / 2)
        smooth(cor)
        mat_to(cor, M["cornea"])
        cor.rotation_euler = rot

        # eyelids: a skin shell around the eyeball with an almond-shaped opening
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=48, location=center)
        lid = active()
        lid.name = f"Eyelids_{tag}"
        bm = bmesh.new(); bm.from_mesh(lid.data)
        kill = []
        for v in bm.verts:
            x, y, z = v.co
            xl = x * s           # + toward the temple
            if y > 0.25:
                kill.append(v); continue
            # almond aperture, lower outer corner slightly higher (tilt)
            u = xl / AP_W
            if abs(u) < 1.0 and y < 0:
                top = AP_TOP * (1 - u * u) ** 0.75 + 0.05 * u
                bot = AP_BOT * (1 - u * u) ** 0.9 + 0.04 * u
                if bot < z < top:
                    kill.append(v)
        bmesh.ops.delete(bm, geom=kill, context='VERTS')
        bm.to_mesh(lid.data); bm.free()
        lid.scale = (RL, RL, RL)
        lid.rotation_euler = (math.radians(-4), 0, 0)
        sol = lid.modifiers.new("Solidify", 'SOLIDIFY'); sol.thickness = 0.0010; sol.offset = 1
        add_subsurf(lid, 1)
        smooth(lid)
        mat_to(lid, M["face"])
        # lashes: short dark curve tufts along the upper lid rim
        lash_pts = []
        for k in range(17):
            u = -0.97 + 1.94 * k / 16
            top = AP_TOP * (1 - u * u) ** 0.75 + 0.05 * u + 0.03
            p = Vector((s * u * AP_W, -math.sqrt(max(0.0, 1 - (u * AP_W) ** 2 - top ** 2)), top)) * RL * 1.02
            lash_pts.append(center + Euler((math.radians(-4), 0, 0)).to_matrix() @ p)
        lash = bez_curve(f"Lashes_{tag}", lash_pts, 0.0007, M["hair"], resolution=1)
        parts += [eye, cor, lid, lash]
    return parts


def build_ears(M):
    ears = []
    for s in (1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=40, ring_count=24)
        ear = active()
        ear.name = f"Ear_{'L' if s > 0 else 'R'}"
        for v in ear.data.vertices:
            x, y, z = v.co
            if x > 0:   # concave outer face: bowl with a raised rim (helix)
                rim = (y * y + z * z)
                v.co.x = x * (0.15 + 0.85 * rim ** 2)
            if z < 0:   # narrower lobe
                v.co.y = y * (1.0 - 0.35 * (-z))
        ear.scale = (s * 0.012, 0.020, 0.031)
        ear.location = HEAD_CENTER + Vector((s * 0.0775, 0.012, -0.004))
        ear.rotation_euler = (math.radians(-10), 0, math.radians(s * -14))
        smooth(ear)
        add_subsurf(ear, 1)
        mat_to(ear, M["face"])
        ears.append(ear)
    return ears


def build_brows(M, head_surf):
    """Eyebrows grown from many short hairs along a brow path, pointing toward the temple."""
    cu = bpy.data.curves.new("Brows", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.00030
    cu.bevel_resolution = 0
    for s in (1, -1):
        for i in range(240):
            t = random.random()
            x = s * (0.011 + 0.038 * t)
            # brow arch with thickness that tapers to the tail
            thick = 0.0055 * (1.0 - 0.6 * t ** 1.5)
            z = 0.030 + 0.008 * math.sin(t * math.pi * 0.8) - 0.004 * t + random.uniform(-0.5, 0.5) * thick
            loc, nor = head_surf.hit(HEAD_CENTER + Vector((x, -0.5, z)), (0, 1, 0))
            if loc is None:
                continue
            # inner hairs point up, outer hairs sweep toward the temple
            ang = math.radians(75 - 65 * min(1.0, t * 1.6) + random.uniform(-12, 12))
            d = Vector((s * math.cos(ang), 0, math.sin(ang)))
            d -= nor * d.dot(nor)
            d.normalize()
            L = 0.0050 + 0.003 * random.random()
            p0 = loc + nor * 0.0002
            p1 = p0 + d * L * 0.5 + nor * 0.0005
            p2 = p0 + d * L + nor * 0.0001
            sp = cu.splines.new('POLY')
            sp.points.add(2)
            for k, q in enumerate((p0, p1, p2)):
                sp.points[k].co = (q.x, q.y, q.z, 1)
                sp.points[k].radius = 1.0 - 0.45 * k
    brows = link(bpy.data.objects.new("Brows", cu))
    cu.materials.append(M["hair"])
    return brows


# ---------------------------------------------------------------- hair

def scalp_keep(d):
    """Hair-covered region of the unit-sphere direction d."""
    x, y, z = d
    if z < -0.15:
        return y > 0.35 and z > -0.55 and abs(x) < 0.85   # nape
    if y < -0.40 and z < 0.60 - 0.30 * abs(x):              # forehead
        return False
    if abs(x) > 0.70 and z < 0.30 and y < 0.30:             # ears/temples
        return False
    return True


def shell(p, k):
    """Point on the head ellipsoid in direction p, scaled by k (1 = scalp)."""
    a, b, c = HEAD_AXES
    q = Vector((p.x / a, p.y / b, p.z / c))
    q.normalize()
    pt = Vector((a * q.x, b * q.y * (1.07 if q.y > 0 else 1.0), c * q.z))
    return pt * k


def build_hair(M):
    """Swept-back, slightly tousled short hair from clumped tapered strands that follow the skull."""
    hc = HEAD_CENTER
    cu = bpy.data.curves.new("Hair", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0026
    cu.bevel_resolution = 1
    cu.resolution_u = 4
    taper = bpy.data.curves.new("HairTaper", 'CURVE')
    tsp = taper.splines.new('BEZIER')
    tsp.bezier_points.add(2)
    for i, (x, y) in enumerate([(0, 0.9), (0.6, 0.7), (1.0, 0.05)]):
        bp = tsp.bezier_points[i]
        bp.co = (x, y, 0)
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    tob = link(bpy.data.objects.new("HairTaper", taper))
    tob.hide_render = tob.hide_viewport = True
    cu.taper_object = tob

    def flow_at(p):
        """Hair direction field: swept back over the top with a side part, down at sides and nape."""
        d = p.normalized()
        f = Vector((0.0, 1.0, 0.0))                                 # backwards
        f += Vector((0.0, 0.0, -1.0)) * smoothstep(0.55, -0.2, d.z) * 1.6  # fall down lower on the head
        f += Vector((math.copysign(1.0, d.x + 0.25), 0, 0)) * 0.35 * smoothstep(0.2, 0.9, d.z)  # part at x = -0.25
        f += Vector((0.35, 0, 0))                                    # overall sweep toward camera-right
        n = d
        f -= n * f.dot(n)                                           # tangent to the skull
        return f.normalized()

    golden = math.pi * (3 - math.sqrt(5))
    N_GUIDE = 1500
    guides = []
    for i in range(N_GUIDE):
        zz = 1 - (i / (N_GUIDE - 1)) * 2
        r = math.sqrt(max(0.0, 1 - zz * zz))
        th = golden * i
        d = Vector((math.cos(th) * r, math.sin(th) * r, zz))
        if scalp_keep(d):
            guides.append(d)

    count = 0
    for d in guides:
        L = 0.055 + 0.035 * random.random()
        if d.z > 0.5 and d.y < 0.2:
            L += 0.03                        # longer on top/front for the swept look
        if d.z < 0.1:
            L *= 0.6                         # short at the sides and nape
        lift = 0.004 + 0.016 * smoothstep(0.55, 0.95, d.z) * smoothstep(0.2, -0.6, d.y)  # volume at the front
        # a clump of child strands around each guide
        for c in range(5):
            jitter = Vector((random.gauss(0, 0.05), random.gauss(0, 0.05), random.gauss(0, 0.05)))
            p = shell(d + jitter, 1.0)
            pts = []
            steps = 6
            curl = random.uniform(-0.6, 0.6)
            layer = random.uniform(0.012, 0.06)
            for k in range(steps):
                t = k / (steps - 1)
                k_out = 1.0 + layer + lift * math.sin(t * math.pi * 0.9) + 0.025 * t
                pts.append(hc + shell(p, k_out))
                f = flow_at(p)
                side = f.cross(p.normalized())
                p = shell(p + (f + side * curl * t) * (L / (steps - 1)), 1.0)
            # tips peel off the skull a little
            last = pts[-1] - hc
            pts[-1] = hc + last * 1.02
            sp = cu.splines.new('BEZIER')
            sp.bezier_points.add(len(pts) - 1)
            for k, pt in enumerate(pts):
                bp = sp.bezier_points[k]
                bp.co = pt
                bp.handle_left_type = bp.handle_right_type = 'AUTO'
                bp.radius = 1.0 + 0.4 * random.random()
            count += 1
    hair = link(bpy.data.objects.new("Hair", cu))
    cu.materials.append(M["hair"])

    # scalp cap under the strands so no skin shows through
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    cap = active()
    cap.name = "HairCap"
    bm = bmesh.new(); bm.from_mesh(cap.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not scalp_keep(v.co)], context='VERTS')
    bm.to_mesh(cap.data); bm.free()
    for v in cap.data.vertices:
        v.co = shell(v.co.copy(), 1.012)
    cap.location = hc
    smooth(cap)
    mat_to(cap, M["hair"])
    return hair, cap


# ---------------------------------------------------------------- clothing

def scaled_nodes(nodes, keys, factor):
    out = {}
    for k in keys:
        co, r = nodes[k]
        out[k] = (co, (r[0] * factor, r[1] * factor) if isinstance(r, tuple) else r * factor)
    return out


V_POLY = [(0.0, 1.17), (0.060, 1.40), (0.090, 1.52), (0.090, 1.70), (-0.090, 1.70), (-0.090, 1.52), (-0.060, 1.40)]


def v_edge(z):
    """Half-width of the jacket's open V at height z."""
    pts = V_POLY[:3]
    if z <= pts[0][1]:
        return 0.0
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        if z <= z1:
            return x0 + (x1 - x0) * (z - z0) / (z1 - z0)
    return pts[-1][0]


def cloth_from_body(body, name, keep, offset, mat, thickness=0.004):
    """Clothing shell made by copying the body surface, trimming it and inflating along normals.

    keep(center) -> bool selects faces; offset(co) -> float gives the inflation at each vertex.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    bm = bmesh.new()
    bm.from_mesh(ev.to_mesh())
    ev.to_mesh_clear()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not keep(f.calc_center_median())], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.normal_update()
    moves = [(v, v.normal.copy() * offset(v.co)) for v in bm.verts]
    for v, d in moves:
        v.co += d
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    obj = link(bpy.data.objects.new(name, me))
    smooth(obj)
    if thickness:
        sol = obj.modifiers.new("Solidify", 'SOLIDIFY')
        sol.thickness = thickness
        sol.offset = -1
    mat_to(obj, mat)
    return obj


def is_arm(c, wrist_z=0.935):
    return abs(c.x) > 0.16 and c.z > 0.80 and not (abs(c.x) < 0.2 and c.z < 1.0)


def build_jacket(M, body):
    def keep(c):
        if c.z < 0.865:
            return False
        if is_arm(c):
            return c.z > 0.985
        if c.z > 1.50:
            return False
        if c.z > 1.465 and math.hypot(c.x, c.y - 0.008) < 0.085:
            return False                                   # neck hole
        if c.y < 0 and c.z > V_POLY[0][1] and abs(c.x) < v_edge(c.z):
            return False                                   # open V
        return True

    def offset(co):
        o = 0.016
        o += 0.020 * smoothstep(1.02, 0.87, co.z) * (0.0 if abs(co.x) > 0.24 else 1.0)   # hem flares over the hips
        o += 0.008 * smoothstep(1.30, 1.42, co.z) * smoothstep(0.12, 0.2, abs(co.x))      # padded shoulders
        return o
    jacket = cloth_from_body(body, "Jacket", keep, offset, M["leather"], thickness=0.006)
    tex = bpy.data.textures.new("JacketWrinkles", 'CLOUDS')
    tex.noise_scale = 0.05
    dm = jacket.modifiers.new("Wrinkles", 'DISPLACE')
    dm.texture = tex
    dm.strength = 0.004
    jacket.modifiers.move(1, 0)
    return jacket


def build_shirt(M, body):
    def keep(c):
        if is_arm(c) or c.z < V_POLY[0][1] - 0.03 or c.z > 1.545:
            return False
        if c.y < -0.02 and c.z > 1.44 and abs(c.x) < 0.012 + 0.25 * (c.z - 1.44):
            return False                                   # open shirt collar
        if c.z < 1.47 and (c.y > 0.0 or abs(c.x) > v_edge(c.z) + 0.04):
            return False                                   # hidden under the jacket
        return True
    return cloth_from_body(body, "Shirt", keep, lambda co: 0.006 + 0.004 * smoothstep(1.47, 1.53, co.z),
                           M["shirt"], thickness=0.002)


def build_pants(M, body):
    def keep(c):
        return 0.27 < c.z < 1.06 and not is_arm(c) and not (abs(c.x) > 0.2)

    def offset(co):
        o = 0.010
        o += 0.016 * gauss(co.z, 0.72, 0.12) * smoothstep(0.02, 0.09, abs(co.x))   # baggy cargo thighs
        o += 0.008 * smoothstep(0.45, 0.30, co.z)                                   # bloused over the boots
        return o
    pants = cloth_from_body(body, "Pants", keep, offset, M["pants"], thickness=0.003)
    # cloth folds: low-frequency noise displacement
    tex = bpy.data.textures.new("PantsFolds", 'CLOUDS')
    tex.noise_scale = 0.06
    dm = pants.modifiers.new("Folds", 'DISPLACE')
    dm.texture = tex
    dm.strength = 0.008
    dm.mid_level = 0.5
    smooth(pants)
    # cargo pockets on the outer thighs
    for s in (1, -1):
        surf = Surface(pants)
        loc, nor = surf.onto((s * 0.3, 0.0, 0.66), (s * 0.10, -0.004))
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc + nor * 0.008)
        pk = active()
        pk.name = f"CargoPocket_{s}"
        pk.scale = (0.016, 0.105, 0.12)
        pk.rotation_euler = Vector((1, 0, 0)).rotation_difference(nor * s).to_euler()
        bevel(pk, 0.007, 3)
        add_subsurf(pk, 1)
        mat_to(pk, M["pants"])
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc + nor * 0.017 + Vector((0, 0, 0.042)))
        fl = active()
        fl.name = f"CargoFlap_{s}"
        fl.scale = (0.008, 0.112, 0.045)
        fl.rotation_euler = pk.rotation_euler
        bevel(fl, 0.004, 3)
        mat_to(fl, M["pants"])
    return pants


def build_boots(M, nodes):
    for s, sx in (("L", 1), ("R", -1)):
        an = nodes[f"ankle_{s}"][0]
        bx = an[0]
        bn = {
            "top":   ((bx, an[1] - 0.004, 0.37), (0.070, 0.072)),
            "shin":  ((bx, an[1] - 0.002, 0.23), 0.060),
            "ankle": ((bx, an[1], 0.100), (0.050, 0.056)),
            "heel":  ((bx, an[1] + 0.034, 0.050), 0.048),
            "ball":  ((bx + sx * 0.004, -0.085, 0.042), (0.052, 0.040)),
            "toe":   ((bx + sx * 0.006, -0.150, 0.040), (0.042, 0.034)),
        }
        be = [("top", "shin"), ("shin", "ankle"), ("ankle", "heel"), ("ankle", "ball"), ("ball", "toe")]
        boot = skin_object(f"Boot_{s}", bn, be, "ankle", mat=M["dark_leather"])
        apply_modifiers(boot)
        for v in boot.data.vertices:
            if v.co.z < 0.022:
                v.co.z = 0.022
        smooth(boot)
        # sole fitted to the boot's footprint
        xs = [v.co.x for v in boot.data.vertices if v.co.z < 0.05]
        ys = [v.co.y for v in boot.data.vertices if v.co.z < 0.05]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=1.0, location=(cx, cy, 0.0125))
        sole = active()
        sole.name = f"Sole_{s}"
        sole.scale = ((max(xs) - min(xs)) / 2 + 0.006, (max(ys) - min(ys)) / 2 + 0.008, 0.025)
        for v in sole.data.vertices:
            if v.co.y > 0.3:
                v.co.x *= 0.9
        bevel(sole, 0.004, 3)
        mat_to(sole, M["dark_leather"])
        # heel block
        bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, max(ys) - 0.035, 0.012))
        hb = active()
        hb.scale = (0.07, 0.06, 0.024)
        bevel(hb, 0.004, 2)
        mat_to(hb, M["dark_leather"])

        surf = Surface(boot)
        # folded cuff at the top
        bpy.ops.mesh.primitive_torus_add(major_radius=0.071, minor_radius=0.010, location=(bx, an[1] - 0.004, 0.365))
        cuff = active()
        cuff.scale = (1.02, 1.04, 1.5)
        smooth(cuff)
        mat_to(cuff, M["leather"])
        # strap with buckle around the shaft
        pts = []
        for k in range(24):
            a = k / 24 * math.tau
            p, _ = surf.onto((bx + math.sin(a), an[1] - math.cos(a), 0.29), (bx, an[1]), 0.002)
            pts.append(p)
        strap_curve(f"BootStrap_{s}", pts, 0.009, 0.0018, M["leather"], cyclic=True)
        p, nrm = surf.onto((bx + sx, an[1], 0.29), (bx, an[1]), 0.004)
        bpy.ops.mesh.primitive_cube_add(size=1, location=p)
        bk = active()
        bk.scale = (0.004, 0.024, 0.022)
        bevel(bk, 0.0015)
        mat_to(bk, M["brass"])
        # laces crossing up the front, with brass eyelets
        for i in range(8):
            z0 = 0.105 + i * 0.030
            row = []
            for flip in (1, -1):
                p, nrm = surf.onto((bx + 0.016 * flip, an[1] - 1, z0), (bx, an[1]), 0.0)
                row.append((p, nrm))
                bpy.ops.mesh.primitive_torus_add(major_radius=0.0030, minor_radius=0.0010, location=p + nrm * 0.0008)
                eyelet = active()
                eyelet.rotation_euler = Vector((0, 0, 1)).rotation_difference(nrm).to_euler()
                mat_to(eyelet, M["brass"])
            if i > 0:
                for (a, an_), (b, bn_) in ((prev[0], row[1]), (prev[1], row[0])):
                    mid = (a + b) / 2 + (an_ + bn_).normalized() * 0.004
                    bez_curve(f"Lace_{s}_{i}", [a + an_ * 0.0015, mid, b + bn_ * 0.0015], 0.0013, M["shirt"], resolution=1)
            prev = row
        # bow at the top
        top = (prev[0][0] + prev[1][0]) / 2 + prev[0][1] * 0.004
        for flip in (1, -1):
            bez_curve(f"LaceBow_{s}_{flip}", [top, top + Vector((flip * 0.012, -0.004, 0.006)),
                                             top + Vector((flip * 0.016, -0.006, -0.004)), top], 0.0012,
                      M["shirt"], resolution=1)


def build_bracers(M, nodes):
    for s in "LR":
        el = Vector(nodes[f"forearm_{s}"][0])
        wr = Vector(nodes[f"wrist_{s}"][0])
        a = el.lerp(wr, 0.52)
        dirv = (wr - el).normalized()
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.047, depth=0.105, location=a)
        br = active()
        br.name = f"Bracer_{s}"
        br.rotation_euler = Vector((0, 0, 1)).rotation_difference(dirv).to_euler()
        for v in br.data.vertices:   # taper toward the wrist (-z local is toward the elbow)
            if v.co.z > 0:
                v.co.x *= 0.76; v.co.y *= 0.76
        bm = bmesh.new(); bm.from_mesh(br.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if len(f.verts) > 4], context='FACES')
        bm.to_mesh(br.data); bm.free()
        br.modifiers.new("Solidify", 'SOLIDIFY').thickness = 0.005
        bevel(br, 0.0018)
        smooth(br)
        mat_to(br, M["dark_leather"])
        for k in (-0.032, 0.0, 0.032):
            bpy.ops.mesh.primitive_torus_add(major_radius=0.0475 - (k + 0.05) * 0.06, minor_radius=0.0035,
                                             location=a + dirv * k)
            st = active()
            st.rotation_euler = br.rotation_euler
            smooth(st)
            mat_to(st, M["leather"])


# ---------------------------------------------------------------- accessories

def build_collar(M, jacket_surf):
    """Shearling collar rolled around the back of the neck and down the open V."""
    pts = []
    radii = []
    # down the right side of the V, around the back of the neck, down the left side
    for i in range(7):                              # right lapel bottom -> top
        t = i / 6
        z = 1.19 + 0.31 * t
        p, n = jacket_surf.front(-(v_edge(z) + 0.012), z, 0.006)
        pts.append(p)
        radii.append(0.35 + 0.85 * t)
    for i in range(1, 10):                          # behind the neck
        a = -math.pi * 0.62 + (i / 10) * math.pi * 1.24
        x = 0.098 * math.sin(a)
        y = 0.012 + 0.085 * math.cos(a)
        pts.append(Vector((x, y, 1.515 + 0.012 * math.cos(a))))
        radii.append(1.25)
    for i in range(7):                              # left lapel top -> bottom
        t = 1 - i / 6
        z = 1.19 + 0.31 * t
        p, n = jacket_surf.front(v_edge(z) + 0.012, z, 0.006)
        pts.append(p)
        radii.append(0.35 + 0.85 * t)
    keep = [i for i, p in enumerate(pts) if p is not None]
    pts = [pts[i] for i in keep]
    radii = [radii[i] for i in keep]
    collar = bez_curve("Collar", pts, 0.0, M["shearling"], radii=radii)
    collar.data.bevel_mode = 'OBJECT'
    collar.data.bevel_object = profile("CollarProfile", [(-0.010, -0.022), (0.006, -0.016), (0.010, 0.0),
                                                         (0.006, 0.016), (-0.010, 0.022), (-0.012, 0.0)],
                                       smooth_profile=True)
    collar.data.twist_mode = 'Z_UP'
    apply_modifiers(collar)
    collar.name = "Collar"
    smooth(collar)
    tex = bpy.data.textures.new("Fleece", 'VORONOI')
    tex.noise_scale = 0.004
    add_subsurf(collar, 1)
    dm = collar.modifiers.new("Fleece", 'DISPLACE')
    dm.texture = tex
    dm.strength = 0.0025
    return collar


def build_belt(M, pants_surf):
    pts = []
    for k in range(40):
        a = k / 40 * math.tau
        p, _ = pants_surf.onto((math.sin(a), -math.cos(a), 0.985), (0, 0.0), 0.004)
        pts.append(p)
    belt = strap_curve("Belt", pts, 0.020, 0.0035, M["dark_leather"], cyclic=True)
    front, n = pants_surf.front(0.0, 0.985, 0.008)
    # buckle: rounded rectangular frame + prong
    bpy.ops.mesh.primitive_torus_add(major_radius=0.026, minor_radius=0.0045, major_segments=4,
                                     minor_segments=10, location=front,
                                     rotation=(math.radians(90), math.radians(45), 0))
    buckle = active()
    buckle.name = "BeltBuckle"
    buckle.scale = (1.2, 0.85, 1.0)
    bevel(buckle, 0.003, 3)
    mat_to(buckle, M["brass"])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0028, depth=0.042, location=front + Vector((0, -0.002, 0)),
                                        rotation=(0, math.radians(90), 0))
    mat_to(active(), M["brass"])
    # belt loops / keeper
    bpy.ops.mesh.primitive_cube_add(size=1, location=front + Vector((-0.040, 0.003, 0)))
    kp = active()
    kp.scale = (0.012, 0.006, 0.046)
    bevel(kp, 0.002)
    mat_to(kp, M["dark_leather"])
    return pts


def pouch(M, name, loc, normal, size):
    rot = Vector((0, -1, 0)).rotation_difference(-normal.normalized() * -1 if False else normal).to_euler()
    rot = Vector((0, -1, 0)).rotation_difference(Vector((normal.x, normal.y, 0)).normalized()).to_euler()
    center = loc + Vector((normal.x, normal.y, 0)).normalized() * size[1] * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1, location=center, rotation=rot)
    p = active()
    p.name = name
    p.scale = size
    bevel(p, 0.008, 4)
    add_subsurf(p, 1)
    mat_to(p, M["leather"])
    m = rot.to_matrix()
    bpy.ops.mesh.primitive_cube_add(size=1, location=center + m @ Vector((0, -size[1] * 0.08, size[2] * 0.30)), rotation=rot)
    f = active()
    f.name = name + "_Flap"
    f.scale = (size[0] * 1.06, size[1] * 1.12, size[2] * 0.45)
    bevel(f, 0.006, 3)
    mat_to(f, M["dark_leather"])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0065, depth=0.004,
                                        location=center + m @ Vector((0, -size[1] * 0.64, size[2] * 0.14)),
                                        rotation=(rot.to_matrix() @ Matrix.Rotation(math.radians(90), 3, 'X')).to_euler())
    mat_to(active(), M["brass"])


def build_gear(M, jacket_surf, pants_surf):
    # pouches on the belt
    for s, ang in ((1, 50), (-1, 46), (-1, 112)):
        a = math.radians(ang) * s
        loc, n = pants_surf.onto((math.sin(a), -math.cos(a), 0.965), (0, 0), 0.006)
        pouch(M, f"Pouch_{s}_{ang}", loc, n, (0.068, 0.034, 0.072))

    # satchel strap: right shoulder over the chest to the left hip
    raw = [(-0.12, 1.47), (-0.095, 1.40), (-0.05, 1.31), (0.0, 1.24), (0.06, 1.15), (0.12, 1.06)]
    pts = []
    for x, z in raw:
        p, n = jacket_surf.front(x, z, 0.012)
        if p is None:
            continue
        pts.append(p)
    # over the shoulder top and down the back
    pts = [Vector((-0.115, 0.10, 1.44)), Vector((-0.125, 0.03, 1.505))] + pts
    pts += [Vector((0.205, -0.050, 0.99)), Vector((0.230, -0.01, 0.96))]
    strap_curve("SatchelStrap", pts, 0.016, 0.0025, M["dark_leather"], tilt=0)
    p, n = jacket_surf.front(-0.030, 1.28, 0.016)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.016, minor_radius=0.003, major_segments=4,
                                     location=p, rotation=(math.radians(90), math.radians(45), math.radians(-30)))
    bpy.context.active_object.scale = (1.0, 1.3, 1.0)
    mat_to(active(), M["brass"])
    # the satchel itself, riding behind the left hip
    ang = math.radians(118)
    out = Vector((math.sin(ang), -math.cos(ang), 0))
    rot = Vector((0, -1, 0)).rotation_difference(out).to_euler()
    m = rot.to_matrix()
    c = Vector((0, 0.01, 0.90)) + Vector((out.x * 0.235, out.y * 0.17, 0))
    bpy.ops.mesh.primitive_cube_add(size=1, location=c, rotation=rot)
    bag = active()
    bag.name = "Satchel"
    bag.scale = (0.19, 0.06, 0.15)
    bevel(bag, 0.02, 5)
    add_subsurf(bag, 1)
    mat_to(bag, M["leather"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=c + m @ Vector((0, -0.033, 0.035)), rotation=rot)
    flap = active()
    flap.name = "Satchel_Flap"
    flap.scale = (0.196, 0.010, 0.095)
    bevel(flap, 0.012, 4)
    mat_to(flap, M["dark_leather"])
    for dx in (-0.05, 0.05):
        bpy.ops.mesh.primitive_cube_add(size=1, location=c + m @ Vector((dx, -0.040, 0.0)), rotation=rot)
        b = active()
        b.scale = (0.014, 0.004, 0.018)
        bevel(b, 0.002)
        mat_to(b, M["brass"])
    pts[-2:] = [c + m @ Vector((-0.09, -0.02, 0.07)) + Vector((0, 0, 0.03)), c + m @ Vector((-0.095, 0, 0.07))]
    strap_curve("SatchelStrapEnd", [pts[-3], pts[-2], pts[-1]], 0.016, 0.0025, M["dark_leather"])

    # brass buttons down the closed part of the jacket, offset placket to the left
    for i, z in enumerate((1.14, 1.07, 1.00, 0.93)):
        p, n = jacket_surf.front(0.018, z, 0.001)
        if p is None:
            continue
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.0078, depth=0.004, location=p,
                                            rotation=Vector((0, 0, 1)).rotation_difference(n).to_euler())
        btn = active()
        btn.name = f"Button_{i}"
        bevel(btn, 0.0016, 2)
        mat_to(btn, M["brass"])
    # placket seam
    seam = []
    for z in (0.90, 0.98, 1.06, 1.14, 1.18):
        p, n = jacket_surf.front(0.005, z, 0.0005)
        if p is not None:
            seam.append(p)
    bez_curve("Placket", seam, 0.0018, M["dark_leather"], resolution=1)
    # chest pocket with flap on the wearer's left
    p, n = jacket_surf.front(0.095, 1.31, 0.0)
    bpy.ops.mesh.primitive_cube_add(size=1, location=p + n * 0.003)
    cp = active()
    cp.name = "ChestPocketFlap"
    cp.scale = (0.075, 0.006, 0.030)
    cp.rotation_euler = Vector((0, -1, 0)).rotation_difference(n).to_euler()
    bevel(cp, 0.004, 3)
    mat_to(cp, M["dark_leather"])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.003, location=p + n * 0.0075 + Vector((0, 0, -0.004)),
                                        rotation=Vector((0, 0, 1)).rotation_difference(n).to_euler())
    mat_to(active(), M["brass"])
    # a small brass wing pin above the pocket (sky-pirate insignia)
    p, n = jacket_surf.front(0.095, 1.36, 0.004)
    for side in (1, -1):
        for k in range(3):
            q = p + Vector((side * (0.008 + k * 0.007), 0, 0.004 - k * 0.0025))
            bpy.ops.mesh.primitive_cube_add(size=1, location=q)
            f = active()
            f.scale = (0.008, 0.002, 0.003 - k * 0.0006)
            f.rotation_euler = (0, side * math.radians(-15 - 10 * k), 0)
            bevel(f, 0.0008)
            mat_to(f, M["brass"])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.004, location=p)
    mat_to(active(), M["brass"])


def build_scarf(M):
    """Red scarf: two bunched loops around the neck, a knot, and a tail falling over the chest."""
    neck_c = Vector((0.0, 0.004, 1.545))
    for loop, (rx, ry, z0, phase) in enumerate(((0.074, 0.070, 1.548, 0.0), (0.078, 0.073, 1.512, 1.3))):
        pts = []
        for i in range(20):
            a = i / 20 * math.tau
            r = 1.0 + 0.06 * math.sin(3 * a + phase)
            pts.append(Vector((rx * r * math.sin(a), neck_c.y - ry * r * math.cos(a),
                               z0 + 0.010 * math.cos(a) + 0.006 * math.sin(2 * a + phase))))
        sc = bez_curve(f"Scarf_Loop{loop}", pts, 0.0, M["scarf"], cyclic=True)
        sc.data.bevel_mode = 'OBJECT'
        sc.data.bevel_object = profile(f"ScarfLoopProf{loop}", [(-0.012, -0.020), (0.0, -0.024), (0.012, -0.020),
                                                               (0.014, 0.0), (0.012, 0.020), (0.0, 0.024),
                                                               (-0.012, 0.020), (-0.014, 0.0)], smooth_profile=True)
        sc.data.twist_mode = 'Z_UP'
        apply_modifiers(sc)
        smooth(sc)
        tex = bpy.data.textures.new(f"ScarfBunch{loop}", 'CLOUDS')
        tex.noise_scale = 0.025
        add_subsurf(sc, 1)
        dm = sc.modifiers.new("Bunch", 'DISPLACE')
        dm.texture = tex
        dm.strength = 0.009
    # knot
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.026, location=(0.026, -0.080, 1.505))
    knot = active()
    knot.name = "Scarf_Knot"
    knot.scale = (1.2, 0.8, 1.0)
    tex = bpy.data.textures.new("KnotTex", 'CLOUDS')
    tex.noise_scale = 0.015
    add_subsurf(knot, 2)
    dm = knot.modifiers.new("Displace", 'DISPLACE'); dm.texture = tex; dm.strength = 0.008
    smooth(knot)
    mat_to(knot, M["scarf"])
    # two tails, one longer, hanging down the chest inside the V
    for t_i, (pts, w) in enumerate((
            ([Vector((0.030, -0.092, 1.495)), Vector((0.045, -0.125, 1.440)), Vector((0.050, -0.140, 1.380)),
              Vector((0.044, -0.146, 1.320)), Vector((0.040, -0.146, 1.255))], 0.026),
            ([Vector((0.020, -0.090, 1.495)), Vector((0.006, -0.122, 1.445)), Vector((-0.004, -0.136, 1.400)),
              Vector((-0.008, -0.142, 1.350))], 0.022))):
        tail = bez_curve(f"Scarf_Tail{t_i}", pts, 0.0, M["scarf"],
                         radii=[1.0 - 0.1 * i / len(pts) for i in range(len(pts))],
                         tilts=[math.radians(10 * i * (1 if t_i else -1)) for i in range(len(pts))])
        tail.data.bevel_mode = 'OBJECT'
        tail.data.bevel_object = profile(f"ScarfTailProf{t_i}", [(-w, 0), (0, 0.006), (w, 0), (0, -0.006)],
                                         smooth_profile=True)
        tail.data.twist_mode = 'Z_UP'
        apply_modifiers(tail)
        smooth(tail)
        # fringe at the end
        end = pts[-1]
        for k in range(9):
            q = end + Vector((-w + w * 2 * k / 8, -0.004, 0.002))
            bez_curve(f"Fringe_{t_i}_{k}", [q, q + Vector((0.002 * (k - 4) * 0.3, -0.002, -0.020 - 0.004 * (k % 3)))],
                      0.0014, M["scarf"], resolution=1)


def build_goggles(M, cap_surf):
    """Brass aviator goggles pushed up onto the hair, strap around the head."""
    hc = HEAD_CENTER
    for s in (1, -1):
        loc, nor = cap_surf.hit(hc + Vector((s * 0.035 * 4, -0.4, 0.088 * 4)) * 1.0 + Vector((0, 0, 0)), Vector((0, 0, 0)) - Vector((s * 0.035 * 4, -0.4, 0.088 * 4)))
        # simpler and robust: shoot from a point in front/above toward the head center
        o = hc + Vector((s * 0.040, -0.35, 0.24))
        loc, nor = cap_surf.hit(o, (hc + Vector((s * 0.030, -0.02, 0.06))) - o)
        nor = (nor + Vector((0, -0.25, 0.1))).normalized()
        rot = Vector((0, 0, 1)).rotation_difference(nor).to_euler()
        rm = rot.to_matrix()
        base = loc + nor * 0.004
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.025, depth=0.018, location=base + nor * 0.009, rotation=rot)
        cup = active()
        cup.name = f"GoggleCup_{s}"
        for v in cup.data.vertices:          # slightly conical cup
            if v.co.z > 0:
                v.co.x *= 0.92; v.co.y *= 0.92
        bm = bmesh.new(); bm.from_mesh(cup.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if len(f.verts) > 4], context='FACES')
        bm.to_mesh(cup.data); bm.free()
        cup.modifiers.new("Solidify", 'SOLIDIFY').thickness = 0.003
        bevel(cup, 0.001)
        smooth(cup)
        mat_to(cup, M["brass"])
        top = base + nor * 0.018
        bpy.ops.mesh.primitive_torus_add(major_radius=0.0235, minor_radius=0.0033, location=top, rotation=rot)
        smooth(active()); mat_to(active(), M["brass"])
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.0215, depth=0.002, location=top - nor * 0.001, rotation=rot)
        lens = active()
        lens.name = f"GoggleLens_{s}"
        mat_to(lens, M["glass"])
        bpy.ops.mesh.primitive_torus_add(major_radius=0.0245, minor_radius=0.0055, location=base, rotation=rot)
        smooth(active()); mat_to(active(), M["dark_leather"])
        for k in range(8):
            a = k / 8 * math.tau
            p = top + rm @ Vector((math.cos(a) * 0.0236, math.sin(a) * 0.0236, 0.0028))
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0015, segments=8, ring_count=6, location=p)
            mat_to(active(), M["steel"])
        if s == 1:
            left_top = top
        else:
            right_top = top
    mid = (left_top + right_top) / 2 + Vector((0, -0.006, 0.004))
    bez_curve("GoggleBridge", [left_top + (right_top - left_top).normalized() * 0.024, mid,
                               right_top + (left_top - right_top).normalized() * 0.024], 0.0028, M["brass"])
    # strap: around the head on a tilted ring through the cups
    pts = []
    for i in range(30):
        a = (i / 29) * math.pi * 1.55 - math.pi * 0.775
        dvec = Vector((math.sin(a), math.cos(a), 0.0))
        z = 0.070 - 0.055 * (0.5 + 0.5 * math.cos(a))
        o = hc + dvec * 0.4 + Vector((0, 0, z))
        loc, nor = cap_surf.hit(o, hc + Vector((0, 0, z)) - o)
        if loc is not None:
            pts.append(loc + nor * 0.004)
    strap_curve("GoggleStrap", pts, 0.0035, 0.012, M["dark_leather"])


# ---------------------------------------------------------------- scene

def setup_scene(view, samples, res):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.view_settings.exposure = -1.4

    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    wn = world.node_tree
    bg = wn.nodes["Background"]
    grad = wn.nodes.new("ShaderNodeTexGradient")
    mapn = wn.nodes.new("ShaderNodeMapping")
    tc = wn.nodes.new("ShaderNodeTexCoord")
    ramp = wn.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.020, 0.018, 0.016, 1)
    ramp.color_ramp.elements[1].color = (0.12, 0.15, 0.20, 1)
    mapn.inputs["Rotation"].default_value = (0, math.radians(-90), 0)
    wn.links.new(tc.outputs["Generated"], mapn.inputs["Vector"])
    wn.links.new(mapn.outputs["Vector"], grad.inputs["Vector"])
    wn.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    wn.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.5

    # curved cyclorama backdrop
    bm = bmesh.new()
    W, D, H, R = 6.0, 4.0, 4.0, 1.4
    prof = [(y, 0.0) for y in (-D, 0.0)]
    for k in range(1, 13):
        a = k / 12 * math.pi / 2
        prof.append((R * math.sin(a), R - R * math.cos(a)))
    prof.append((R, H))
    rows = []
    for x in (-W, W):
        rows.append([bm.verts.new((x, y + 1.0, z)) for y, z in prof])
    for i in range(len(prof) - 1):
        bm.faces.new((rows[0][i], rows[1][i], rows[1][i + 1], rows[0][i + 1]))
    me = bpy.data.meshes.new("Backdrop")
    bm.to_mesh(me); bm.free()
    bd = link(bpy.data.objects.new("Backdrop", me))
    smooth(bd)
    mat_to(bd, bpy.data.materials["Ground"])

    def area(name, loc, energy, size, color, target=(0, 0, 1.1)):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.size = size
        ld.color = color
        lo = link(bpy.data.objects.new(name, ld))
        lo.location = loc
        lo.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return lo
    area("Key", (-1.9, -2.3, 2.7), 420, 2.4, (1.0, 0.90, 0.78), target=(0, 0, 1.3))
    area("Fill", (2.4, -2.0, 1.4), 90, 2.5, (0.70, 0.82, 1.0))
    area("Rim", (1.3, 1.8, 2.3), 320, 0.8, (1.0, 0.80, 0.58), target=(0, 0, 1.45))
    area("Rim2", (-1.5, 1.6, 2.0), 180, 0.8, (0.6, 0.75, 1.0), target=(0, 0, 1.2))
    area("Face", (-0.7, -1.4, 1.75), 25, 0.5, (1.0, 0.92, 0.85), target=(0, 0, 1.66))

    cam_d = bpy.data.cameras.new("Cam")
    cam = link(bpy.data.objects.new("Cam", cam_d))
    sc.camera = cam
    views = {
        "full":     ((-1.35, -3.70, 1.20), (0.02, 0.0, 0.89), 58),
        "portrait": ((-0.48, -1.30, 1.68), (0.0, -0.02, 1.60), 110),
        "face":     ((-0.30, -0.95, 1.69), (0.0, -0.03, 1.66), 135),
        "back":     ((2.1, 3.4, 1.35), (0.0, 0.0, 0.92), 52),
        "side":     ((3.6, -0.6, 1.10), (0.0, 0.0, 0.92), 52),
    }
    loc, target, lens = views[view]
    cam.location = loc
    cam_d.lens = lens
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    if view in ("portrait", "face"):
        cam_d.dof.use_dof = True
        cam_d.dof.focus_distance = (Vector(loc) - (HEAD_CENTER + Vector((0, -0.09, 0.01)))).length
        cam_d.dof.aperture_fstop = 5.6


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"--render": None, "--view": "full", "--samples": "128", "--save": None, "--res": "1080x1440"}
    for i in range(0, len(argv) - 1, 2):
        opts[argv[i]] = argv[i + 1]

    clear_scene()
    M = make_materials()
    body, nodes = build_body(M)
    build_hand(M, "L", nodes)
    build_hand(M, "R", nodes)
    head = build_head(M)
    head_surf = Surface(head)
    build_eyes(M, head, head_surf)
    build_ears(M)
    build_brows(M, head_surf)
    hair, cap = build_hair(M)
    jacket = build_jacket(M, body)
    build_shirt(M, body)
    pants = build_pants(M, body)
    build_boots(M, nodes)
    build_bracers(M, nodes)
    jacket_surf = Surface(jacket)
    pants_surf = Surface(pants)
    build_collar(M, jacket_surf)
    build_belt(M, pants_surf)
    build_gear(M, jacket_surf, pants_surf)
    build_scarf(M)
    build_goggles(M, Surface(cap))
    setup_scene(opts["--view"], int(opts["--samples"]), tuple(int(v) for v in opts["--res"].split("x")))

    if opts["--save"]:
        bpy.ops.wm.save_as_mainfile(filepath=opts["--save"])
    if opts["--render"]:
        bpy.context.scene.render.filepath = opts["--render"]
        bpy.ops.render.render(write_still=True)


main()
