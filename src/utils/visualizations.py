import numpy as np


def ijk0_to_xyz(ijk0, a=0.98, b=0.866):
    """
    ijk0: (N,3) array with 0-based indices [i0, j0, k0]
    returns: (N,3) array [x, y, z] matching the MATLAB mapping
    """
    ijk0 = np.asarray(ijk0)
    i0, j0, k0 = ijk0[:, 0], ijk0[:, 1], ijk0[:, 2]

    # convert to MATLAB-style 1-based indices
    i = i0 + 1
    j = j0 + 1
    k = k0 + 1

    km = k % 3

    # Z is always the same
    z = a * b * (2 * k - 1)

    # allocate x,y
    x = np.empty_like(z, dtype=np.float64)
    y = np.empty_like(z, dtype=np.float64)

    # Case km == 1
    m1 = (km == 1)
    x[m1] = a * b * (3 * i[m1] - 2) * (2/3)
    y[m1] = a * ((2 * j[m1] - 1) - ((i[m1] + 1) % 2))

    # Case km == 2
    m2 = (km == 2)
    x[m2] = a * b * (3 * i[m2] - 4) * (2/3)
    y[m2] = a * ((2 * j[m2] - 1) - ((i[m2] + 1) % 2))

    # Case km == 0
    m0 = (km == 0)
    x[m0] = a * b * (3 * i[m0] - 3) * (2/3)
    y[m0] = a * ((2 * j[m0] - 1) - (i[m0] % 2))

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz

def plot_cube_vista(data, color_dict, exclude_type=None, node_indices=None, output_file="outputs/figures/atomic_lattice.png", screenshot=True):
    import pyvista as pv

    nodes = ijk0_to_xyz(data.pos.numpy())

    atomic_type = np.argmax(data.x.numpy(), axis=1)

    base_sphere = pv.Sphere(radius=0.8, theta_resolution=40, phi_resolution=40)

    p = pv.Plotter(window_size=[3000, 3000], off_screen=False)

    actors = {}
    for t in np.unique(atomic_type):
        mask = (atomic_type == t)
        pos_t = nodes[mask]

        if pos_t.shape[0] == 0:
            continue

        mesh_t = pv.PolyData(pos_t)
        glyphs_t = mesh_t.glyph(geom=base_sphere)
        rgb = np.array(pv.Color(color_dict[int(t)]).float_rgb)
        glyphs_t.point_data["RGB"] = np.tile(rgb, (glyphs_t.n_points, 1))

        actor = p.add_mesh(glyphs_t, scalars="RGB", rgb=True, name=f"type_{t}",  smooth_shading=True, specular=0.3, specular_power=40, ambient=0.25)
        actors[int(t)] = actor

    # Toggle visibility of atom types
    if exclude_type is not None:
        for t in exclude_type:
            actors[t].SetVisibility(False)

    # --- highlight selected nodes with a wireframe shell ---
    if node_indices is not None and len(node_indices) > 0:
        selected_nodes = nodes[node_indices]
        sel_mesh = pv.PolyData(selected_nodes)

        # slightly larger sphere than base_sphere so it sits outside
        outline_sphere = pv.Sphere(
            radius=0.95,              # <-- must be > 0.8
            theta_resolution=40,
            phi_resolution=40
        )
        outline_glyphs = sel_mesh.glyph(geom=outline_sphere)

        p.add_mesh(
            outline_glyphs,
            style="wireframe",
            color="yellow",
            line_width=2,
            opacity=1.0,
            lighting=False
        )

    p.camera.Azimuth(80)
    p.camera.Elevation(-10)

    if screenshot:
        p.show(auto_close=False)
        p.screenshot(output_file, window_size=[3000, 3000], transparent_background=True)
        p.close()

def plot_scalar_map(data, scalar_field, exclude_type=None, output_file="outputs/figures/att_map.png", screenshot=True):
    import pyvista as pv

    scalar_max, scalar_min = np.max(scalar_field), np.min(scalar_field)
    scaled_weights = (scalar_field - scalar_min) / (scalar_max - scalar_min)
    vmax, vmin = scaled_weights.max(), scaled_weights.min()

    nodes = ijk0_to_xyz(data.pos.numpy())
    atomic_type = np.argmax(data.x.numpy(), axis=1)

    base_sphere = pv.Sphere(radius=0.8, theta_resolution=40, phi_resolution=40)

    p = pv.Plotter(window_size=[3000, 3000], off_screen=False)

    actors = {}
    for t in np.unique(atomic_type):
        mask = atomic_type == t
        pos_t = nodes[mask]
        mesh_t = pv.PolyData(pos_t)
        glyphs_t = mesh_t.glyph(geom=base_sphere)
        weights_t = scaled_weights[mask]
        glyphs_t.point_data['Scalar'] = np.repeat(weights_t, base_sphere.n_points, axis=0)

        actor = p.add_mesh(glyphs_t, scalars='Scalar', cmap='viridis', name=f"type_{t}",  smooth_shading=True, specular=0.3, specular_power=40, ambient=0.25, clim=[vmin, vmax], show_scalar_bar=False)
        actors[int(t)] = actor

    # Toggle visibility of atom types
    if exclude_type is not None:
        for t in exclude_type:
            actors[t].SetVisibility(False)

    p.camera.Azimuth(80)
    p.camera.Elevation(-10)

    if screenshot:
        p.show(auto_close=False)
        p.screenshot(output_file, window_size=[3000, 3000], transparent_background=True)
        p.close()
