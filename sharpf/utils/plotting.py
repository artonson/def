import copy
from base64 import b64decode

import k3d
import numpy as np
import trimesh
import trimesh.transformations as tt
from IPython.display import Image
import randomcolor

import sharpf.utils.abc_utils.abc.feature_utils as def_feature_utils
#
#
# def display_sharpness(
#         mesh: trimesh.base.Trimesh = None,
#         samples: np.ndarray = None,
#         distances=None,
#         directions=None,
#         sharp_vertices=None,
#         sharp_curves=None,
#         directions_width=0.0025,
#         plot_meshvert=True,
#         samples_color=0x0000ff,
#         samples_psize=0.002,
#         mesh_color=0xbbbbbb,
#         meshvert_color=0x666666,
#         meshvert_psize=0.0025,
#         sharpvert_color=0xff0000,
#         sharpvert_psize=0.0025,
#         sharpcurve_color=None,
#         sharpcurve_width=0.0025,
#         plot_height=768,
#         plot_bgcolor=0xffffff,
#         distances_cmap=k3d.colormaps.matplotlib_color_maps.coolwarm_r,
#         plot=None,
#         display=True,
#         max_distance_to_feature=1.0):
#
#     if None is plot:
#         plot = k3d.plot(
#             grid_visible=True,
#             height=plot_height,
#             background_color=plot_bgcolor,
#             camera_auto_fit=True)
#
#     if None is not mesh:
#         plot += k3d.mesh(
#             mesh.vertices,
#             mesh.faces,
#             color=mesh_color,
#             side='both',
#             flat_shading=False)
#
#         if plot_meshvert:
#             k3d_points = k3d.points(mesh.vertices,
#                                     point_size=meshvert_psize, color=meshvert_color)
#             plot += k3d_points
#             k3d_points.shader = 'flat'
#
#     if None is not samples:
#         colors = None
#         if None is not samples_distances and not np.all(samples_distances == 1.0):
#             colors = k3d.helpers.map_colors(
#                 samples_distances, distances_cmap, [0, max_distance_to_feature]
#             ).astype(np.uint32)
#             k3d_points = k3d.points(samples, point_size=samples_psize, colors=colors)
#         else:
#             v = -np.array([0., 1., 0.])
#             max_dist = np.max(np.dot(samples, v))
#             min_dist = np.min(np.dot(samples, v))
#             colors = k3d.helpers.map_colors(
#                 np.dot(samples, v), k3d.colormaps.matplotlib_color_maps.viridis, [min_dist, max_dist]
#             ).astype(np.uint32)
#             k3d_points = k3d.points(samples, point_size=samples_psize, colors=colors)
#
#         plot += k3d_points
#         k3d_points.shader = 'flat'
#
#         if None is not directions:
#             colors = k3d.helpers.map_colors(
#                 samples_distances, distances_cmap, [0, max_distance_to_feature]
#             ).astype(np.uint32)
#             colors = [(c, c) for c in colors]
#
#             vectors = k3d.vectors(
#                 samples,
#                 directions * samples_distances[..., np.newaxis],
#                 use_head=False,
#                 line_width=directions_width,
#                 colors=colors)
#             #             print(vectors)
#             plot += vectors
#
#     #             directions_to_plot = np.hstack((samples, samples + directions))
#
#     #             for i, dir_to_plot in enumerate(directions_to_plot):
#     #                 dir_to_plot = dir_to_plot.reshape((2, 3))
#     #                 if np.all(dir_to_plot[0] == dir_to_plot[1]):
#     #                     continue
#     #                 color = int(colors[i]) if None is not colors else samples_color
#     #                 plt_line = k3d.line(dir_to_plot,
#     #                                     shader='mesh', width=directions_width, color=color)
#     #                 plot += plt_line
#
#     if None is not sharp_vert:
#         k3d_points = k3d.points(sharp_vert,
#                                 point_size=sharpvert_psize, color=sharpvert_color)
#         plot += k3d_points
#         k3d_points.shader = 'flat'
#
#         if None is not sharp_curves:
#             if None is not sharpcurve_color:
#                 color = sharpcurve_color
#             else:
#                 import randomcolor
#                 rand_color = randomcolor.RandomColor()
#             for i, vert_ind in enumerate(sharp_curves):
#                 sharp_points_curve = mesh.vertices[vert_ind]
#
#                 if None is sharpcurve_color:
#                     color = rand_color.generate(hue='red')[0]
#                     color = int('0x' + color[1:], 16)
#                 plt_line = k3d.line(sharp_points_curve,
#                                     shader='mesh', width=sharpcurve_width, color=color)
#                 plot += plt_line
#
#     plot.grid_visible = False
#     if display:
#         plot.display()
#
#     return plot


def display_depth_sharpness(
        depth_images=None,
        sharpness_images=None,
        axes_size=(8, 8),
        ncols=1,
        max_sharpness=1.0,
        bgcolor='white',
        sharpness_hard_thr=None,
        sharpness_hard_values=None,
        depth_bg_value=0.0,
        sharpness_bg_value=0.0,
        depth_cmap='viridis_r',
        sharpness_cmap='plasma_r'):
    import matplotlib.cm
    import matplotlib.pyplot as plt

    def fix_chw_array(image_array):
        if None is not image_array:
            image_array = np.asanyarray(image_array).copy()
            assert len(image_array.shape) in [2, 3], "Don't understand the datatype with shape {}".format(
                image_array.shape)
            if len(image_array.shape) == 2:
                image_array = image_array[np.newaxis, ...]
        return image_array

    depth_images = fix_chw_array(depth_images)
    sharpness_images = fix_chw_array(sharpness_images)

    if None is not depth_images and None is not sharpness_images:
        assert len(depth_images) == len(sharpness_images), 'depth and sharpness images dont coincide by length'
        n_images = len(depth_images)
        ncols, nrows, series = 2 * ncols, n_images // ncols, 2

        axes_size = axes_size[0] * ncols, axes_size[1] * nrows
        _, axs = plt.subplots(figsize=axes_size, nrows=nrows, ncols=ncols)

    elif None is not depth_images:
        n_images = len(depth_images)
        ncols, nrows, series = ncols, n_images // ncols, 1

        axes_size = axes_size[0] * ncols, axes_size[1] * nrows
        _, axs = plt.subplots(figsize=axes_size, nrows=nrows, ncols=ncols)

    elif None is not sharpness_images:
        n_images = len(sharpness_images)
        ncols, nrows, series = ncols, n_images // ncols, 1

        axes_size = axes_size[0] * ncols, axes_size[1] * nrows
        _, axs = plt.subplots(figsize=axes_size, nrows=nrows, ncols=ncols)

    else:
        raise ValueError('at least one of "depth_images" or "sharpness_images" must be specified')

    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = np.atleast_2d(axs)
    elif ncols == 1:
        axs = np.atleast_2d(axs).T

    is_hard_label = False
    if None is not sharpness_hard_thr:
        is_hard_label = True
        if None is not sharpness_hard_values:
            assert isinstance(sharpness_hard_values, (tuple, list)) and len(sharpness_hard_values) == 2, \
                '"sharpness_hard_values" must be a tuple of size 2'
            low, high = sharpness_hard_values
        else:
            low, high = 0.0, max_sharpness

    if None is not depth_images:
        depth_colormap = copy.copy(matplotlib.cm.get_cmap(depth_cmap))
        depth_colormap.set_bad(color=bgcolor)

        for row in range(nrows):
            for col in range(0, ncols, series):
                depth_idx = (row * ncols + col) // series
                depth_ax = axs[row][col]

                depth_image = depth_images[depth_idx].copy()
                background_idx = depth_image == depth_bg_value
                depth_image[background_idx] = np.nan

                depth_ax.imshow(depth_image, interpolation='nearest', cmap=depth_colormap)
                depth_ax.axis('off')

    if None is not sharpness_images:
        sharpness_colormap = copy.copy(matplotlib.cm.get_cmap(sharpness_cmap))
        sharpness_colormap.set_bad(color=bgcolor)

        for row in range(nrows):
            for col in range(0, ncols, series):
                sharpness_idx = (row * ncols + col) // series
                sharpness_ax = axs[row][col + 1] if series == 2 else axs[row][col]

                sharpness_image = sharpness_images[sharpness_idx].copy()
                background_idx = sharpness_image == sharpness_bg_value
                sharpness_image[background_idx] = np.nan
                if is_hard_label:
                    sharpness_image[sharpness_image <= sharpness_hard_thr] = low
                    sharpness_image[sharpness_image > sharpness_hard_thr] = high

                tol = 1e-3
                sharpness_ax.imshow(sharpness_image, interpolation='nearest', cmap=sharpness_colormap,
                                    vmin=-tol, vmax=max_sharpness + tol)
                sharpness_ax.axis('off')

    plt.tight_layout(pad=0, h_pad=0.25, w_pad=0.25)


def get_random_color(hue=None):
    rand_color = randomcolor.RandomColor()
    color = rand_color.generate(hue=hue)[0]
    color = int('0x' + color[1:], 16)
    return color


def illustrate_camera(
        camera_pose,
        l=1.0,
        w=1.0,
        use_head=False,
        hs=1.0,
):
    camera_center = np.array([
        camera_pose.frame_origin,
        camera_pose.frame_origin,
        camera_pose.frame_origin])

    camera_frame = np.array([
        camera_pose.frame_axes
    ]) * l

    x_color = 0xff0000
    y_color = 0x00ff00
    z_color = 0x0000ff

    vectors = k3d.vectors(
        camera_center,
        camera_frame,
        use_head=use_head,
        head_size=hs,
        line_width=w,
        colors=[x_color, x_color, y_color, y_color, z_color, z_color], )

    return vectors


def plot_views(
        views,
        mesh=None,
        camera_l=1,
        camera_w=1,
):
    plot = k3d.plot(grid_visible=True, height=768)

    for view in views:
        scan_mesh = trimesh.base.Trimesh(
            view.depth,
            view.faces,
            process=False,
            validate=False)

        plot += k3d.points(
            scan_mesh.vertices,
            point_size=0.25,
            color=get_random_color(hue='green'),
            shader='flat')

        plot += illustrate_camera(
            view.pose,
            l=camera_l,
            w=camera_w)

    if mesh is not None:
        plot += k3d.mesh(
            mesh.vertices,
            mesh.faces,
            color=0xaaaaaa,
            flat_shading=False)

    return plot


def display_patch_decomposition(
        mesh,
        features,
):
    plot = k3d.plot(grid_visible=True, height=768)

    for s in features['surfaces']:
        color = get_random_color(hue='green')
        submesh = def_feature_utils.get_surface_as_mesh(mesh, s)

        plot += k3d.mesh(
            submesh.vertices,
            submesh.faces,
            flat_shading=False,
            color=color,
            side='both')

    plot.display()
