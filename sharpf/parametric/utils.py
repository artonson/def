import k3d
# import randomcolor

import numpy as np
from scipy.interpolate import splev


def recalculate(points, distances, matching, corner_pairs):
    # using aabb, create lists of lists for points and distances that correspond to each segment
    curve_data = []
    curve_distances = []
    for i in range(len(corner_pairs)):
        curve_data.append(points[np.array(matching) == i]) 
        curve_distances.append(distances[np.array(matching) == i]) 
    return curve_data, curve_distances


def draw(points, corner_positions, corner_pairs, path_to_save, filename, DISPLAY_RES, tcks=[], closed_tcks=[], straight_lines=[]):
#     print(corner_positions.shape)
    plot = k3d.plot()

    k3d_points = k3d.points(points, point_size=DISPLAY_RES, opacity=0.1, shader='3d', name='sharp_points')
    plot += k3d_points
    
    if np.logical_and(np.logical_and(len(tcks) == 0, len(straight_lines) == 0), len(closed_tcks) == 0):
        
        points_corner_centers = k3d.points(corner_positions,
                                           color=0xFF0000, point_size=DISPLAY_RES, shader='3d', name='polyline_nodes')
        plot += points_corner_centers

        for edge in corner_pairs:
            e = k3d.line(np.array(corner_positions)[edge], name='polyline_edge')
            plot+=e
            
        with open('{path_to_save}/{filename}__result.html'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
            f.write(plot.get_snapshot())

    else:
        for i in range(len(straight_lines)):
            spline = k3d.line(straight_lines[i], color=0xff0000, width=DISPLAY_RES-0.015)
            plot += spline
        for i in range(len(tcks)):    
            spline = k3d.points(np.array(splev(np.linspace(0,1,2500), tcks[i])).T, color=0xff0000, point_size=DISPLAY_RES-0.015, shader='flat')
            plot += spline
        for i in range(len(closed_tcks)):    
            spline = k3d.points(np.array(splev(np.linspace(0,1,2500), closed_tcks[i])).T, color=0xff0000, point_size=DISPLAY_RES-0.015, shader='flat')
            plot += spline
        with open('{path_to_save}/{filename}__final_result.html'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
            f.write(plot.get_snapshot())
