import argparse
import numpy as np
import h5py
import yaml

from scipy.interpolate import UnivariateSpline, splev, splprep
from scipy.optimize import minimize
from optimization import get_paths_and_corners
from topological_graph import separate_graph_connected_components
from utils import *
from tqdm import tqdm


def err(c, points, distances, u, t, k, endpoints):
    c = c.reshape(3,-1)
    eval_spline = np.array(splev(u, (t,c,k))).T
    proj_distances = np.linalg.norm(points - eval_spline, axis=1)
    residual = np.sum(np.abs(proj_distances - distances))
    return residual


def process_paths(paths, corner_pairs, corner_positions, curve_data, curve_distances):
    path_points = []
    path_distances = []
    path_params = []
    path_knots = []
    for path in paths:
        path_len = len(path)
        path_pairs = list(zip(path, path[1:]))
        pair_nodes = np.array(corner_positions)[np.array(path_pairs)]
        max_dist_pair_ind = np.linalg.norm(pair_nodes[:,1] - pair_nodes[:,0], axis=1).argmax()
        current_points = []
        current_distances = []
        current_params = []
        current_knots = [0]
        for i,current_pair in enumerate(path_pairs):
            endpoints = np.array(corner_positions)[np.array(current_pair)]
#             if i == 0:
#                 current_params.append(np.array(0)[None])
#                 current_points.append(endpoints[0][None,:])
#                 current_distances.append(np.array(0)[None])
            ind = np.where(np.isin(corner_pairs, current_pair).sum(1) == 2)[0][0]
            cosines = np.dot(curve_data[ind] - endpoints[0], endpoints[1] - endpoints[0]) / np.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0])
            param = np.linalg.norm(cosines[:,None] * (endpoints[1] - endpoints[0]), axis=1)
            knot_param = np.linalg.norm(endpoints[1] - endpoints[0])
    #         param = cosines
            argsort = np.argsort(param)
            if len(current_params) < 2:
                max_param = 0
            else:
                max_param = np.concatenate(current_params).max()
            current_params.append(param[argsort] + max_param)
            knot_param += max_param
#             if np.logical_and(path_len < 8, i == max_dist_pair_ind):
#                 current_knots.append(current_knots[-1] / 4 + 3 * knot_param / 4)
#                 current_knots.append((current_knots[-1] + knot_param) / 2)
#                 current_knots.append(3 * current_knots[-1] / 4 + knot_param / 4)
            current_knots.append(knot_param)
            current_points.append(curve_data[ind][argsort])
            current_distances.append(curve_distances[ind][argsort])
#             if i == len(path_pairs) - 1:
    #             max_param = np.concatenate(current_params).max()
    #             current_params.append(np.linalg.norm(endpoints[1] - endpoints[0])[None] + max_param)
    #             current_points.append(endpoints[1][None,:])       
    #             current_distances.append(np.array(0)[None])
        path_points.append(np.concatenate(current_points))
        path_distances.append(np.concatenate(current_distances))
        path_params.append(np.concatenate(current_params))
        path_knots.append(current_knots[:-1])
        
    return path_points, path_distances, path_params, path_knots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', required=True, 
                        help='path to points')
    parser.add_argument('--preds', required=True, 
                        help='path to predictions')
    parser.add_argument('--save_folder', required=True, 
                        help='path to corner positions and pairs')
    
    parser.add_argument('--res', type=float, default=0.02, 
                        help='point cloud resolution (avg pointwise distance) [default: 0.02]')
    parser.add_argument('--sharp', type=float, default=1.5, 
                        help='sharpness threshold [default: 1.5]')
    
    parser.add_argument('--knn_radius', type=float, default=3, 
                        help='max distance to connect a pair in knn graph [default: 3]')
    parser.add_argument('--filt_factor', type=int, default=30, 
                        help='min number of points in connected component to get through the filtering [default: 30]')
    
#     parser.add_argument('--fps_factor', type=int, default=5, 
#                         help='how much less points to sample for fps [default: 5]')
    
#     parser.add_argument('--corner_R', type=float, default=6, 
#                         help='ball radius for corner detection [default: 6]')
#     parser.add_argument('--corner_r', type=float, default=4, 
#                         help='ball radius for corner separation [default: 4]')
#     parser.add_argument('--corner_up_thr', type=float, default=0.2, 
#                         help='upper variance threshold to compute cornerness [default: 0.2]')
#     parser.add_argument('--corner_low_thr', type=float, default=0.1, 
#                         help='lower variance threshold to compute cornerness [default: 0.1]')
#     parser.add_argument('--cornerness', type=float, default=1.25, 
#                         help='threshold to consider neighbourhood as a corner [default: 1.25]')
    
#     parser.add_argument('--endpoint_R', type=float, default=6, 
#                         help='ball radius for endpoint detection [default: 6]')
#     parser.add_argument('--endpoint_thr', type=float, default=0.4, 
#                         help='threshold to consider neighbourhood as an enpoint [default: 0.4]')
    
#     parser.add_argument('--connect_R', type=float, default=20, 
#                         help='distance for endpoint connection to the corners [default: 20]')
    
#     parser.add_argument('--init_thr', type=float, default=3, 
#                         help='initial polyline subdivision distance [default: 3]')
#     parser.add_argument('--opt_thr', type=float, default=3, 
#                         help='final polyline subdivision distance [default: 3]')
#     parser.add_argument('--alpha_fid', type=float, default=0, 
#                         help='corner optimization fidelity term weight [default: 0]')
#     parser.add_argument('--alpha_fit', type=float, default=1, 
#                         help='corner optimization fit term weight [default: 1]')
#     parser.add_argument('--alpha_ang', type=float, default=0, 
#                         help='corner optimization rigidity term weight [default: 0]')
    
    parser.add_argument('--draw', type=bool, default=True, 
                        help='whether to draw result [default: True]')

    return parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    options = parse_args()
    
    path_to_points = options.points
    path_to_preds = options.preds
    path_to_save = options.save_folder
    
    RES = options.res
    sharpness_threshold = RES * options.sharp
    
    filtering_radius = RES * options.knn_radius # max distance to connect a pair in knn filtering
#     corner_connected_components_radius = RES * options.knn_radius # max distance to connect a pair in knn corner separation
#     curve_connected_components_radius = RES * options.knn_radius # max distance to connect a pair in knn curve separation
    
    filtering_factor = options.filt_factor    
#     fps_factor = options.fps_factor
    
#     corner_detector_radius = RES * options.corner_R
#     corner_extractor_radius = RES * options.corner_r
#     upper_variance_threshold = options.corner_up_thr
#     lower_variance_threshold = options.corner_low_thr
#     cornerness_threshold = options.cornerness
#     box_margin = sharpness_threshold
    
#     endpoint_detector_radius = RES * options.endpoint_R
#     endpoint_threshold = options.endpoint_thr
    
#     corner_connector_radius = RES * options.connect_R
    
#     initial_split_threshold = RES * options.init_thr 
#     optimization_split_threshold = RES * options.opt_thr
#     alpha_fid = options.alpha_fid
#     alpha_fit = options.alpha_fit
#     alpha_ang = options.alpha_ang
    
    draw_result = options.draw

    
    print('loading data from {path_to_points}'.format(path_to_points=path_to_points))
#     whole_model_points = np.load(path_to_points)
#     whole_model_distances = np.load(path_to_preds)
    with h5py.File(path_to_points, 'r') as f:
        whole_model_points = f['points'][:]
        whole_model_distances = f['distances'][:]
    points = whole_model_points[whole_model_distances < sharpness_threshold]
    distances = whole_model_distances[whole_model_distances < sharpness_threshold]
    filename = path_to_save.split('/')[-2]
    print('processing {size} points'.format(size=len(points)))

    print('filtering')
    filtered_clusters = separate_graph_connected_components(points, radius=filtering_radius, filtering_mode=True, 
                                                            filtering_factor=filtering_factor)
    points = points[np.unique(np.concatenate(filtered_clusters))]
    distances = distances[np.unique(np.concatenate(filtered_clusters))]
# print('loading data')
# points = np.load('/home/Albert.Matveev/tmp/fused/50/abc_0050_00500041_5aa40dcd43fa0b14df9bdcf8_010/points_sp.npy')
# distances = np.load('/home/Albert.Matveev/tmp/fused/50/abc_0050_00500041_5aa40dcd43fa0b14df9bdcf8_010/distances_sp.npy')
    corner_pairs = np.load('{path_to_preds}/{filename}__corner_pairs.npy'.format(path_to_preds=path_to_preds, filename=filename))
    corner_positions = np.load('{path_to_preds}/{filename}__corner_positions.npy'.format(path_to_preds=path_to_preds, filename=filename))
    
    print('creating paths')
    _, paths, _, closed_paths, _ = get_paths_and_corners(corner_pairs, corner_positions)


    print('aabboxing')
    aabboxes = create_aabboxes(np.array(corner_positions)[np.array(corner_pairs)])
    matching = parallel_nearest_point(aabboxes, np.array(corner_positions)[np.array(corner_pairs)], points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)


    path_points, path_distances, path_params, path_knots = process_paths(paths, 
                                                                 corner_pairs, corner_positions, 
                                                                 curve_data, curve_distances)


    straight = np.zeros((len(paths)))
    for i in range(len(paths)):
        endpoints = np.array(corner_positions)[np.array(paths[i])[[0,-1]]]
        midpoints = np.array(corner_positions)[np.array(paths[i])[1:-1]]
        cosines = np.dot(midpoints - endpoints[0], endpoints[1] - endpoints[0]) / np.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0])
        projections = endpoints[0] + cosines[:,None] * (endpoints[1] - endpoints[0])
        proj_distances = np.linalg.norm(projections - midpoints, axis = 1)
        if (proj_distances < RES*1.1).all():
            straight[i] = 1

    print('start splining')        
    tcks = []
    straight_lines = []
    closed_tcks = []
#     us = []
    for i in tqdm(range(len(paths))):
        linspace = (path_params[i] - path_params[i].min()) / (path_params[i].max() - path_params[i].min())
        knots = (path_knots[i] - path_params[i].min()) / (path_params[i].max() - path_params[i].min())
        knots_as_needed = np.zeros((8+len(knots)))
        knots_as_needed[-4:] = 1
        knots_as_needed[4:-4] = knots
        weights = 1 - path_distances[i]
        endpoints = np.array(corner_positions)[np.array(paths[i])[[0,-1]]]
        if straight[i] == 1:
            straight_lines.append(endpoints)
            continue
#         print(knots)
        (t,c0,k), u = splprep(u=linspace, x=path_points[i].T, w=weights, task=-1, t=np.sort(knots))
        con = ({'type': 'eq',
               'fun': lambda c: (splev(0, (t, c.reshape(3,-1), k))-endpoints[0])
               },
               {'type': 'eq',
               'fun': lambda c: (splev(1, (t, c.reshape(3,-1), k))-endpoints[1])
               })
        opt = minimize(err, np.array(c0).flatten(), (path_points[i], path_distances[i], u, t, k, endpoints),
                      constraints=con)
        copt = opt.x
        tcks.append((t, copt.reshape(3,-1), k))
#         us.append(u)
        
    print('saving open curves')
    tck_dict = {}
    for i, tck in enumerate(tcks):
        tck_subdict = {}
        tck_subdict['t'] = np.array(tck[0]).tolist()
        tck_subdict['c'] = np.array(tck[1]).tolist()
        tck_subdict['k'] = np.array(tck[2]).tolist()
        tck_dict['open_spline_{i}'.format(i=i)] = tck_subdict
    for i, endpoints in enumerate(straight_lines):
        tck_subdict = {}
        tck_subdict['ends'] = np.array(straight_lines[i]).tolist()
        tck_dict['line_{i}'.format(i=i)] = tck_subdict
#     with open('{path_to_save}/{filename}__open_splines.txt'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
#         for i in range(len(tcks)):
#             f.write(str(tcks[i]) + '\n')
# #     np.savetxt(tcks, '{path_to_save}/{filename}__open_splines.npy'.format(path_to_save=path_to_save, filename=filename))
#     np.save('{path_to_save}/{filename}__lines.npy'.format(path_to_save=path_to_save, filename=filename), straight_lines)
    with open('{path_to_save}/{filename}__curves.txt'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
        yaml.dump(tck_dict, f)
        
    if len(closed_paths) > 0:
        print('splining closed curves')
        path_points, path_distances, path_params, path_knots = process_paths(closed_paths, 
                                                             corner_pairs, corner_positions, 
                                                             curve_data, curve_distances)
        
        
#         closed_us = []
        for i in tqdm(range(len(closed_paths))):
            linspace = (path_params[i] - path_params[i].min()) / (path_params[i].max() - path_params[i].min())
            knots = (path_knots[i] - path_params[i].min()) / (path_params[i].max() - path_params[i].min())
            knots_as_needed = np.zeros((8+len(knots)))
            knots_as_needed[-4:] = 1
            knots_as_needed[4:-4] = knots
            weights = 1 - path_distances[i]
            endpoints = np.array(corner_positions)[np.array(closed_paths[i])[[0,-1]]]

            (t,c0,k), u = splprep(u=linspace, x=path_points[i].T, w=weights, task=-1, t=knots)
            con = ({'type': 'eq',
                   'fun': lambda c: (np.array(splev(0, (t, c.reshape(3,-1), k)))-np.array(splev(1, (t, c.reshape(3,-1), k))))
                   },
#                    {'type': 'eq',
#                    'fun': lambda c: (splev(1, (t, c.reshape(3,-1), k))-endpoints[1])
#                    },
                   {'type': 'eq',
                   'fun': lambda c: (np.sum(np.abs(np.array(splev(0, (t, c.reshape(3,-1), k), der=1))-np.array(splev(1, (t, c.reshape(3,-1), k), der=1)))))
                   }
                  )
            opt = minimize(err, np.array(c0).flatten(), (path_points[i], path_distances[i], u, t, k, endpoints),
                          constraints=con)
            copt = opt.x
            closed_tcks.append((t, copt.reshape(3,-1), k))
#             closed_us.append(u)
        print('saving closed curves')
        for i, tck in enumerate(closed_tcks):
            tck_subdict = {}
            tck_subdict['t'] = np.array(tck[0]).tolist()
            tck_subdict['c'] = np.array(tck[1]).tolist()
            tck_subdict['k'] = np.array(tck[2]).tolist()
            tck_dict['closed_spline_{i}'.format(i=i)] = tck_subdict
        with open('{path_to_save}/{filename}__curves.txt'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
            yaml.dump(tck_dict, f)
#         with open('{path_to_save}/{filename}__closed_splines.txt'.format(path_to_save=path_to_save, filename=filename), 'w') as f:
#             for i in range(len(closed_tcks)):
#                 f.write(str(closed_tcks[i]) + '\n')
            
    if draw_result:
        print('drawing')
        DISPLAY_RES = RES * 1.5
        draw(points, corner_positions, corner_pairs, path_to_save, filename, DISPLAY_RES, tcks, closed_tcks, straight_lines)
        
    print('done!')