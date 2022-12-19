import numpy as np
import networkx as nx
import torch
from torch import optim
from tqdm import tqdm, trange
from utils import *


def subdivide_wireframe(corner_positions, corner_pairs, curve_data, curve_distances, split_threshold):
    """
    Make a split in polyline segment if max projections onto the segment is greater than threshold
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        curve_data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        curve_distances (list of lists): per-point sharpness distances that correspond to each of the segments
        split_threshold (float): threshold for splits
    Returns:
        new_corners (list of lists): updated 3D coordinates of polyline nodes
        new_pairs (list of lists): updated pairs of indices from corner_positions that compose a segment
    """
    new_pairs = []
    new_corners = corner_positions.copy()
    assert len(corner_pairs) == len(curve_data)
    assert len(curve_distances) == len(curve_data)
    
    # for each segment
    for i in range(len(curve_data)):
        edge = corner_pairs[i]
        endpoints = np.array(corner_positions)[edge]
        data = curve_data[i]
        distances = curve_distances[i]
        if len(data) == 0:
            continue
            
        # compute prejections of points onto segments
        cosines = np.dot(data - endpoints[0], endpoints[1] - endpoints[0]) / (np.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0]) + 1e-8)
        projections = endpoints[0] + cosines[:,None] * (endpoints[1] - endpoints[0])
        proj_distances = np.linalg.norm(projections - data, axis=1)
        where = np.logical_and(cosines >= 0, cosines <= 1)
        if where.sum() == 0:
            new_pairs.append(edge)
            continue
        residual = np.abs(proj_distances - distances)[where]
        
        # check condition
        if residual.max() < split_threshold:
            new_pairs.append(edge)
            continue
        # do split
        else:   
            argmax = np.arange(len(cosines))[where][residual.argmax()]
            midpoint = data[argmax].tolist()
            new_corners.append(midpoint)
            if len(new_pairs) == 0:
                midpoint_ind = np.max(np.concatenate(corner_pairs)) + 1
            else:
                midpoint_ind = np.max(np.concatenate([new_pairs, np.array(corner_pairs).reshape(-1,2)])) + 1
            new_pairs.append([edge[0], midpoint_ind])
            new_pairs.append([midpoint_ind, edge[1]])
    return new_corners, new_pairs


def corner_loss(init_corners, corners_improved, pairs, data, distances, corner_triplets=[], triplets=[], dangling=[], alpha_fid=0, alpha_fit=1, alpha_ang=0, corner_alpha_ang=0):
    """
    Compute a projection loss
    Args:
        init_corners (Tensor): 3D coordinates of polyline nodes
        corners_improved (Tensor): placeholder for updated 3D coordinates of polyline nodes
        pairs (list of lists): pairs of indices from corner_positions that compose a segment
        data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        distances (list of lists): per-point sharpness distances that correspond to each of the segments
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
    Returns:
        loss (Tensor): loss value
    """
    fidelity_term = torch.sum((init_corners - corners_improved) ** 2)
    fit_term = 0
    # for each segment
    for i in range(len(data)):
        if len(data[i]) == 0:
            continue
        endpoints = corners_improved[pairs[i]]
        distances_curve = torch.Tensor(distances[i])
        data_curve = torch.Tensor(data[i])
        # compute projections
        cosines = torch.mm(torch.Tensor(data[i]) - endpoints[0][None,:], endpoints[1][:, None] - endpoints[0][:, None])/ (torch.dot(endpoints[1] - endpoints[0], endpoints[1] - endpoints[0]) + 1e-8)
        projections = endpoints[0] + cosines * (endpoints[1] - endpoints[0])
        proj_distances = torch.norm(projections - data_curve, dim=1)
        residual = torch.mean(torch.abs(proj_distances - distances_curve)) + 5*proj_distances.mean()# - torch.norm(endpoints[1] - endpoints[0])
        fit_term += residual
    angles = 0
    corner_angles = 0
    for triplet in triplets:
        trp = corners_improved[triplet]
        angles += torch.clamp(torch.dot((trp[1] - trp[0])/torch.norm(trp[1] - trp[0]),(trp[2] - trp[1])/torch.norm(trp[2] - trp[1])), -1, 1)
    for triplet in corner_triplets:
        trp = corners_improved[triplet]
        corner_angles += torch.clamp(torch.dot((trp[1] - trp[0])/torch.norm(trp[1] - trp[0]),(trp[2] - trp[1])/torch.norm(trp[2] - trp[1])), -1, 1)
    
    return alpha_fit  * fit_term + alpha_fid * fidelity_term - alpha_ang * angles - corner_alpha_ang  * corner_angles, alpha_fit  * fit_term, - alpha_ang  * angles - corner_alpha_ang  * corner_angles


def move_corners(corner_positions, corner_pairs, curve_data, curve_distances, corner_triplets=[], triplets=[], dangling=[], alpha_fid=0, alpha_fit=1, alpha_ang=0, corner_alpha_ang=0): 
    """
    Do corner positions optimization based on corner loss
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        curve_data (list of lists): 3D coordiantes of points that correspond to each of the segments 
        curve_distances (list of lists): per-point sharpness distances that correspond to each of the segments
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
    Returns:
        corners_improved (list of lists): updated 3D coordinates of polyline nodes
    """
    init_corners = torch.Tensor(corner_positions)
    corners_improved = torch.Tensor(corner_positions)
    corners_improved.requires_grad_()
    
#     matching = parallel_nearest_point(np.array(corner_positions), np.array(corner_pairs), points)
#     curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)
    
    # create optimizer
    optimizer = optim.SGD([corners_improved], lr=0.001, momentum=0.9)
    t = trange(150, desc='Optimization', leave=True)
    prev_loss = 0
    for i in t:
        optimizer.zero_grad()
        
        loss, fit_loss, ang_loss = corner_loss(init_corners, corners_improved, corner_pairs, curve_data, curve_distances, corner_triplets, triplets, dangling, alpha_fid, alpha_fit, alpha_ang, corner_alpha_ang)
        loss.backward()
        optimizer.step()
        
#         matching = parallel_nearest_point(corners_improved.detach().numpy(), np.array(corner_pairs), points)
#         curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)
        try:
            s = 'Optimization: step #{0:}, fit loss: {1:3.1f}, angle loss: {2:3.1f}'.format(i, fit_loss.item(), ang_loss.item())
            t.set_description(s)
            t.refresh()
        except:
            continue
            
        
        if torch.abs(loss - torch.Tensor([prev_loss])) < 1e-5:
            return corners_improved.detach().numpy().tolist()
        prev_loss = loss.item()
    return corners_improved.detach().numpy().tolist()


def get_paths_and_corners(corner_pairs, corner_positions, corner_centers):
    G = nx.Graph()
    G.add_edges_from(corner_pairs)
    nodes = [k for k,v in dict(G.degree()).items() if v != 2]
    nodes_1 = [k for k,v in dict(G.degree()).items() if v == 1]
    corners = [k for k,v in dict(G.degree()).items() if v > 2]
    paths = []
    triplets = []
    corner_triplets = []
    if len(corner_centers) > 0:
        nodes = np.unique(np.concatenate([nodes, np.arange(len(corner_centers))]))
    for i in nodes:
        for j in nodes:
            try:
                path = nx.shortest_path(G, i, j)
            except:
                continue
            if len(path) == 1:
                continue
            if (len(path) - len(np.setdiff1d(path, nodes))) > 2:
                continue
            if path[::-1] in paths:
                continue
            if path in paths:
                continue
            paths.append(path)
            if len(path) > 3:
                corner_triplet = [[path[k], path[k+1], path[k+2]] for k in [0,len(path)-3] if k+2 < len(path)]
                if len(corner_triplet) > 0:
                    corner_triplets.append(corner_triplet)
            if len(path) > 2:
                triplet = [[path[k], path[k+1], path[k+2]] for k in range(len(path)) if k+2 < len(path)]
                if len(triplet) > 0:
                    triplets.append(triplet)
    if paths != []:
        closed_nodes = np.setdiff1d(np.arange(len(corner_positions)), np.unique(np.concatenate(paths)))
    else:
        closed_nodes = np.setdiff1d(np.arange(len(corner_positions)), [])
    closed_paths = []
    path = []
    for i in closed_nodes:
        if np.isin(i, path):
            continue
        try:
            path = nx.algorithms.cycles.find_cycle(G, i)
        except:
            continue
        if np.isin([item for sublist in paths for item in sublist], path).any():
            continue
        if np.isin([item for sublist in closed_paths for item in sublist], path).any():
            continue
        closed_paths.append(np.append(np.array(path)[:,0], np.array(path)[0,0]).tolist()) 

    pairs_left = [None]
    repeat_counter = 0
    while len(pairs_left) > 0:
        repeat_counter += 1
        if repeat_counter > 5:
            break
        all_path_pairs = []
        for path in paths:
            if len(path) == 2:
                all_path_pairs.append(path)
            else:
                for pair in zip(path[:-1], path[1:]):
                    all_path_pairs.append(list(pair))
        for path in closed_paths:
            for pair in zip(path[:-1], path[1:]):
                    all_path_pairs.append(list(pair))

        pairs_left = np.array(corner_pairs)[np.logical_not(np.isin(corner_pairs, all_path_pairs).all(-1))]

        G = nx.Graph()
        G.add_edges_from(pairs_left)
    #     nodes = [k for k,v in dict(G.degree()).items() if v != 2]
        for i in nodes:
            for j in nodes:
                try:
                    path = nx.shortest_path(G, i, j)
                except:
                    continue
                if len(path) == 1:
                    continue
                if (len(path) - len(np.setdiff1d(path, nodes))) > 2:
                    continue
                if path[::-1] in paths:
                    continue
                if path in paths:
                    continue
                if np.isin(path[1:-1], np.unique(np.concatenate(paths))).any():
                    continue
                paths.append(path)
                if len(path) > 3:
                    corner_triplet = [[path[k], path[k+1], path[k+2]] for k in [0,len(path)-3] if k+2 < len(path)]
                    if len(corner_triplet) > 0:
                        corner_triplets.append(corner_triplet)
                if len(path) > 2:
                    triplet = [[path[k], path[k+1], path[k+2]] for k in range(len(path)) if k+2 < len(path)]
                    if len(triplet) > 0:
                        triplets.append(triplet)
        
#     triplets = np.concatenate(triplets).tolist()
    if len(corner_triplets) > 0:
        corner_triplets = np.concatenate(corner_triplets).tolist()
    else: corner_triplets = []
    if len(triplets) > 0:
        triplets = np.concatenate(triplets).tolist()
    else: triplets = []
    return np.array(corner_positions)[corners], paths, [corner_triplets, triplets], closed_paths, nodes_1


def optimize_topological_graph_old(corner_positions, corner_pairs, points, distances, split_threshold, alpha_ang, corner_alpha_ang):
    """
    Do final optimization of the topological graph.
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        points (np.array): 3D coordiantes of points
        distances (np.array): per-point sharpness distances
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
        split_threshold (float): threshold for splits
    Returns:
        new_corners (list of lists): updated 3D coordinates of polyline nodes
        new_pairs (list of lists): updated pairs of indices from corner_positions that compose a segment
    """
    # match points to segments using aabb
    matching = parallel_nearest_point(np.array(corner_positions), np.array(corner_pairs), points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)

    # optimize corner positions
    _, _, triplets, _, _ = get_paths_and_corners(corner_pairs, corner_positions)
    
    corner_triplets, triplets = triplets
    
    corner_positions = move_corners(corner_positions, corner_pairs, curve_data, curve_distances, corner_triplets, triplets, [], 
                                    alpha_fit=1, alpha_fid=0, alpha_ang=alpha_ang, corner_alpha_ang=corner_alpha_ang)

    return corner_positions, corner_pairs


def optimize_topological_graph(corner_positions, corner_pairs, corner_centers, points, distances, split_threshold, alpha_fit, alpha_ang, corner_alpha_ang):
    """
    Do final optimization of the topological graph.
    Args:
        corner_positions (list): 3D coordinates of polyline nodes
        corner_pairs (list of lists): pairs of indices from corner_positions that compose a segment
        points (np.array): 3D coordiantes of points
        distances (np.array): per-point sharpness distances
        alpha_fid (float): fidelity term weight
        alpha_fit (float): fit term weight
        split_threshold (float): threshold for splits
    Returns:
        new_corners (list of lists): updated 3D coordinates of polyline nodes
        new_pairs (list of lists): updated pairs of indices from corner_positions that compose a segment
    """
    # match points to segments using aabb
    matching = parallel_nearest_point(np.array(corner_positions), np.array(corner_pairs), points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)

    # optimize corner positions
    _, _, triplets, _, _ = get_paths_and_corners(corner_pairs, corner_positions, corner_centers)
    corner_triplets, triplets = triplets
    
    corner_positions = move_corners(corner_positions, corner_pairs, curve_data, curve_distances, corner_triplets, triplets, [], 
                                    alpha_fit=alpha_fit, alpha_fid=0, alpha_ang=alpha_ang, corner_alpha_ang=corner_alpha_ang)
#     print(points)

#     rematch points to segments
    matching = parallel_nearest_point(np.array(corner_positions), np.array(corner_pairs), points)
    curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)

# #     while splits are available, do splits
#     previous_len = 0
#     i = 0
#     while len(corner_positions) != previous_len:
#         if i > 2:
#             break
#     print('split')
#     previous_len = len(corner_positions)
#     corner_positions, corner_pairs = subdivide_wireframe(corner_positions, corner_pairs, 
#                                                          curve_data, curve_distances,
#                                                          split_threshold)
#     matching = parallel_nearest_point(np.array(corner_positions), np.array(corner_pairs), points)
#     curve_data, curve_distances = recalculate(points, distances, matching, corner_pairs)
#         i += 1
    

# #     final corner positions optimization
    _, _, triplets, _, _ = get_paths_and_corners(corner_pairs, corner_positions, corner_centers)
    corner_triplets, triplets = triplets
    
    corner_positions = move_corners(corner_positions, corner_pairs, curve_data, curve_distances, corner_triplets, triplets, [], 
                                    alpha_fit=alpha_fit, alpha_fid=0, alpha_ang=alpha_ang, corner_alpha_ang=corner_alpha_ang)
    
    return corner_positions, corner_pairs
