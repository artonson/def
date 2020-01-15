import numpy as np
import math  
from matrix_torch import (create_rotation_matrix_x, create_rotation_matrix_y, create_rotation_matrix_z, create_translation_matrix)
import xml.etree.ElementTree as ET
import torch

def from_extrinsic(extrinsic, fix_axes=False):
    if fix_axes:
        extrinsic[:, 1:3] *= -1
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    view_matrix = np.identity(4)
    view_matrix[:3,:3] = R.T
    view_matrix[:3, 3] = -R.T@T
    return view_matrix

def from_xml(xml_file, to_tensor=True):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = int(e.get('label'))
#         print(label)
        transforms[label] = e.find('transform').text
    
    view_matrices = {}
    for label in transforms:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
#         view[:, 1:3] *= -1
        view_matrix = from_extrinsic(extrinsic, fix_axes=True).T
        if to_tensor:
            view_matrix = torch.tensor(view_matrix, dtype=torch.float32)
        view_matrices[label] = view_matrix
    return view_matrices

def parse_line(line):
    return [float(x) for x in line.strip('\n').split()]
            
def parse_txt(file):
    with open(file) as src:
        data = [parse_line(line) for line in src.readlines()]
    return np.array(data)
            
def from_txt(file):
    view_data = parse_txt(file)
    k = 4   
    view_matrices = [np.array(view_data[i:i+k]).T for i in range(0, view_data.shape[0], k)]
    return view_matrices

def from_pose(local_rotation, local_position):
    rotation_x = create_rotation_matrix_x(-local_rotation[0])
    rotation_y = create_rotation_matrix_y(-local_rotation[1])
    rotation_z = create_rotation_matrix_z(-local_rotation[2])
    
    translation = create_translation_matrix(-local_position[0], -local_position[1], -local_position[2])
    rotation = torch.mm(rotation_x, rotation_y)
    rotation = torch.mm(rotation, rotation_z)
    transform = torch.mm(rotation, translation)
    
    return transform.t()

def to_pose(view_matrix):
    R = view_matrix[:3,:3]
    T = view_matrix[3, :3]
    T = -np.linalg.inv(R.T)@T
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z]), T

def rotate_around(transform, rotate_around, angle):
    
    ra_translation = create_translation_matrix(rotate_around[0], \
                                             rotate_around[1], \
                                             rotate_around[2])
    ra_translation_inverse = create_translation_matrix(-rotate_around[0], \
                                             -rotate_around[1], \
                                             -rotate_around[2])
    ra_rotation = create_rotation_matrix_y(-angle[1])
    ra_rotation = torch.mm(ra_rotation, create_rotation_matrix_z(-angle[2]))
    
    ra_transform = torch.mm(transform, ra_translation)
    ra_transform = torch.mm(ra_transform, ra_rotation)
    ra_transform = torch.mm(ra_transform, ra_translation_inverse)
    
    return ra_transform.t()
