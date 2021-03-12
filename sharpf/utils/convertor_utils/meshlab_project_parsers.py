import os
import xml.etree.ElementTree as ET
from io import StringIO

import numpy as np
import trimesh


def load_meshlab_project(meshlab_project_filename):
    """Loads MeshLab .mlp file and gets

    :param meshlab_project_filename:
    :return:
    """
    base_dir = os.path.normpath(
        os.path.dirname(meshlab_project_filename))
    root = ET.parse(meshlab_project_filename).getroot()

    points_by_scan = []
    transform_by_scan_4x4 = []
    item_id = None

    for type_tag in root.findall('MeshGroup/MLMesh'):
        filename = type_tag.get('filename')
        print(filename)
        if filename.endswith('.obj') or filename.endswith('.stl'):
            item_id = filename

        elif filename.endswith('.ply'):
            try:
                transform = np.loadtxt(
                    StringIO(type_tag.find('MLMatrix44').text))
            except ValueError as e:
                pass
            else:
                transform_by_scan_4x4.append(transform)

            try:
                points = trimesh.load(os.path.join(base_dir, filename)).vertices
            except ValueError as e:
                pass
            else:

                points_by_scan.append(points)

    return points_by_scan, transform_by_scan_4x4, item_id
