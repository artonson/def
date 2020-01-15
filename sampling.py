import yaml
import pywavefront
from yaml import CLoader as Loader, CDumper as Dumper
import numpy as np

def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    return config

def process_vertices(vertices):
    '''Centering and scaling'''
    centroid = vertices.mean(axis=0)
#     print(centroid)
    centered = vertices-centroid
    max_dim = abs(centered.max(axis=0) - centered.min(axis=0))
#     print(max_dim)
    max_dim = max(max_dim)
    centered/=max_dim
    return centered


def fibonacci_sphere_sampling(samples=1, randomize=True, radius=1.0, positive_z=False):
    # Returns [x,y,z] tuples of a fibonacci sphere sampling
    if positive_z:
        samples *= 2
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        if positive_z:
            s = np.arcsin(z / radius) * 180.0 / np.pi
            if z > 0.0 and s > 30:
                points.append([radius * x, radius * y, radius * z])
        else:
            points.append([radius * x, radius * y, radius * z])

    return points
