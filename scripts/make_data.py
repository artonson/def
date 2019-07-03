import argparse
import datetime

import numpy as np
import yaml


# the function for patch generator: breadth-first search

def find_and_add(sets, desired_number_of_points, adjacency_graph):
    counter = len(sets) # counter for number of vertices added to the patch;
    # sets is the list of vertices included to the patch
    for verts in sets:
        for vs in adjacency_graph.neighbors(verts):
            if vs not in sets:
                sets.append(vs)
                counter += 1
#                 print(counter)
        if counter >= desired_number_of_points:
            break # stop when the patch has more than 1024 vertices



def generate_patches(addrs_very_good_triangles):
    points = []  # for storing initial coordinates of points
    points_normalized = []  # for storing normalized coordinates of points
    labels = []  # for storing 0-1 labels for non-sharp/sharp points
    normals = []
    surface_rate = []  # for counting how many surfaces are there in the patch
    sharp_rate = []  # for indicator whether the oatch containes sharp vertices at all
    times = []  # for times (useless)
    p_names = []  # for names of the patches in the format "initial_mesh_name_N", where N is the starting vertex index
    desired_number_of_points = 1024

    for addr in addrs_very_good_triangles:

        t_0 = datetime.datetime.now()

        if addr[:4] == '0005':
            folder = 'yml_05/'
        else:
            folder = 'yml_06/'

        yml = yaml.load(
            open('/home/Albert.Matveev/sharp/abc_fine/' + folder + addr[:-11] + 'features' + addr[-4:] + '.yml', 'r'))

        t_yml_read = datetime.datetime.now()

        sharp_idx = []
        short_idx = []
        for i in yml['curves']:
            if len(i['vert_indices']) < 5:  # this is for filtering based on short curves:
                # append all the vertices which are in the curves with less than 5 vertices
                short_idx.append(np.array(i['vert_indices']) - 1)  # you need to substract 1 from vertex index,
                # since it starts with 1
            if ('sharp' in i.keys() and i['sharp'] == True):
                sharp_idx.append(np.array(i['vert_indices']) - 1)  # append all the vertices which are marked as sharp
        if len(sharp_idx) > 0:
            sharp_idx = np.unique(np.concatenate(sharp_idx))
        if len(short_idx) > 0:
            short_idx = np.unique(np.concatenate(short_idx))

        t_curves_read = datetime.datetime.now()

        surfaces = []
        for i in yml['surfaces']:
            if 'vert_indices' in i.keys():
                surfaces.append(np.array(i['face_indices']) - 1)

        t_surfaces_read = datetime.datetime.now()

        vertices = []
        faces = []
        if addr[:4] == '0005':
            folder = 'obj_05/'
        else:
            folder = 'obj_06/'

        for line in open("/home/Albert.Matveev/sharp/abc_fine/" + folder + addr + ".obj",
                         "r"):  # read the mesh: since trimesh
            # messes the indices,
            # this has to be done manually
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertices.append(np.array(values[1:4], dtype='float'))
            elif values[0] == 'f':
                faces.append(np.array([values[1].split('//')[0], values[2].split('//')[0], values[3].split('//')[0]],
                                      dtype='int'))

        t_obj_read = datetime.datetime.now()

        vertices = np.array(vertices)
        faces = np.array(faces) - 1
        sharp_indicator = np.zeros((len(vertices),))
        sharp_indicator[sharp_idx] = 1

        mesh = trm.base.Trimesh(vertices=vertices, faces=faces, process=False)  # create a mesh from the vertices
        # and faces read previously
        adjacency_graph = mesh.vertex_adjacency_graph

        t_mesh_read = datetime.datetime.now()

        for j in np.linspace(0, len(vertices), 7, dtype='int')[:-1]:  # select starting vertices to grow patches from,
            # while iterating over them use BFS to
            # generate patches
            #         for j in [529]:
            set_of_verts = [j]
            surfaces_numbers = []
            find_and_add(sets=set_of_verts, desired_number_of_points=desired_number_of_points,
                         adjacency_graph=adjacency_graph)  # BFS function
            a = sharp_indicator[np.array(set_of_verts)[-100:]]
            b = np.isin(np.array(set_of_verts)[-100:], np.array(set_of_verts)[-100:] - 1)
            if (a[b].sum() > 3):
                #                 print('here! border!',j)
                continue
            set_of_verts = np.unique(np.array(set_of_verts))  # the resulting list of vertices in the patch
            if np.isin(set_of_verts, short_idx).any():  # discard a patch if there are short lines
                continue
            patch_vertices = mesh.vertices[set_of_verts]
            patch_sharp = sharp_indicator[set_of_verts]
            patch_normals = mesh.vertex_normals[set_of_verts]

            if patch_sharp.sum() != 0:
                sharp_rate.append(1)
            else:
                sharp_rate.append(0)

            if patch_vertices.shape[0] >= desired_number_of_points:
                # select those vertices, which are not sharp in order to use them for counting surfaces (sharp vertices
                # are counted twice, since they are on the border between two surfaces, hence they are discarded)
                appropriate_verts = set_of_verts[:desired_number_of_points][
                    patch_sharp[:desired_number_of_points].astype(int) == 0]
                for surf_idx, surf_faces in enumerate(surfaces):
                    surf_verts = np.unique(mesh.faces[surf_faces].ravel())
                    if len(np.where(np.isin(appropriate_verts, surf_verts))[0]) > 0:
                        surface_ratio = sharp_indicator[np.unique(np.array(surf_verts))].sum() / len(
                            np.unique(np.array(surf_verts)))
                        if (surface_ratio > 0.6):
                            break
                        surfaces_numbers.append(surf_idx)  # write indices of surfaces which are present in the patch
                        continue
                if (surface_ratio > 0.6):
                    continue
                surface_rate.append(np.unique(np.array(surfaces_numbers)))
                patch_vertices = patch_vertices[:desired_number_of_points]
                points.append(patch_vertices)
                patch_vertices_normalized = patch_vertices - patch_vertices.mean(axis=0)
                patch_vertices_normalized = patch_vertices_normalized / np.linalg.norm(patch_vertices_normalized,
                                                                                       ord=2, axis=1).max()
                points_normalized.append(patch_vertices_normalized)
                patch_normals = patch_normals[:desired_number_of_points]
                normals.append(patch_normals)
                labels.append(patch_sharp[:desired_number_of_points])

                p_names.append('%s_%i' % (addr, j))

        t_patches_ready = datetime.datetime.now()
        times.append(np.array([(t_yml_read - t_0).microseconds, (t_curves_read - t_yml_read).microseconds,
                               (t_surfaces_read - t_curves_read).microseconds,
                               (t_obj_read - t_surfaces_read).microseconds,
                               (t_mesh_read - t_obj_read).microseconds, (t_patches_ready - t_mesh_read).microseconds,
                               (t_patches_ready - t_0).microseconds]))

    times = np.array(times)
    p_names = np.array(p_names)
    points = np.array(points)
    points_normalized = np.array(points_normalized)
    labels = np.array(labels).reshape(-1, 1024, 1)
    normals = np.array(normals)
    sharp_rate = np.array(sharp_rate)
    return times, p_names, points, points_normalized, labels, sharp_rate, surface_rate, normals




def main(options):
    pass



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int, default=4, help='CPU jobs to use in parallel [default: 4].')

    subparsers = parser.add_subparsers(help='sub-command help')

     # create the parser for the "a" command
    stats_parser = subparsers.add_parser('stats', help='compute statistics')
    stats_parser.add_argument('bar', type=int, help='bar help')

    # create the parser for the "b" command
    patches_parser = subparsers.add_parser('patches', help='generate patches')
    patches_parser.add_argument('bar', type=int, help='bar help')





    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=128, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    parser.add_argument('--batches-before-val', type=int, default=1024, dest='batches_before_val',
                        help='how many batches to train before validation [default: 1024].')
    parser.add_argument('--batches-before-imglog', type=int, default=12, dest='batches_before_imglog',
                        help='log images each batches-before-imglog validation batches [default: 12].')
    parser.add_argument('--mini-val-batches-n-per-subset', type=int, default=12, dest='mini_val_batches_n_per_subset',
                        help='how many batches per subset to run for mini validation [default: 12].')

    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')
    parser.add_argument('--infer-from-spec', dest='infer_from_spec', action='store_true', default=False,
                        help='if set, --model, --save-model-file, --logging-file, --tboard-json-logging-file,'
                             'and --tboard-dir are formed automatically [default: False].')
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')

    parser.add_argument('-s', '--save-model-file', dest='save_model_filename',
                        help='Path to output vectorization model file [default: none].')
    parser.add_argument('--batches_before_save', type=int, default=1024, dest='batches_before_save',
                        help='how many batches to run before saving the model [default: 1024].')

    parser.add_argument('--data-root', required=True, dest='data_root', help='root of the data tree (directory).')
    parser.add_argument('--data-type', required=True, dest='dataloader_type',
                        help='type of the train/val data to use.', choices=dataloading.prepare_loaders.keys())
    parser.add_argument('--handcrafted-train', required=False, action='append',
                        dest='handcrafted_train_paths', help='dirnames of handcrafted datasets used for training '
                                                             '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val', required=False, action='append',
                        dest='handcrafted_val_paths', help='dirnames of handcrafted datasets used for validation '
                                                           '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val-part', required=False, type=float, default=.1,
                        dest='handcrafted_val_part', help='portion of handcrafted_train used for validation')
    parser.add_argument('-M', '--memory-constraint', required=True, type=int, dest='memory_constraint',help='maximum RAM usage in bytes.')

    parser.add_argument('-r', '--render-resolution', dest='render_res', default=64, type=int,
                        help='resolution used for rendering.')

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-l', '--logging-file', dest='logging_filename',
                        help='Path to output logging text file [default: output to stdout only].')
    parser.add_argument('-tl', '--tboard-json-logging-file', dest='tboard_json_logging_file',
                        help='Path to output logging JSON file with scalars [default: none].')
    parser.add_argument('-x', '--tboard-dir', dest='tboard_dir',
                        help='Path to tensorboard [default: do not log events].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
