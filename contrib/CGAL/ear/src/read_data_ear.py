import os
import sys
import h5py
import numpy as np
import tempfile

def metric(pred, gt):
    # balanced accuracy score
    pass

if __name__ == '__main__':
    for i in range(len(sys.argv)):
        if (sys.argv[i] == '-directory' or sys.argv[i] == '-d'):
            directory = sys.argv[i+1]
        if (sys.argv[i] == '-o'):
            out_directory = sys.argv[i+1]
    print('EAR Method')
    for filename in os.listdir(directory):
        print('Dataset from {}'.format(filename))
        out_filename = filename.split('.')[0] + '_ear_output.h5'
        out = h5py.File(out_directory+'/'+out_filename, 'w') 
        with h5py.File(directory+'/'+filename, 'r') as f:
            classif_data = out.create_dataset("classification", (f['data'].shape[0], f['data'].shape[1], 1))
            classification = np.zeros((f['data'].shape[0], f['data'].shape[1], 1))
            for i, data in enumerate(f['data']):
                if i <= 4:
                    continue
                print('datafile #{}'.format(i))
                tmp_f = open('ear_tmp_{}.xyz'.format(i), 'w')
                for point in data:
                    tmp_f.write('{} {} {}\n'.format(point[0], point[1], point[-1]))
                tmp_filename = 'ear_tmp_{}.xyz'.format(i)
                tmp_f.close()
                exec_cmd = ' '.join(['./edge_aware -filename', tmp_filename])
                os.system(exec_cmd)
            #temp_f = open('points_classification_' + tmp_filename.split('.')[0] + '.txt')
            #classification[i] = np.array(list(map(int, temp_f.readline().split(' ')[:-1])))[:, np.newaxis]
                print(os.listdir('.'))
                exec_cmd = 'mv points_2.txt {}/{}_points_2.txt'.format(out_directory, i)
                os.system(exec_cmd)
                exec_cmd = 'mv normals_2.txt {}/{}_normals_2.txt'.format(out_directory, i)
                os.system(exec_cmd)
                exec_cmd = 'mv sharp_classif.txt {}/{}_sharp_classif.txt'.format(out_directory, i)
                os.system(exec_cmd)
                os.remove(tmp_filename)
                
                if i >= 8:
                    break

