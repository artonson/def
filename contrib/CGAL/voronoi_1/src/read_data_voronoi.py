import os
import sys
import h5py
import numpy as np
import tempfile

def metric(pred, gt):
    # balanced accuracy score
    pass

if __name__ == '__main__':
    R = 0.2
    r = 0.05
    threshold = 0.15 
    for i in range(len(sys.argv)):
        if (sys.argv[i] == '-d'):
            directory = sys.argv[i+1]
        if (sys.argv[i] == '-o'):
            out_directory = sys.argv[i+1]
        if (sys.argv[i] == '-R'):
            R = sys.argv[i+1]
        if (sys.argv[i] == '-r'):
            r = sys.argv[i+1]
        if (sys.argv[i] == '-t' or sys.argv[i] == '-threshold'):
            threshold = sys.argv[i+1]
    print('Voronoi Method 1, parameters R={}, r={}, t={}'.format(R, r, threshold))
#    exec_cmd = 'g++ -o voronoi voronoi_1.cpp -lCGAL -I/home/CGAL-4.14/include -lgmp'
#    os.system(exec_cmd)
    metric_results = []
    files = os.listdir(directory)
    for filename in files:
        print('Dataset from {}'.format(filename))
        out_filename = filename.split('.')[0] + '_voronoi_output.h5'
        exec_cmd = 'touch {}/{}'.format(out_directory, out_filename)
        os.system(exec_cmd)
        out = h5py.File(out_directory+'/'+out_filename, 'w')
        with h5py.File(directory+'/'+filename, 'r') as f:
            classif_data = out.create_dataset("classification", (f['points'].shape[0], f['points'].shape[1], 1))
            classification = np.zeros((f['points'].shape[0], f['points'].shape[1], 1))
            for i, data in enumerate(f['points']):
                print('datafile #{}'.format(i))
                tmp_filename_xyz = out_directory+'/'+'voronoi_tmp_{}.xyz'.format(i)
                exec_cmd = 'touch {}'.format(tmp_filename_xyz)
                os.system(exec_cmd)
                tmp_f = open(tmp_filename_xyz, 'w')
                for point in data:
                    tmp_f.write('{} {} {}\n'.format(point[0], point[1], point[-1]))
                tmp_f.close()   
                exec_cmd = ' '.join(['./voronoi -f', tmp_filename_xyz, '-R', str(R), '-r', str(r), '-t', str(threshold), '-o', out_directory])
                os.system(exec_cmd)
                temp_f = open(out_directory+'/'+'points_classification_voronoi_tmp_{}.txt'.format(i))
                classification[i] = np.array(list(map(int, temp_f.readline().split(' ')[:-1])))[:, np.newaxis]
                os.remove(tmp_filename_xyz)
                os.remove(out_directory+'/'+'points_classification_voronoi_tmp_{}.txt'.format(i))
                
            classif_data = classification
            #metric_results.append(metric(classification, gt))
        #average_metric = sum(metric_results)/len(metric_results)
        #print('Metric results (avg balanced acc): {}'.format(average_metric))
            
