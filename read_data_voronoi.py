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
        if (sys.argv[i] == '-filename' or sys.argv[i] == '-f'):
            filename = sys.argv[i+1]
        if (sys.argv[i] == '-o'):
            out_filename = sys.argv[i+1]
        if (sys.argv[i] == '-R'):
            R = sys.argv[i+1]
        if (sys.argv[i] == '-r'):
            r = sys.argv[i+1]
        if (sys.argv[i] == '-t' or sys.argv[i] == '-threshold'):
            threshold = sys.argv[i+1]
    print('Dataset from {}'.format(filename))
    print('Voronoi Method 1, parameters R={}, r={}, t={}'.format(R, r, threshold))
    exec_cmd = 'g++ -o voronoi voronoi_1.cpp -lCGAL -I/home/CGAL-4.14/include -lgmp'
    os.system(exec_cmd)
    metric_results = []
    out = h5py.File(out_filename, 'w')
    with h5py.File(filename, 'r') as f:
        classif_data = out.create_dataset("classification", (f['data'].shape[0], f['data'].shape[1], 1))
        classification = np.zeros((f['data'].shape[0], f['data'].shape[1], 1))
        for i, data in enumerate(f['data']):
            print('datafile #{}'.format(i))
            tmp_f = open('tmp.xyz', 'w')
            for point in data:
                tmp_f.write('{} {} {}\n'.format(point[0], point[1], point[-1]))
            tmp_filename = 'tmp.xyz'
            tmp_f.close()
             
            exec_cmd = ' '.join(['./voronoi -filename', tmp_filename, '-R', str(R), '-r', str(r), '-t', str(threshold)])
            os.system(exec_cmd)
            temp_f = open('points_classification_' + tmp_filename.split('.')[0] + '.txt')
            classification[i] = np.array(list(map(int, temp_f.readline().split(' ')[:-1])))[:, np.newaxis]

            os.remove(tmp_filename)
            break
        classif_data = classification
            #metric_results.append(metric(classification, gt))
        #average_metric = sum(metric_results)/len(metric_results)
        #print('Metric results (avg balanced acc): {}'.format(average_metric))
            
