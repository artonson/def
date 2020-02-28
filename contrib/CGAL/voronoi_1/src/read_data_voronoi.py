import os
import sys
import h5py
import numpy as np
import tempfile
import subprocess

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
   
    tmp_directory = os.path.join(out_directory, 'voronoi_tmp')
    os.makedirs(os.path.join(out_directory, tmp_directory), exist_ok=True)
       
    metric_results = []
    files = os.listdir(directory)
    
    for filename in files:
        
        if not filename.endswith('.hdf5') or filename.endswith('.h5'):
            continue
            
        print('Dataset from {}'.format(filename))
        out_filename = filename.split('.')[0] + '_voronoi_output.h5'     
        out_path = os.path.join(out_directory, out_filename)
        
        out = h5py.File(out_path, 'w')
        
        with h5py.File(os.path.join(directory, filename), 'r') as f:
            
            classif_data = out.create_dataset("classification", (f['points'].shape[0], f['points'].shape[1], 1))
            classification = np.zeros((f['points'].shape[0], f['points'].shape[1], 1))
            
            for i, data in enumerate(f['points']):
                
                print('datafile #{}'.format(i))
                
                tmp_filename_xyz = os.path.join(tmp_directory, 'tmp_{}.xyz'.format(i))                
                with open(tmp_filename_xyz, 'w') as tmp_f:
                    for point in data:
                        tmp_f.write('{} {} {}\n'.format(point[0], point[1], point[-1]))
                                
                exec_cmd = [
                    './voronoi', '-f', tmp_filename_xyz, 
                    '-R', str(R), 
                    '-r', str(r), 
                    '-t', str(threshold), 
                    '-o', tmp_directory
                ]
                print('Running {}'.format(' '.join(exec_cmd)))
                subprocess.run(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                tmp_filename_classification = os.path.join(tmp_directory, 'points_classification_tmp_{}.txt'.format(i))
                with open(tmp_filename_classification) as tmp_f:
                    classification[i] = np.array(list(map(int, tmp_f.readline().split(' ')[:-1])))[:, np.newaxis]
                                
            classif_data = classification 
            
        out.close()
            #metric_results.append(metric(classification, gt))
        #average_metric = sum(metric_results)/len(metric_results)
        #print('Metric results (avg balanced acc): {}'.format(average_metric))
            
