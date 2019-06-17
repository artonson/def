import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_h5_dataset_n

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn_sharp_n', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=5000, help='Point Number [default: 5000]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()




EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 1

#DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
#TEST_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=FLAGS.category, split='test')
TEST_DATASET = modelnet_h5_dataset_n.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/4pps_sel_short_n/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes=1):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl)
        loss_tensor = MODEL.get_loss_tensor(pred, labels_pl)
        pred = tf.sigmoid(pred)
        tf.summary.scalar('loss', loss)
        tf.summary.tensor_summary('loss_tensor', loss_tensor)
        #losses = tf.get_collection('losses')
        #total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'loss_tensor': loss_tensor}

    pcs, losses, preds, labels = eval_one_epoch(sess, ops, num_votes)
    return pcs, losses, preds, labels

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,6))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_POINT,1))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    pcs = []
    losses = []
    preds = []
    labels = []

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize,...] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
           
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            loss_val, loss_tensor_val, pred_val = sess.run([ops['loss'], ops['loss_tensor'], ops['pred']], feed_dict=feed_dict)
            
        log_string('eval loss: %f' % (loss_val))
        loss_sum += loss_val
        batch_idx += 1
        pcs.append(batch_data)
        losses.append(loss_tensor_val)
        preds.append(pred_val)
        labels.append(batch_label)
        #for i in range(bsize):
            #l = batch_label[i]
            #total_seen_class[l] += 1
            #total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    return np.array(pcs).reshape(-1, NUM_POINT, 6), np.array(losses).reshape(-1, NUM_POINT), np.array(preds).reshape(-1, NUM_POINT, 1), np.array(labels).reshape(-1, NUM_POINT, 1)


if __name__=='__main__':
    with tf.Graph().as_default():
        pcs, losses, preds, labels = evaluate()
    LOG_FOUT.close()
    
    file = open('../../sharp/dataset_fine/abc_05_sharp_1024_patches_normalized_4pps_selected_short_curves_surfaces_names_test.txt', 'r')
    names = file.readlines()
    file.close()
    file = open('../../sharp/dataset_fine/abc_06_sharp_1024_patches_normalized_4pps_selected_short_curves_surfaces_names_test.txt', 'r')
    names2 = file.readlines()
    file.close()
    for name in names2:
        names.append(name)

    '''file = open('../../sharp/dataset_fine/abc_05_06_sharp_1024_patches_normalized_4pps_selected_neural_v3_names_test.txt', 'r')
    names = file.readlines()
    file.close()'''
    for i in range(len(pcs)):
        out = np.hstack((pcs[i], preds[i].reshape(len(labels[i]),1), labels[i].reshape(len(labels[i]),1), losses[i].reshape(len(losses[i]),1)))
        np.savetxt("../../sharp/networks/predictions/dgcnn/log_4pps_sel_cs_39k_11ep_n_aug/%s.csv" % names[i][8:-1], out, delimiter=",", header='x,y,z,nx,ny,nz,s,s_t,loss')
