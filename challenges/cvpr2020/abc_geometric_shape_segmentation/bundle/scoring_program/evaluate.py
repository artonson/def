#!/usr/bin/env python

import sys
import os.path
import yaml
import numpy as np

from utilities import (
    read_txt_file,
    read_targz_npy_file,
    segmentation_balanced_accuracy_error,
    segmentation_iou_error,
)


# data structure presupposed:
# <input-dir>/
#   |
#   +-- ref/
#   |    |
#   |    +-- evaluate.yml
#   |    +-- high_res/
#   |    |      +-- high_res_[test|val]_0_target.tar.gz
#   |    +-- med_res/
#   |    |      +-- med_res_[test|val]_0_target.tar.gz
#   |    +-- low_res/
#   |           +-- low_res_[test|val]_0_target.tar.gz
#   |
#   +-- res/
#        |
#        +-- high_res/
#        |      +-- field_000001_target.txt
#        |      +-- field_000002_target.txt
#        |      +-- ...
#        +-- med_res/
#        |      +-- field_000001_target.txt
#        |      +-- field_000002_target.txt
#        |      +-- ...
#        +-- low_res/
#        |      +-- field_000001_target.txt
#        |      +-- field_000002_target.txt
#               +-- ...
#
# <output-dir>
#   |
#   +-- scores.txt


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submission_dir = os.path.realpath(os.path.join(input_dir, 'res'))
reference_dir = os.path.realpath(os.path.join(input_dir, 'ref'))

if not os.path.isdir(submission_dir):
    print("{} doesn't exist".format(submission_dir))

if os.path.isdir(submission_dir) and os.path.isdir(reference_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluate_yml_filename = os.path.join(reference_dir, 'evaluate.yml')
    with open(evaluate_yml_filename) as fi:
        config = yaml.load(fi)

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as output_file:

        for split, split_config in config.items():

            reference_filename = split_config['reference_file']
            filenames = split_config['expected_files']
            # read GT targets off disk
            _, ref_targets = read_targz_npy_file(
                os.path.join(reference_dir, split, reference_filename),
                read_points=False,
            )

            # this computes mean value of a metric for all files in the dataset
            balanced_accuracy_scores = []
            iou_scores = []
            for idx, filename in enumerate(filenames):
                count_iou = True
                ref_target = ref_targets[idx]
                try:
                    _, pred_target = read_txt_file(
                        os.path.join(submission_dir, split, filename),
                        read_points=False,
                    )
                except Exception as e:
                    print('Encountered error reading file {filename}: {what}'.format(
                        filename=filename, what=str(e)
                    ))
                    # score the file using the lowest possible metric value
                    acc = 0.
                    iou = 0.
                else:
                    try:
                        acc = segmentation_balanced_accuracy_error(ref_target, pred_target)
                    except Exception as e:
                        acc = 0.
                        print('Encountered error computing Acc for file {filename}: {what}'.format(
                            filename=filename, what=str(e)
                        ))
                    if not np.any(ref_target):
                        # don't try computing IoU if no ground truth sharpness
                        if np.any(pred_target):
                            iou = 0.
                        else:
                            count_iou = False
                    else:
                        # try to perform the actual computation
                        try:
                            iou = segmentation_iou_error(ref_target, pred_target)
                        except Exception as e:
                            iou = 0.
                            print('Encountered error computing IoU for file {filename}: {what}'.format(
                                filename=filename, what=str(e)
                            ))
                balanced_accuracy_scores.append(acc)
                if count_iou:
                    iou_scores.append(iou)

            mean_balanced_accuracy = np.mean(balanced_accuracy_scores)
            output_file.write(
                '{split}_{metric}_score:{value}\n'.format(
                    split=split,
                    metric='balanced_accuracy',
                    value=mean_balanced_accuracy
                )
            )
            mean_iou = np.mean(iou_scores)
            output_file.write(
                '{split}_{metric}_score:{value}\n'.format(
                    split=split,
                    metric='iou',
                    value=mean_iou
                )
            )
