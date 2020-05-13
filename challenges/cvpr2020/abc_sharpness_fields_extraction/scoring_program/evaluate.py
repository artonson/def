#!/usr/bin/env python

import sys
import os.path
import yaml
import numpy as np

from utilities import (
    read_txt_file,
    read_targz_file,
    mean_error_field,
)


# data structure presupposed:
# <input-dir>/
#   |
#   +-- ref/
#   |    |
#   |    +-- evaluate.yml
#   |    +-- high_res_[test|val]_0_target.tar.gz
#   |
#   +-- res/
#        |
#        +-- field_000001_target.txt
#        +-- field_000002_target.txt
#        +-- ...
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

    reference_filename = config['reference_file']
    filenames_by_split = config['expected_files']

    # read GT targets off disk
    _, ref_targets = read_targz_file(
        os.path.join(reference_dir, reference_filename),
        read_points=False,
    )

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as output_file:

        for split, filenames in filenames_by_split.items():

            # this computes mean value of a metric for all files in the dataset
            metrics = []
            for idx, filename in enumerate(filenames):
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
                    rmse = mean_error_field(np.zeros_like(ref_target), np.ones_like(ref_target))
                else:
                    try:
                        rmse = mean_error_field(ref_target, pred_target)
                    except Exception as e:
                        rmse = mean_error_field(np.zeros_like(ref_target), np.ones_like(ref_target))
                        print('Encountered error computing quality for file {filename}: {what}'.format(
                            filename=filename, what=str(e)
                        ))
                print(filename, rmse)
                metrics.append(rmse)

            mean_rmse = np.mean(metrics)

            output_file.write(
                '{split}_{metric}_score:{value}'.format(
                    split=split,
                    metric='rmse',
                    value=mean_rmse
                )
            )
