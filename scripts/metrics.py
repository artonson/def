#!/usr/bin/env python3

# Intended usage of this script:
#
# 1. Compute the quality measures.
# -------------------------------
# Usage:
# ./metrics.py compute -t ground_truth_dir -p pred_dir \
#                      -l distances -m regress_sharpdf -m MSELoss \
#                      -s num_surfaces -b 10 -a 10 -w 10 -g 100 \
#                      -o metrics.json
#
# This will take each file in ground_truth_dir, look for a file with the same filename in pred_dir,
# load both files and compute metrics 'regress_sharpdf' and 'MSELoss', section them using values
# num_surfaces, and output 10 best, 10 avg, and 10 worst instances according to each metric.
#
#
# 2. Form an HTML report to show to people.
# ----------------------------------------
# Usage example:
# ./metrics.py report -i voronoi.json -i ours.json \
#                     -o report.html
#
#



import argparse
import glob
import json
import os
import sys
from collections import defaultdict, OrderedDict

import torch
import torch.nn
import h5py

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from sharpf.modules.losses import LOSSES, get_loss_function


def compute_metrics(options):
    filenames = sorted(glob.glob(os.path.join(options.true_dir, '*.hdf5')))

    split_by_values = None
    metrics_by_name = defaultdict(dict)  # metric_name -> filename -> {values, split_values}
    for true_pathname in filenames:

        true_filename = os.path.basename(true_pathname)
        if options.verbose:
            print('=== Reading GT file %s ===' % true_filename)
        with h5py.File(true_pathname, 'r') as f:
            true_label = torch.from_numpy(f[options.target_label][:])
            if None is not options.split_by:
                split_by_values = torch.from_numpy(f[options.split_by][:])

        pred_pathname = os.path.join(options.pred_dir, true_filename)
        if options.verbose:
            print('=== Reading pred file %s ===' % os.path.basename(pred_pathname))
        with h5py.File(pred_pathname, 'r') as f:
            pred_label = torch.from_numpy(f[options.target_label][:])

        for metric_name in options.metrics:
            if options.verbose:
                print('=== Computing %s ===' % metric_name)
            try:
                loss_function = get_loss_function(metric_name, reduction='none')
            except Exception as e:
                print(str(e), file = sys.stderr)
                continue

            loss_values = loss_function(true_label, pred_label)
            metrics_by_name[metric_name][true_filename] = {
                'values_by_instance': loss_values.mean(1),
                'split_by': split_by_values
            }

    def compute_statistics(loss_values, indexes=None):
        n_instances = len(loss_values)
        if None is indexes:
            indexes = torch.arange(n_instances)
        ascending_idx = torch.argsort(loss_values)
        mean_value = loss_values.mean()
        std_value = loss_values.std()
        values_hist = torch.histc(loss_values, bins=options.n_histogram_bins)
        bins_hist = torch.linspace(torch.min(loss_values),
                                   torch.max(loss_values),
                                   options.n_histogram_bins)

        statistics = {
            'values_by_instance': loss_values.tolist(),
            'mean_value': mean_value.item(),
            'std_value': std_value.item(),
            'values_hist': values_hist.tolist(),
            'bins_hist': bins_hist.tolist(),
        }
        if None is not options.n_best_instances:
            statistics['best_ids'] = indexes[ascending_idx[:options.n_best_instances]].tolist()

        if None is not options.n_worst_instances:
            statistics['worst_ids'] = indexes[ascending_idx[-options.n_worst_instances:]].tolist()

        if None is not options.n_avg_instances:
            # average here denotes "close to median"
            avg_min_idx = n_instances // 2 - options.n_avg_instances // 2
            avg_max_idx = n_instances // 2 + options.n_avg_instances // 2
            statistics['avg_ids'] = indexes[ascending_idx[avg_min_idx:avg_max_idx]].tolist()

        return statistics

    if options.verbose:
        print('=== Creating dictionary ===')

    statistics_by_metric = defaultdict(OrderedDict)
    for metric_name in options.metrics:
        metric_by_filename = metrics_by_name[metric_name].values()

        if options.verbose:
            print('    computing default metrics')

        loss_values = torch.cat([value['values_by_instance'] for value in metric_by_filename])
        statistics_by_metric[metric_name].update({
            'default': compute_statistics(loss_values)
        })

        if None is not options.split_by:
            split_by_values = torch.cat([value['split_by'] for value in metric_by_filename])
            unique_split_by_values = sorted(torch.unique(split_by_values).tolist())
            for value in unique_split_by_values:
                key = '{}={}'.format(options.split_by, value)

                if options.verbose:
                    print('    computing metrics for {}'.format(key))

                selector = (split_by_values == value)
                statistics_by_metric[metric_name].update({
                    key: compute_statistics(loss_values[selector], indexes=selector.nonzero().reshape((-1)))
                })

    metric_names = sorted(statistics_by_metric.keys())
    method_slices = []
    for slice_name in statistics_by_metric[metric_names[0]].keys():
        slice_metrics = [
            {
                'metric_name': metric_name,
                'metric_data': statistics_by_metric[metric_name][slice_name]
            } for metric_name in metric_names
        ]
        method_slices.append({
            'slice_name': slice_name,
            'slice_metrics': slice_metrics,
        })

    if options.verbose:
        print('=== Dumping to JSON ===')

    with open(options.save_filename, 'w') as write_file:
        json.dump(statistics_by_metric, write_file)

    if options.verbose:
        print('=== Finished ===')


def generate_report(options):
    from jinja2 import Environment, FileSystemLoader

    metric_names = []
    methods_metrics = []
    for filename in options.input_filename:
        method_name = os.path.splitext(os.path.basename(filename))[0],
        with open(filename) as input_file:
            method_info = json.load(input_file)
            if not metric_names:
                metric_names = method_info['metric_names']
            else:
                if metric_names != method_info['metric_names']:
                    raise ValueError('Inconsistent sets of metrics defined in files.')

            methods_metrics.append({
                'method_name': method_name,
                'method_slices': method_info['method_slices'],
            })

    templates_dir = os.path.join(__dir__, 'scripts')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('metrics_template.html')

    with open(options.output_filename, 'w') as fh:
        fh.write(template.render(
            metric_names=metric_names,
            methods_metrics=methods_metrics,
            display_slices=options.display_slices,
        ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')

    subparsers = parser.add_subparsers(help='sub-command help')

    compute_parser = subparsers.add_parser('compute', help='compute command help')
    compute_parser.add_argument('-t', '--true-dir', dest='true_dir',
                                required=True, help='Path to GT')
    compute_parser.add_argument('-p', '--pred-dir', dest='pred_dir',
                                required=True, help='Path to prediction')
    compute_parser.add_argument('-l', '--target-label', dest='target_label',
                                required=True, help='Target label to look for in both datafiles.')
    compute_parser.add_argument('-o', '--save-filename', default='save_filename',
                                required=True, help='Path to save to (JSON file).')
    compute_parser.add_argument('-m', '--metric', required=True, dest='metrics', action='append',
                                choices=LOSSES, help='Choose loss function.')
    compute_parser.add_argument('-s', '--split-by', dest='split_by',
                                help='If set, specifies a variable in GT file '
                                     'where statistics is computed separately for each unique value.')
    compute_parser.add_argument('-b', '--best', dest='n_best_instances', type=int, default=None,
                                help='Number of instances with the lowest value of each metric to save.')
    compute_parser.add_argument('-a', '--average', dest='n_avg_instances', type=int, default=None,
                                help='Number of instances with an average value of each metric to save.')
    compute_parser.add_argument('-w', '--worst', dest='n_worst_instances', type=int, default=None,
                                help='Number of instances with the highest value of each metric to save.')
    compute_parser.add_argument('-g', '--histogram-bins', dest='n_histogram_bins', type=int, default=10,
                                help='Number of histogram bins for of each metric to save.')
    compute_parser.set_defaults(func=compute_metrics)

    report_parser = subparsers.add_parser('report', help='report command help')
    report_parser.add_argument('-i', '--input-file', default='input_filename', action='append',
                               required=True, help='Path to read from (JSON file).')
    report_parser.add_argument('-o', '--output-file', default='output_filename',
                               required=True, help='Path to save to (HTML file).')
    report_parser.add_argument('-t', '--true-dir', dest='true_dir',
                               required=True, help='Path to GT')
    report_parser.add_argument('-p', '--pred-path', dest='pred_dir',
                               required=True, help='Path to prediction')
    report_parser.add_argument('-ns', '--no-slices', dest='display_slices', action='store_false',
                               detault=True, help='If set, slices will be turned off.')
    report_parser.set_defaults(func=generate_report)

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    options.func(options)