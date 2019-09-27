#!/usr/bin/env python3

import argparse


def main(options):
    print(options.__dict__)


def parse_args():
    parser = argparse.ArgumentParser()

    # all the original args go here
    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')
    parser.add_argument('--infer-from-spec', dest='infer_from_spec', action='store_true', default=False,
                        help='if set, --model, --save-model-file, --logging-file, --tboard-json-logging-file,'
                             'and --tboard-dir are formed automatically [default: False].')
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')

    # add two more args corresponding to the parsing of job args
    parser.add_argument('-jac', '--job-array-config', dest='job_array_config',
                        help='Job array-formatted file (one set of commandline parameters per file).')
    parser.add_argument('-jati', '--job-array-task-id', dest='job_array_task_id', type=int,
                        help='Task ID to parse from the job array-formatted file.')

    # parse the actual commandline
    options = parser.parse_args()

    # get a parser
    for action in parser._actions:
        action.required = False  # so that we won't get errors when the action
        action.default = None  # so that we won't get falsely set args when they're not really encountered

    # parse the job array command line
    with open(options.job_array_config) as jac_file:
        lines = [line.strip() for line in jac_file.readlines()]
        job_array_options = parser.parse_args(lines[options.job_array_task_id].split())

    for name, value in job_array_options.__dict__.items():
        if None is not value:
            setattr(options, name, value)

    return options


if __name__ == '__main__':
    options = parse_args()
    main(options)
