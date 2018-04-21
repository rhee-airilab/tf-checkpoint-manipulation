#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function,division,absolute_import
import os
import sys
import glob
import re

# model_checkpoint_path: "model-9040"
# all_model_checkpoint_paths: "model-8588"

def save_cleanup(dry_run=True):
    # read checkpoint
    with open('checkpoint','r') as f:
        # example: model_checkpoint_path: "model-9040"
        first     = f.readline(4000).strip()
        pattern   = 'model_checkpoint_path: "'
        assert first.startswith(pattern),first
        model     = first[len(pattern):-1]
        m         = re.match(r'^(.*)-[0-9]+$',model)
        prefix    = m[1]

        print('found last model',model,prefix)

        for f in glob.glob('*'):
            if not os.path.isdir(f):
                if f.startswith(prefix) and not f.startswith(model+'.'):
                    print('rm -fv "{:s}"'.format(f))
                    if not dry_run:
                        os.unlink(f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes',action='store_true')
    parser.add_argument('dirs',nargs='*')
    args = parser.parse_args()
    if args.dirs:
        pdir = os.getcwd()
        for dir in args.dirs:
            os.chdir(os.path.join(pdir,dir))
            save_cleanup(dry_run=not args.yes)
    else:
        save_cleanup(dry_run=not args.yes)
