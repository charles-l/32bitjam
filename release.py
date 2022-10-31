#!/usr/bin/env python
import os
import argparse
import glob
import zipfile
parser = argparse.ArgumentParser()

parser.add_argument('version')
parser.add_argument('--skip-build', action='store_true', default=False)

args = parser.parse_args()

if not args.skip_build:
    os.system('python -m nuitka --onefile game.py')

dirname = f'hydroplane-{args.version}'
files = sum([glob.glob(p) for p in '*.glb *.ogg *.wav *.png *.vert *.frag game.bin game.exe'.split()], [])
with zipfile.ZipFile(f'{dirname}.zip', 'w') as zipf:
    for f in files:
        zipf.write(f, arcname=f'{dirname}/{f}')
