import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('version')
parser.add_argument('--skip-build', action='store_true', default=False)

args = parser.parse_args()

if not args.skip_build:
    os.system('python -m nuitka --onefile game.py')

dirname = f'hydroplane-{args.version}'
os.mkdir(dirname)
os.system(f'cp *.glb *.ogg *.wav *.png *.vert *.frag game.bin {dirname}')
os.system(f'zip {dirname}.zip -r {dirname}')
