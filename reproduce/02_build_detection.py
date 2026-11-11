"""Run the ORIGINAL preprocessing/ann_v1.py (nodule + non-nodule slices, MIN_DOCTORS=2) over the tree."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import load_original
T = '/mnt/legend800/lbw_project/lidc_data/tree'
mod = load_original('/mnt/legend800/lbw_project/LIDC_IDRI/preprocessing/ann_v1.py')
os.chdir(T)
mod.main()
