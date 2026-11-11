"""Run the ORIGINAL classification_cnn/dataloader.py (ROI 32x32 + full-CT PNGs per malignancy class)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import load_original
D = '/mnt/legend800/lbw_project/lidc_data'
mod = load_original('/mnt/legend800/lbw_project/LIDC_IDRI/classification_cnn/dataloader.py',
                    [("'D:/LUNA16'", f"'{D}/tree'"), ("'./outputct+roi_0406'", f"'{D}/cls_data'")])
mod.main()
