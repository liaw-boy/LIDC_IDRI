"""Helpers for running the ORIGINAL preprocessing scripts unchanged, only faster.

The original find_dicom_file() re-reads every .dcm in the folder for each SOP UID lookup
(O(n^2) dcmread). _cached_find returns the same result via a per-folder SOP->path index.
"""
import importlib.util, os, threading
import pydicom

_index, _lock = {}, threading.Lock()

def _cached_find(sop_uid, directory):
    with _lock:
        if directory not in _index:
            m = {}
            for f in os.listdir(directory):
                if f.endswith('.dcm'):
                    try:
                        ds = pydicom.dcmread(os.path.join(directory, f), stop_before_pixels=True)
                        m[ds.SOPInstanceUID] = os.path.join(directory, f)
                    except Exception:
                        continue
            _index[directory] = m
    return _index[directory].get(sop_uid)

def load_original(path, replacements=()):
    src = open(path, encoding='utf-8').read()
    for a, b in replacements:
        assert a in src, a
        src = src.replace(a, b)
    spec = importlib.util.spec_from_loader('orig_mod', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    exec(compile(src, path, 'exec'), mod.__dict__)
    mod.find_dicom_file = _cached_find
    return mod
