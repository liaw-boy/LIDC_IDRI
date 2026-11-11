"""Lay out downloaded DICOM series + LIDC XML the way the original scripts expect:
   tree/LIDC-IDRI-XXXX/<SeriesUID>/{*.dcm, <name>.xml}   (files are symlinks)"""
import json, os, sys, glob
D = '/mnt/legend800/lbw_project/lidc_data'
sel = json.load(open(f'{D}/selection.json')); sx = json.load(open(f'{D}/series_xml.json'))
n = 0
for s in sel:
    src = f"{D}/dcm/{s['uid']}"
    if not os.path.isdir(src): continue
    xmls = [x for x in sx.get(s['uid'], []) if 'old' not in os.path.basename(x)] or sx[s['uid']]
    dst = f"{D}/tree/{s['pid']}/{s['uid']}"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f'{src}/*.dcm'):
        l = f"{dst}/{os.path.basename(f)}"
        if not os.path.lexists(l): os.symlink(f, l)
    x = sorted(xmls)[-1]
    open(f"{dst}/{os.path.basename(x)}", 'wb').write(open(f'{D}/xml_zip/'+x if not x.startswith('/') else x, 'rb').read())
    n += 1
print('series laid out:', n)
