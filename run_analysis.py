"""Reproduce all reported results using the included feature CSV.

Usage: python run_analysis.py            # complete nested-CV evaluation
       python run_analysis.py --smoke    # quick data/baseline test only
"""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='Only run imports/data/baselines to verify setup')
args = parser.parse_args()
base = Path(__file__).resolve().parent
notebook = nbformat.read(base/'UrbanSound.ipynb',as_version=4)
# The 6 GB raw-WAV feature-extraction section is optional. CSV is bundled.
filtered=[]
inside_audio=False
for cell in notebook.cells:
    source=cell.source
    if cell.cell_type=='markdown' and '### **1. Loading & feature Extraction' in source:
        inside_audio=True
        continue
    if cell.cell_type=='markdown' and '### **2. Implementing Baseline Classifiers' in source:
        inside_audio=False
    if inside_audio:
        continue
    filtered.append(cell)
notebook.cells=filtered
if args.smoke:
    # Verify imports, data, and the first figure without expensive model runs.
    end=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and '#All-positive' in c.source)
    notebook.cells=notebook.cells[:end]
print('Executing',len(notebook.cells),'cells from the bundled CSV...',flush=True)
client=NotebookClient(notebook,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(base)}})
client.execute()
out=base/('UrbanSound_smoke.ipynb' if args.smoke else 'UrbanSound_reproduced.ipynb')
nbformat.write(notebook,out)
if not args.smoke:
    for cell in reversed(notebook.cells):
        if cell.cell_type=='code' and 'REPRODUCED HEADLINE' in cell.source:
            for output in cell.get('outputs',[]):
                if output.output_type=='stream':print(output.text,end='')
            break
print('Saved:',out)
