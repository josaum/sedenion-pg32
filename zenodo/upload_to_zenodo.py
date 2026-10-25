#!/usr/bin/env python3
"""Create a Zenodo deposit for the dual-pol package.
Usage:  ZENODO_TOKEN=... python3 zenodo/upload_to_zenodo.py zenodo/sedenion_dualpol_pivot12_2026-10-25.zip [--sandbox] [--publish]
Without --publish the deposit is left as a draft for review in https://zenodo.org/me/uploads.
Requires a personal access token with deposit:write (and deposit:actions for --publish)."""
import json, os, sys, urllib.request, pathlib
args=[a for a in sys.argv[1:] if not a.startswith('--')]
zip_path=pathlib.Path(args[0]); meta=json.load(open(pathlib.Path(__file__).with_name('.zenodo.json')))
base='https://sandbox.zenodo.org/api' if '--sandbox' in sys.argv else 'https://zenodo.org/api'
tok=os.environ.get('ZENODO_TOKEN') or sys.exit('set ZENODO_TOKEN')
def req(url, data=None, method='GET', ctype='application/json'):
    r=urllib.request.Request(url, data=data, method=method, headers={'Authorization':f'Bearer {tok}', **({'Content-Type':ctype} if data is not None else {})})
    with urllib.request.urlopen(r) as resp: return json.loads(resp.read() or b'{}')
dep=req(f'{base}/deposit/depositions', json.dumps({'metadata':meta}).encode(), 'POST')
bucket=dep['links']['bucket']
req(f"{bucket}/{zip_path.name}", zip_path.read_bytes(), 'PUT', 'application/octet-stream')
print('draft deposit:', dep['links']['html'])
if '--publish' in sys.argv:
    pub=req(dep['links']['publish'], b'', 'POST'); print('published DOI:', pub['doi'])
