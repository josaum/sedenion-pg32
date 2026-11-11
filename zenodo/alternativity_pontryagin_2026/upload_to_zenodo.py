#!/usr/bin/env python3
"""Zenodo deposit helper (synchronized with sedenion-pg32/zenodo/upload_to_zenodo.py).

New standalone deposit (metadata read automatically from sibling .zenodo.json):
    ZENODO_TOKEN=... python3 docs/sedenion_pg32_papers_and_patent/upload_to_zenodo.py \
        docs/sedenion_pg32_papers_and_patent/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf \
        docs/sedenion_pg32_papers_and_patent/Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip \
        [--sandbox] [--publish]

New VERSION of an existing record (e.g. 10.5281/zenodo.23588274):
    ZENODO_TOKEN=... python3 docs/sedenion_pg32_papers_and_patent/upload_to_zenodo.py \
        docs/sedenion_pg32_papers_and_patent/Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip \
        --new-version-of 23588274 --version 2026.11.10 [--keep-files] [--publish]

Without --publish the deposit is left as a draft to review at https://zenodo.org/me/uploads.
--keep-files keeps the previous version's files in the new version (default replaces them with the
files given on the command line). --sandbox targets sandbox.zenodo.org. Token scopes: deposit:write
(+ deposit:actions for --publish)."""
import json, os, sys, urllib.request, urllib.error, pathlib

args = sys.argv[1:]
def flag(name, default=None):
    if name in args:
        i = args.index(name); v = args[i+1]; del args[i:i+2]; return v
    return default
new_version_of = flag('--new-version-of'); version = flag('--version')
publish = '--publish' in args; keep = '--keep-files' in args; sandbox = '--sandbox' in args
files = [pathlib.Path(a) for a in args if not a.startswith('--')]
if not files: sys.exit(__doc__)
base = 'https://sandbox.zenodo.org/api' if sandbox else 'https://zenodo.org/api'
tok = os.environ.get('ZENODO_TOKEN') or sys.exit('set ZENODO_TOKEN')

def req(url, data=None, method='GET', ctype='application/json'):
    hdr = {'Authorization': f'Bearer {tok}'}
    if data is not None: hdr['Content-Type'] = ctype
    r = urllib.request.Request(url, data=data, method=method, headers=hdr)
    try:
        with urllib.request.urlopen(r) as resp: return json.loads(resp.read() or b'{}')
    except urllib.error.HTTPError as e:
        sys.exit(f'{method} {url} -> {e.code}: {e.read().decode()[:500]}')

if new_version_of:
    nv = req(f'{base}/deposit/depositions/{new_version_of}/actions/newversion', b'', 'POST')
    dep = req(nv['links']['latest_draft'])
    if not keep:
        for f in dep.get('files', []): req(f['links']['self'], None, 'DELETE')
    md = dep['metadata']
    if version: md['version'] = version
    md.pop('doi', None); md.pop('prereserve_doi', None)
    dep = req(dep['links']['self'], json.dumps({'metadata': md}).encode(), 'PUT')
else:
    meta = json.load(open(pathlib.Path(__file__).with_name('.zenodo.json')))
    dep = req(f'{base}/deposit/depositions', json.dumps({'metadata': meta}).encode(), 'POST')

bucket = dep['links']['bucket']
for f in files:
    req(f'{bucket}/{f.name}', f.read_bytes(), 'PUT', 'application/octet-stream'); print('uploaded', f.name)
print('draft:', dep['links']['html'])
if publish:
    pub = req(dep['links']['publish'], b'', 'POST'); print('published DOI:', pub['doi'])
