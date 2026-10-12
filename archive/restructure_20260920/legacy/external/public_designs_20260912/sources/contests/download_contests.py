"""Bounded downloads from official public contest websites; atomic publication."""
import concurrent.futures, datetime, gzip, hashlib, json, pathlib, shutil, urllib.request
BASE=pathlib.Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def fetch(rec):
 url=rec['url']; folder=BASE/'ispd2008_gr';folder.mkdir(exist_ok=True)
 dest=folder/url.rsplit('/',1)[-1]; part=dest.with_name(dest.name+'.partial')
 if not dest.exists():
  with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as out:
   n=0
   while b:=response.read(4*1024*1024):
    n+=len(b)
    if n>3_000_000_000:raise ValueError('File exceeds 3 GB guard')
    out.write(b)
  if rec.get('length') and part.stat().st_size!=int(rec['length']):raise ValueError('Content length mismatch')
  part.replace(dest)
 expanded=dest.with_suffix('')
 if not expanded.exists():
  partial=expanded.with_name(expanded.name+'.partial')
  with gzip.open(dest,'rb') as inp,partial.open('wb') as out:
   n=0
   while b:=inp.read(4*1024*1024):
    n+=len(b)
    if n>3_000_000_000:raise ValueError('Decompressed file exceeds 3 GB guard')
    out.write(b)
  partial.replace(expanded)
 design=dest.name.split('.')[0]
 row={'design_id':'ispd2008_gr_'+design,'family':'ispd2008_global_routing','design_family':design,'path':str(expanded),'parser':'contests.parse_design','format':'ISPD07/08 global-routing ASCII; physical XY, layer ignored only on 2D projection','source_url':url,'source_page':'https://www.ispd.cc/contests/08/ispd08rc.html','download_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_path':str(dest),'archive_bytes':dest.stat().st_size,'archive_sha256':sha(dest),'bytes':expanded.stat().st_size,'sha256':sha(expanded),'deduplication_note':('Also an ISPD 2007 3D contest input; included once in the ISPD 2008 16-design suite.' if '/07/' in url else 'ISPD 2008 new benchmark.'),'license_note':'Official publicly downloadable academic contest benchmark; no standalone data license specified on source page. Industrial ASIC derived; no unrestricted redistribution claim.'}
 print(json.dumps({'downloaded':row['design_id'],'bytes':row['bytes']}),flush=True)
 return row

if __name__=='__main__':
 heads=json.loads((BASE/'ispd08_urls_head.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(fetch,heads))
 target=BASE/'designs_manifest.json';temp=target.with_suffix('.json.partial');temp.write_text(json.dumps({'schema_version':1,'designs':rows,'count':len(rows),'normalization':'Parser preserves original pin order and duplicates; no normalization/deduplication/degree conversion.'},indent=2)+'\n');temp.replace(target)
 print('DONE',len(rows),flush=True)
