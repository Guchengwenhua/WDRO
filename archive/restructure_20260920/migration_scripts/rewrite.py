#!/usr/bin/env python3
"""Rewrite maintained code and reading documents using the recorded relocation map."""
from pathlib import Path
import json,os,re,hashlib,shutil
R=Path('/remote-home/Guilin/DRO/Codex'); A=R/'archive/restructure_20260920'
mapping=json.loads((A/'path_map.json').read_text())
mapping.update({'mainline/precision_v2':'wdro_precision_v2','mainline/legacy_v1':'wdro_legacy_v1/wdro_rest_min',
 'improvements/adversarial_v3_1':'adversarial_v3_1_20260912','external/public_designs_20260912':'benchmarks/public_designs_20260912',
 'external/v31_models_20260912':'benchmarks/v31_models_20260912'})
order=sorted(mapping.items(),key=lambda x:-len(x[0]))
def mapped(rel):
 for a,b in order:
  if rel==a or rel.startswith(a+'/'):return b+rel[len(a):]
 return rel
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((A/'frozen_files.json').read_text())
changed=[]; docs=[];code=[]
link_re=re.compile(r'(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)')
def absolute_strings(text):
 # First replace full absolute paths in a single pass, preventing nested replacement.
 pattern=re.compile(re.escape(str(R))+'/('+ '|'.join(re.escape(a) for a,b in order)+r')(?=/|[\s\x27\x22`),;:]|$)')
 return pattern.sub(lambda m:str(R/mapping[m.group(1)]),text)
for row in frozen:
 old=row['old_path']; new=mapped(old);p=R/new
 if old.startswith(('archive/','mainline/','external/')) or not p.is_file() or p.is_symlink():continue
 if any(x in p.parts for x in ['.cache','.deps','sources','__pycache__','vendor_harrl']):continue
 if p.suffix not in {'.py','.sh','.md'}:continue
 s=p.read_text();t=s
 if p.suffix=='.md':
  if old=='README.md':continue
  old_base=(R/old).parent
  def doclink(m):
   raw=m.group(1);url=raw.strip('<>'); fragment=''
   if '#' in url:url,fragment=url.split('#',1);fragment='#'+fragment
   if not url or re.match(r'[a-zA-Z]+:',url):return m.group(0)
   path=Path(os.path.normpath(str(old_base/url)))
   dest=R/mapped(str(path.relative_to(R))) if path.is_relative_to(R) else path
   target=os.path.relpath(dest,p.parent)+fragment
   if url.endswith('/'):target=target.rstrip('/')+'/'
   return m.group(0).replace(raw,'<'+target+'>' if ' ' in target else target)
  t=link_re.sub(doclink,t)
  t=absolute_strings(t)
  # Inline paths and command snippets use canonical project-relative names.
  for a,b in order:
   t=t.replace('`'+a, '`'+b).replace('Codex/'+a,'Codex/'+b)
  docs.append(new)
 else:
  t=absolute_strings(t)
  # Relative references to a different method become absolute, so joining them
  # cannot accidentally nest the new method inside the current one.
  prefixes={a:b for a,b in order if a.startswith(('revisions/','improvements/','external_benchmarks/')) and not a.endswith('.md')}
  for a,b in sorted(prefixes.items(),key=lambda x:-len(x[0])):
   t=re.sub(r'([\x27\x22])'+re.escape(a)+r'(?=/|[\x27\x22])',lambda m:m.group(1)+str(R/b),t)
  if '/wdro_rest_min/' in '/'+new:
   t=t.replace('WORKSPACE_ROOT = Path(__file__).resolve().parents[3]', 'WORKSPACE_ROOT = Path(__file__).resolve().parents[4]')
   t=re.sub(r'WORKSPACE_ROOT\s*/\s*([\x27\x22])Codex\1', 'Path(__file__).resolve().parents[2]',t)
   t=re.sub(r'PROJECT_ROOT\.parents\[1\](\s*/\s*[\x27\x22]DRO[\x27\x22])',r'PROJECT_ROOT.parents[2]\1',t)
  if old.startswith('improvements/') and not new.startswith('benchmarks/'):
   t=t.replace('ROOT.parents[1]','ROOT.parent').replace('root.parents[1]','root.parent')
  # Project-level report/status writers now address method docs.
  if '/wdro_rest_min/' in '/'+new:
   method=new.split('/')[0]
   names=[Path(b).name for a,b in order if b.startswith(method+'/docs/')]
  else:
   method=new.split('/')[0] if not new.startswith('benchmarks/') else '/'.join(new.split('/')[:2])
   names=[Path(b).name for a,b in order if b.startswith(method+'/docs/')]
  for name in set(names):
   if name=='README.md':continue
   # Only whole filename string literals, not data inside historical prose.
   t=re.sub(r'([\x27\x22])'+re.escape(name)+r'\1', lambda m:m.group(1)+'docs/'+name+m.group(1),t)
  code.append(new)
 if t!=s:
  before=sha(p);p.write_text(t)
  changed.append(dict(old=old,new=new,before_sha256=before,after_sha256=sha(p),snapshot=row['snapshot']))
json.dump(changed,(A/'rewritten_files.json').open('w'),ensure_ascii=False,indent=2)
json.dump(dict(code=code,docs=docs),(A/'maintained_files.json').open('w'),ensure_ascii=False,indent=2)
print('REWRITTEN',len(changed),'files; maintained',len(code),'code and',len(docs),'documents')
