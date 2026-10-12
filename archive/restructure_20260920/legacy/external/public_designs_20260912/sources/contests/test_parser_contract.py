"""Pin-coordinate preservation and corrupt-input checks for the contest parser."""
import itertools,json,pathlib,sys,tempfile
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent.parent/'parsers'))
from contests import parse_design

def main():
 tests=[]
 manifest=json.loads((BASE/'designs_manifest.json').read_text())
 row=next(x for x in manifest['designs'] if x['design_id']=='ispd2008_gr_adaptec1')
 plain=list(itertools.islice(parse_design(row['path']),100))
 compressed=list(itertools.islice(parse_design(row['archive_path']),100))
 for (i,a),(j,b) in zip(plain,compressed):
  assert i==j;np.testing.assert_array_equal(a,b)
 tests.append('first_100_original_and_gzip_inputs_identical')
 np.testing.assert_array_equal(plain[0][1],[[2095,7887],[2160,7842],[2095,7855],[2842,8584]])
 tests.append('independently_read_official_first_net_coordinates_preserved')
 header='grid 10 10 2\nvertical capacity 0 1\nhorizontal capacity 1 0\nminimum width 1 1\nminimum spacing 1 1\nvia spacing 1 1\n0 0 10 10\nnum net 1\n'
 with tempfile.TemporaryDirectory(prefix='parser_contract_',dir=BASE) as tmp:
  p=pathlib.Path(tmp)/'fixture.gr'
  p.write_text(header+'duplicate_layers 71 3 1\n1 2 1\n1 2 2\n4 5 1\n0\n')
  k,xy=next(parse_design(p));assert k=='duplicate_layers::71';np.testing.assert_array_equal(xy,[[1,2],[1,2],[4,5]])
  tests.append('multilayer_repeated_XY_preserves_original_three_pin_degree_and_order')
  for name,body in [('truncated','n 0 3 1\n1 2 1\n'),('invalid_layer','n 0 2 1\n1 2 3\n2 3 1\n'),('nonfinite','n 0 2 1\nnan 2 1\n2 3 1\n')]:
   p.write_text(header+body)
   try:list(parse_design(p))
   except ValueError:tests.append(name+'_rejected')
   else:raise AssertionError(name+' accepted')
 result={'status':'PASS','checks':tests,'full_read_coordinate_audit':str(BASE/'coordinate_audit.json'),'designs':manifest['count']}
 (BASE/'parser_contract_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
