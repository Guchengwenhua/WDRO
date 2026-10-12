"""Independent, read-only CircuitOps pin XY reconstruction for evaluated designs.

Reads final DEF placement and original Nangate45 LEF RECT geometry directly.
Never imports the production parser or mutates source/evaluation data. The
1 DBU tolerance is inherited from the original GCD audit, not an EDA standard.
Only this script's JSON/log artifacts are written, under notes/.
"""
from __future__ import annotations
import collections
import csv
import datetime
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'sources/real_designs'
REPO=SOURCE/'CircuitOps'
MANIFEST=SOURCE/'design_manifest.json'
LEF=REPO/'platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef'
DESIGNS=('aes','ibex','jpeg')
TOLERANCE_DBU=Fraction(1)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def artifact(path):
    path=Path(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}


def text_file(path):
    return gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else Path(path).open(newline='')


def canonical(name):
    return re.sub(r'\\(.)',r'\1',name)


def read_lef(path):
    """Read complete macro/pin scopes; preserve every RECT, not just a bbox."""
    macros={}
    macro=pin=None
    with Path(path).open() as f:
        for lineno,line in enumerate(f,1):
            tokens=line.strip().split()
            if not tokens:continue
            if tokens[0]=='MACRO':
                if macro is not None:raise ValueError(f'Nested LEF macro at {lineno}')
                macro=tokens[1]
                if macro in macros:raise ValueError(f'Duplicate LEF macro {macro}')
                macros[macro]={'pins':{},'origin':None,'size':None}
            elif macro is None:
                continue
            elif tokens[0]=='PIN':
                if pin is not None:raise ValueError(f'Nested LEF pin at {lineno}')
                pin=tokens[1]
                if pin in macros[macro]['pins']:raise ValueError(f'Duplicate LEF pin at {lineno}')
                macros[macro]['pins'][pin]=[]
            elif tokens[0]=='ORIGIN':
                macros[macro]['origin']=tuple(Fraction(x) for x in tokens[1:3])
            elif tokens[0]=='SIZE':
                macros[macro]['size']=(Fraction(tokens[1]),Fraction(tokens[3]))
            elif pin is not None and tokens[0]=='RECT':
                coords=tokens[1:-1]
                if len(coords)!=4:raise ValueError(f'Unsupported LEF RECT syntax at {lineno}')
                xy=tuple(Fraction(x) for x in coords)
                if xy[2]<xy[0] or xy[3]<xy[1]:raise ValueError(f'Invalid LEF rectangle at {lineno}')
                macros[macro]['pins'][pin].append(xy)
            elif pin is not None and tokens[0] in ('POLYGON','VIA','PATH'):
                raise ValueError(f'Unsupported LEF geometry at {lineno}: {tokens[0]}')
            elif tokens[0]=='END' and len(tokens)==2:
                if pin is not None:
                    if tokens[1]!=pin:raise ValueError(f'Wrong LEF pin END at {lineno}')
                    pin=None
                elif tokens[1]==macro:
                    macro=None
                else:
                    raise ValueError(f'Unexpected LEF named END at {lineno}')
    if macro is not None or pin is not None:raise ValueError('Unclosed LEF scopes')
    for name,m in macros.items():
        if m['size'] is None or m['origin']!=(0,0):
            raise ValueError(f'Unvalidated macro origin/size for {name}')
    return macros


def read_components(path):
    """Independently parse physical instances, stopping before routing sections."""
    units=declared=None
    active=False
    pending=[]
    instances={}
    with text_file(path) as f:
        for lineno,line in enumerate(f,1):
            s=line.strip()
            if not active:
                match=re.fullmatch(r'UNITS DISTANCE MICRONS (\d+)\s*;',s)
                if match:units=int(match[1])
                match=re.fullmatch(r'COMPONENTS (\d+)\s*;',s)
                if match:declared=int(match[1]);active=True
                continue
            if s=='END COMPONENTS':
                if pending:raise ValueError('Unterminated DEF component')
                break
            if not s:continue
            pending.append(s)
            if ';' not in s:continue
            record=' '.join(pending);pending=[]
            name=re.match(r'-\s+(\S+)\s+(\S+)\b',record)
            placement=re.search(r'\+\s+(PLACED|FIXED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)',record)
            if not name:raise ValueError(f'Unparsed component at {lineno}')
            instance,master=map(canonical,name.groups())
            if instance in instances:raise ValueError(f'Duplicate component {instance}')
            if placement:
                _,x,y,orientation=placement.groups()
                instances[instance]=(master,int(x),int(y),orientation)
            else:
                instances[instance]=None
        else:
            raise ValueError('Missing END COMPONENTS')
    if not units or len(instances)!=declared:raise ValueError('DEF units/component coverage mismatch')
    return units,instances


def pin_center_db(macros,master,pin,units):
    m=macros[master]
    rectangles=m['pins'][pin]
    if not rectangles:raise ValueError(f'No RECTs for {master}/{pin}')
    # The downloaded source geometry is exactly representable on DEF's DBU grid.
    scaled=[tuple(v*units for v in rect) for rect in rectangles]
    if any(v.denominator!=1 for rect in scaled for v in rect):
        raise ValueError(f'Noninteger LEF-to-DEF corner in {master}/{pin}')
    x=sum((r[0]+r[2])/2 for r in scaled)/len(scaled)
    y=sum((r[1]+r[3])/2 for r in scaled)/len(scaled)
    width,height=(v*units for v in m['size'])
    return x,y,width,height,len(scaled)


def orient_center(center,orientation):
    x,y,w,h,_=center
    # These are the four orientations encountered in this audited source set.
    # Unsupported rotations fail explicitly instead of being silently assumed N.
    if orientation=='N':return x,y
    if orientation=='S':return w-x,h-y
    if orientation=='FN':return w-x,y
    if orientation=='FS':return x,h-y
    raise ValueError(f'Unsupported orientation {orientation}')


def selected_nets(design_id):
    mpath=BASE/'cache/designs'/design_id/'manifest.json'
    manifest=json.loads(mpath.read_text())
    selected={}
    artifacts=[artifact(mpath)]
    for group in manifest['groups']:
        if not group['valid_count']:continue
        apath=group['input']['logical_path']
        item=artifact(apath)
        if item['sha256']!=group['input']['sha256']:raise ValueError('Changed extraction cache')
        artifacts.append(item)
        with np.load(apath,allow_pickle=False) as data:
            if len(data['net_ids'])!=group['valid_count']:raise ValueError('Selected-net count mismatch')
            prefix=f'public:{design_id}:'
            suffix=f':d{group["degree"]}'
            for identity in data['net_ids']:
                name=identity.decode()
                if not(name.startswith(prefix) and name.endswith(suffix)):raise ValueError('Bad source net identity')
                name=name[len(prefix):-len(suffix)]
                if name in selected:raise ValueError('Selected net duplicated across degree caches')
                selected[name]=group['degree']
    if not selected:raise ValueError(f'No evaluated nets in requested design {design_id}')
    return selected,artifacts


def audit_design(row,macros):
    started=time.monotonic()
    units,instances=read_components(row['def_path'])
    selected,cache_artifacts=selected_nets(row['design_id'])
    centers={}
    checked=set()
    orientations=collections.Counter()
    selected_orientations=collections.Counter()
    selected_pin_counts=collections.Counter()
    rect_histogram=collections.Counter()
    selected_rect_histogram=collections.Counter()
    failures=[]
    mismatch_examples=[]
    mismatches=selected_mismatches=0
    maximum=selected_maximum=Fraction(0)
    n=selected_pins=0
    max_pin=None
    with text_file(row['pin_properties_path']) as f:
        for data in csv.DictReader(f):
            n+=1
            name=canonical(data['pin_name'])
            net=canonical(data['net_name'])
            instance=canonical(data['cell_name'])
            is_selected=net in selected
            if is_selected:selected_pins+=1;selected_pin_counts[net]+=1
            try:
                if name in checked:raise ValueError('duplicate IR pin')
                checked.add(name)
                if not name.startswith(instance+'/'):raise ValueError('pin/cell name inconsistency')
                pin=name[len(instance)+1:]
                location=instances[instance]
                if location is None:raise ValueError('instance has no DEF placement')
                master,tx,ty,orientation=location
                key=(master,pin)
                if key not in centers:centers[key]=pin_center_db(macros,master,pin,units)
                center=centers[key]
                x,y=orient_center(center,orientation)
                x+=tx;y+=ty
                upstream_x,upstream_y=Fraction(data['x']),Fraction(data['y'])
                if upstream_x.denominator!=1 or upstream_y.denominator!=1:raise ValueError('noninteger upstream DBU coordinate')
                error=max(abs(x-upstream_x),abs(y-upstream_y))
                orientations[orientation]+=1
                rect_histogram[center[4]]+=1
                if error>maximum:maximum=error;max_pin=name
                if is_selected:
                    selected_orientations[orientation]+=1
                    selected_rect_histogram[center[4]]+=1
                    selected_maximum=max(selected_maximum,error)
                if error>TOLERANCE_DBU:
                    mismatches+=1
                    selected_mismatches+=int(is_selected)
                    if len(mismatch_examples)<20:
                        example=dict(pin=name,net=net,selected=is_selected,master=master,orientation=orientation,upstream_xy_dbu=[float(upstream_x),float(upstream_y)],reconstructed_xy_dbu=[float(x),float(y)],max_abs_error_dbu=float(error))
                        mismatch_examples.append(example)
                        print(json.dumps({'coordinate_mismatch':row['design_id'],**example}),flush=True)
            except (KeyError,ValueError) as e:
                failures.append(dict(pin=name,net=net,selected=is_selected,error=str(e)))
                if len(failures)<=20:print(json.dumps({'reconstruction_failure':row['design_id'],**failures[-1]}),flush=True)
    degree_coverage=collections.Counter()
    coverage_errors=[]
    for net,degree in selected.items():
        degree_coverage[degree]+=1
        if selected_pin_counts[net]!=degree:coverage_errors.append(dict(net=net,expected_pins=degree,observed_pins=selected_pin_counts[net]))
    ok=n>0 and not failures and not mismatches and not coverage_errors and selected_pins==sum(selected.values())
    result=dict(design_id=row['design_id'],status='PASS' if ok else 'FAIL',dbu_per_micron=units,absolute_tolerance_dbu=1,absolute_tolerance_micron=1/units,
        all_ir_pin_count=n,all_reconstructed_pin_count=sum(orientations.values()),orientation_pin_counts=dict(orientations),pin_rectangle_count_histogram=dict(rect_histogram),
        maximum_abs_coordinate_error_dbu=float(maximum),maximum_error_pin=max_pin,mismatches_gt_1_dbu=mismatches,reconstruction_failure_count=len(failures),reconstruction_failure_examples=failures[:20],mismatch_examples=mismatch_examples,
        selected_evaluated_net_count=len(selected),selected_evaluated_degree_counts=dict(sorted(degree_coverage.items())),selected_expected_pin_count=sum(selected.values()),selected_observed_pin_count=selected_pins,
        selected_orientation_pin_counts=dict(selected_orientations),selected_pin_rectangle_count_histogram=dict(selected_rect_histogram),selected_maximum_abs_coordinate_error_dbu=float(selected_maximum),selected_mismatches_gt_1_dbu=selected_mismatches,
        selected_coverage_errors=coverage_errors,inputs=[artifact(row['def_path']),artifact(row['pin_properties_path']),*cache_artifacts],elapsed_seconds=time.monotonic()-started)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','mismatch_examples','reconstruction_failure_examples')}),flush=True)
    return result


def main():
    began=stamp()
    rows=json.loads(MANIFEST.read_text())['designs']
    chosen=[next(r for r in rows if r['technology']=='nangate45' and r['design_name']==name) for name in DESIGNS]
    macros=read_lef(LEF)
    results=[audit_design(row,macros) for row in chosen]
    passed=all(r['status']=='PASS' for r in results) and len(results)==len(DESIGNS)
    output=dict(status='PASS' if passed else 'FAIL',started_utc=began,finished_utc=stamp(),designs=results,
        totals=dict(designs=len(results),all_ir_pins=sum(r['all_ir_pin_count'] for r in results),selected_evaluated_nets=sum(r['selected_evaluated_net_count'] for r in results),selected_evaluated_pins=sum(r['selected_observed_pin_count'] for r in results),mismatches_gt_1_dbu=sum(r['mismatches_gt_1_dbu'] for r in results),reconstruction_failures=sum(r['reconstruction_failure_count'] for r in results)),
        method='Independent DEF instance master/orientation/placement plus arithmetic mean of all LEF pin RECT centers, using exact Fraction arithmetic in DEF DBU. No production parser imports; every IR pin in AES, ibex and jpeg checked; all accepted target-degree pin coverage reconciled to immutable extraction arrays.',
        tolerance_provenance=dict(value_dbu=1,inherited_from=str(SOURCE/'parser_validation/def_lef_coordinate_crosscheck.json'),explanation='Same absolute per-axis threshold as the existing GCD audit. Source Tcl helpers calculate each integer DBU rectangle midpoint then an integer average; Python helper uses OpenDB ITerm.getAvgXY. Threshold accommodates upstream integer center rounding. It is an explicit local audit tolerance, not an industry standard or a claim about arbitrary LEF geometries.',source_lines={'scripts/tcl/helpers.tcl':'411-435','scripts/python/openroad_helpers.py':'669-675'}),
        scope_limit='Nangate45 AES, ibex, jpeg physical-coordinate convention only; no new claims of tapeout or validation of all CircuitOps technologies. Layer/obstacle/routing rules are outside the 2D REST task.',
        sources=[artifact(MANIFEST),artifact(LEF),artifact(REPO/'scripts/tcl/helpers.tcl'),artifact(REPO/'scripts/python/openroad_helpers.py'),artifact(SOURCE/'parser_validation/def_lef_coordinate_crosscheck.json')],implementation=artifact(__file__),mutation_scope='Only notes/selected_pin_coordinate_audit.json and its log; no source, cache, model, evaluation, or GPU mutations.')
    path=BASE/'notes/selected_pin_coordinate_audit.json'
    partial=path.with_suffix('.json.partial')
    partial.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    os.replace(partial,path)
    print(json.dumps({'status':output['status'],'totals':output['totals'],'output':str(path)}),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
