"""Independent MST, analytic optimum, evaluator and solver-exclusion checks."""
import ctypes
import json
import subprocess
import sys
import unittest
import shutil
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import geometry as G

class GeometryTests(unittest.TestCase):
    def test_mst_and_analytic_three_pin_bounds(self):
        rng=np.random.default_rng(817)
        x=rng.random((100,3,2))
        lb,m,h=G.bounds(x)
        # For any three pins, a median-coordinate Steiner point attains HPWL.
        opt=np.abs(x-np.median(x,axis=1)[:,None]).sum((1,2))
        np.testing.assert_allclose(opt,h,atol=1e-14)
        self.assertTrue(np.all(lb<=opt+1e-12))
        self.assertTrue(np.all(opt<=m+1e-12))
        for p,val in zip(x,m):
            distance=np.abs(p[:,None]-p[None]).sum(-1)
            self.assertAlmostEqual(val,minimum_spanning_tree(distance).sum(),places=12)

    def test_degree30_mst_and_invariance(self):
        x=G.make_data(50,30,'uniform','unit')
        lb,m,h=G.bounds(x)
        for p,val in zip(x,m):
            p=p.astype(np.float64)
            self.assertAlmostEqual(val,minimum_spanning_tree(np.abs(p[:,None]-p[None]).sum(-1)).sum(),places=11)
        for a,b in zip(G.bounds(x[:,::-1]),(lb,m,h)): np.testing.assert_allclose(a,b)
        for a,b in zip(G.bounds(x.astype(float)*3+2),(lb,m,h)): np.testing.assert_allclose(a,b*3)
        np.testing.assert_allclose(G.costs(x,x[:,::-1]),0,atol=1e-14)

    def test_independent_interval_union(self):
        # Independently build all physical route intervals then union each line.
        # Distinct x/y coordinates make union exactly equal REST span accounting.
        rng=np.random.default_rng(901)
        x=rng.random((100,30,2))
        t=[]
        for _ in x:
            row=[]
            for child in range(1,30):
                pair=[child,int(rng.integers(child))]
                row+=pair if rng.random()<.5 else pair[::-1]
            t.append(row)
        t=np.asarray(t,dtype=np.int32)
        def union_length(p,a):
            groups={}
            for u,v in a.reshape(-1,2):
                groups.setdefault(('v',p[u,0]),[]).append(sorted([p[u,1],p[v,1]]))
                groups.setdefault(('h',p[v,1]),[]).append(sorted([p[u,0],p[v,0]]))
            total=0.
            for intervals in groups.values():
                merged=[]
                for lo,hi in sorted(intervals):
                    if not merged or lo>merged[-1][1]: merged.append([lo,hi])
                    else: merged[-1][1]=max(merged[-1][1],hi)
                total+=sum(hi-lo for lo,hi in merged)
            return total
        reference=np.array([union_length(p,a) for p,a in zip(x,t)])
        np.testing.assert_allclose(G.lengths(x,t),reference,rtol=0,atol=1e-13)
        self.assertTrue(np.all(G.lengths(x,t)>=G.bounds(x)[0]-1e-12))

    def test_reject_disconnected(self):
        x=np.array([[[0.,0.],[1.,0.],[1.,1.]]])
        with self.assertRaises(ValueError): G.lengths(x,np.array([[0,1,1,0]]))

    def test_collinear_span_accounting(self):
        x=np.array([[[0.,0.],[0.,1.],[0.,2.]]])
        # Historical REST per-terminal spans double count an overlapping interval.
        np.testing.assert_array_equal(G.lengths(x,np.array([[0,2,1,2]])),[3.])
        lb,m,h=G.bounds(x)
        np.testing.assert_array_equal(lb,[2.]); np.testing.assert_array_equal(m,[2.])
        np.testing.assert_array_equal(h,[2.])

    def test_data_disjointness_and_legalization(self):
        x=G.make_data(32,30,'mix_gauss','train',101)
        y=G.make_data(32,30,'mix_gauss','validation',0)
        self.assertFalse(set(map(bytes,x)).intersection(map(bytes,y)))
        d=np.random.default_rng(7).normal(0,.2,(32,8,30,2))
        z=G.legalize(x,d)
        self.assertTrue(np.isfinite(z).all())
        for p in z.reshape(-1,30,2): self.assertFalse(G.has_spacing_violation(p))
        np.testing.assert_allclose(G.legalize(x,np.zeros_like(x)),x,atol=1e-7)

    def test_guard_blocks_import_binary_and_old_weights(self):
        code='''import geometry as g\ng.install_oracle_guard()\nimport ctypes\ntry: ctypes.CDLL("libeval.so")\nexcept RuntimeError: pass\nelse: raise AssertionError("library guard absent")\ntry: __import__("src.geosteiner")\nexcept RuntimeError: pass\nelse: raise AssertionError("import guard absent")\ntry: open("/tmp/old_weights.pt","rb")\nexcept RuntimeError: pass\nelse: raise AssertionError("weights guard absent")\nprint("GUARD_PASS")'''
        result=subprocess.run([sys.executable,'-c',code],cwd=G.ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('GUARD_PASS',result.stdout)

if __name__=='__main__': unittest.main(verbosity=2)
