"""Own-action reward, direct transport, and RL-gradient checks without reference solvers."""
import subprocess
import sys
import unittest
import numpy as np
import torch
import geometry as G

class EnvironmentTests(unittest.TestCase):
    def test_wirelength_independent_interval_union(self):
        rng=np.random.default_rng(902); x=rng.random((50,30,2)); t=[]
        for p in x:
            row=[]
            for child in range(1,30):
                pair=[child,int(rng.integers(child))]; row+=pair if rng.random()<.5 else pair[::-1]
            t.append(row)
        t=np.asarray(t,np.int32)
        def physical_union(p,edges):
            groups={}
            for u,v in edges.reshape(-1,2):
                groups.setdefault(('h',p[v,1]),[]).append(sorted((p[u,0],p[v,0])))
                groups.setdefault(('v',p[u,0]),[]).append(sorted((p[u,1],p[v,1])))
            total=0.
            for intervals in groups.values():
                merged=[]
                for lo,hi in sorted(intervals):
                    if not merged or lo>merged[-1][1]: merged.append([lo,hi])
                    else: merged[-1][1]=max(merged[-1][1],hi)
                total+=sum(hi-lo for lo,hi in merged)
            return total
        np.testing.assert_allclose(G.lengths(x,t),[physical_union(p,a) for p,a in zip(x,t)],rtol=0,atol=1e-13)

    def test_span_accounting_and_invalid_tree(self):
        x=np.array([[[0.,0.],[0.,1.],[0.,2.]]])
        np.testing.assert_array_equal(G.lengths(x,np.array([[0,2,1,2]])),[3.])
        with self.assertRaises(ValueError): G.lengths(x,np.array([[0,1,1,0]]))

    def test_direct_transport_and_labeled_boundary(self):
        x=np.array([[[.2,.2],[.5,.3],[.8,.8]]])
        np.testing.assert_allclose(G.costs(x,x+.01),.02)
        self.assertGreater(float(G.costs(x,x[:,::-1])[0]),0)
        np.testing.assert_allclose(G.costs(x[:,::-1],(x+.01)[:,::-1]),.02)

    def test_no_bbox_rescale(self):
        x=np.array([[[.3,.3],[.4,.4],[.5,.5]]],dtype=np.float32)
        np.testing.assert_array_equal(G.legalize(x,np.zeros_like(x)),x)
        np.testing.assert_allclose(G.legalize(x,np.full_like(x,.01)),x+.01)

    def test_data_and_legality(self):
        for distribution in G.PARAMETERS:
            a=G.make_data(24,30,distribution,'train',101); b=G.make_data(24,30,distribution,'test',0)
            self.assertFalse(set(map(bytes,a)).intersection(map(bytes,b)))
            self.assertTrue(np.all(a>=.1) and np.all(a<=.9))
            z=G.legalize(a,np.random.default_rng(2).normal(0,.08,a.shape))
            self.assertFalse(any(G.has_spacing_violation(p) for p in z))

    def test_score_function_baseline_identity(self):
        # Enumerate a two-action policy exactly. No known optimum enters estimator.
        for baseline in (-10.,0.,2.,4.,12.):
            theta=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
            prob=torch.sigmoid(theta); p=torch.stack((1-prob,prob)); cost=torch.tensor([2.,4.],dtype=torch.float64)
            objective=(p.detach()*(cost-baseline)*p.log()).sum()
            grad,=torch.autograd.grad(objective,theta)
            torch.testing.assert_close(grad,2*prob.detach()*(1-prob.detach()))

    def test_no_reference_solver_import(self):
        code='''import geometry as g\ng.install_oracle_guard()\nfor name in ["src.geosteiner","scipy.optimize","scipy.sparse.csgraph","networkx"]:\n try: __import__(name)\n except RuntimeError: pass\n else: raise AssertionError(name)\nprint("PASS")'''
        r=subprocess.run([sys.executable,'-c',code],cwd=G.ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__': unittest.main(verbosity=2)
