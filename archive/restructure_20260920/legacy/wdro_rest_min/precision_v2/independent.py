"""Independent HiGHS integer models for checking GeoSteiner reference lengths."""
from __future__ import annotations
import ctypes as C
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix,vstack
from .geosteiner import GeoSteinerEvaluator, SCALE

def fst_milp(pins, time_limit=60):
    """Solve all generated full components with an independent MILP backend.

    A connected hyperedge selection with sum(|edge|-1)=n-1 is a hypertree.
    Separated subtour inequalities enforce an acyclic connected selection.
    A connected optimum proves optimality over the FST universe.
    The FST generator is shared; its geometry is also checked on small Hanan
    instances by the separate model below.
    """
    pts=np.ascontiguousarray(np.asarray(pins,dtype=np.float64)*SCALE)
    assert np.array_equal(pts,np.rint(pts))
    n=len(pts);lib=GeoSteinerEvaluator()._library;ip=C.POINTER(C.c_int);dp=C.POINTER(C.c_double)
    lib.gst_generate_rfsts.argtypes=[C.c_int,dp,C.c_void_p,ip];lib.gst_generate_rfsts.restype=C.c_void_p
    lib.gst_get_hg_edges.argtypes=[C.c_void_p,ip,ip,ip,dp];lib.gst_free_hg.argtypes=[C.c_void_p]
    status=C.c_int();hg=lib.gst_generate_rfsts(n,pts.ctypes.data_as(dp),None,C.byref(status))
    assert status.value==0 and hg
    try:
        ne=C.c_int();assert lib.gst_get_hg_edges(hg,C.byref(ne),None,None,None)==0
        sizes=np.empty(ne.value,dtype=np.int32);costs=np.empty(ne.value,dtype=np.float64)
        assert lib.gst_get_hg_edges(hg,None,sizes.ctypes.data_as(ip),None,costs.ctypes.data_as(dp))==0
        terminals=np.empty(int(sizes.sum()),dtype=np.int32)
        assert lib.gst_get_hg_edges(hg,None,None,terminals.ctypes.data_as(ip),None)==0
        edges=np.split(terminals,np.cumsum(sizes)[:-1]);assert np.array_equal(costs,np.rint(costs))
    finally:
        lib.gst_free_hg(hg)
    m=len(edges);sets=[set(e.tolist()) for e in edges]
    rows=[csc_matrix((sizes-1).reshape(1,-1))];lower=[n-1];upper=[n-1]
    for v in range(n):
        rows.append(csc_matrix(np.array([v in e for e in sets],dtype=float).reshape(1,-1)));lower.append(1);upper.append(np.inf)
    for iteration in range(500):
        result=milp(costs,integrality=np.ones(m),bounds=Bounds(0,1),
                    constraints=LinearConstraint(vstack(rows,format='csc'),lower,upper),
                    options=dict(time_limit=time_limit,mip_rel_gap=0.0))
        if not result.success:raise RuntimeError(f'Independent FST MILP failed: {result.message}')
        selected=np.flatnonzero(result.x>.5);parent=list(range(n))
        def root(v):
            while parent[v]!=v:v=parent[v]
            return v
        for i in selected:
            for v in edges[i][1:]:parent[root(int(v))]=root(int(edges[i][0]))
        comps=[{v for v in range(n) if root(v)==r} for r in sorted({root(v) for v in range(n)})]
        if len(comps)==1:
            assert sum(sizes[selected]-1)==n-1
            integer_cost=int(sum(int(costs[i]) for i in selected))
            assert np.max(np.abs(result.x-np.rint(result.x)))<1e-6 and integer_cost-result.mip_dual_bound<0.25
            return dict(length=integer_cost/SCALE,integer_objective=integer_cost,integer_lower_bound=float(result.mip_dual_bound),
                        fst_count=m,iterations=iteration+1,selected=selected.tolist(),solver='FST subtour/scipy.milp/HiGHS',
                        mip_gap=float(result.mip_gap),shared_component_generator=True)
        added=0
        for comp in comps:
            coeff=np.array([max(0,len(e&comp)-1) for e in sets],dtype=float)
            if coeff[selected].sum()>len(comp)-1:
                rows.append(csc_matrix(coeff.reshape(1,-1)));lower.append(-np.inf);upper.append(len(comp)-1);added+=1
        assert added>0,'disconnected rank-n-1 selection must violate a subtour inequality'
    raise RuntimeError('independent subtour separation did not finish')

def hanan_milp(pins,time_limit=60):
    """Independent Hanan-grid undirected edge + directed single-commodity flow."""
    xy=np.asarray(pins,dtype=np.float64)*SCALE
    assert np.array_equal(xy,np.rint(xy))
    xy=xy.astype(np.int64);xs=np.unique(xy[:,0]);ys=np.unique(xy[:,1]);nx,ny=len(xs),len(ys)
    edges=[];weights=[]
    for i in range(nx):
        for j in range(ny):
            u=i*ny+j
            if i+1<nx:edges.append((u,u+ny));weights.append(int(xs[i+1]-xs[i]))
            if j+1<ny:edges.append((u,u+1));weights.append(int(ys[j+1]-ys[j]))
    m=len(edges);nv=nx*ny;nt=len(xy)
    terminals=[int(np.searchsorted(xs,x)*ny+np.searchsorted(ys,y)) for x,y in xy]
    rr=[];cc=[];vv=[];lb=[];ub=[]
    for v in range(nv):
        for e,(a,b) in enumerate(edges):
            if v in (a,b):
                sign=1 if v==a else -1
                rr.extend([v,v]);cc.extend([m+e,m+m+e]);vv.extend([sign,-sign])
        demand=nt-1 if v==terminals[0] else (-1 if v in terminals[1:] else 0)
        lb.append(demand);ub.append(demand)
    for e in range(m):
        row=nv+e;rr.extend([row,row,row]);cc.extend([e,m+e,m+m+e]);vv.extend([-(nt-1),1,1]);lb.append(-np.inf);ub.append(0)
    A=csc_matrix((vv,(rr,cc)),shape=(nv+m,3*m))
    result=milp(np.r_[weights,np.zeros(2*m)],integrality=np.r_[np.ones(m),np.zeros(2*m)],
        bounds=Bounds(np.zeros(3*m),np.r_[np.ones(m),np.full(2*m,nt-1)]),constraints=LinearConstraint(A,lb,ub),
        options=dict(time_limit=time_limit,mip_rel_gap=0.0))
    if not result.success:raise RuntimeError(f'Hanan MILP failed: {result.message}')
    integer_cost=sum(w for w,x in zip(weights,result.x[:m]) if x>.5)
    assert integer_cost-result.mip_dual_bound<0.25
    return dict(length=integer_cost/SCALE,integer_objective=integer_cost,integer_lower_bound=float(result.mip_dual_bound),
                solver='independent Hanan grid/scipy.milp/HiGHS',mip_gap=float(result.mip_gap),shared_component_generator=False)
