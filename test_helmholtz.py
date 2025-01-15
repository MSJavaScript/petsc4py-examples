# -*- coding:utf-8 -*-
import math
import sys 
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

#python3 test_helmholtz.py -da_refine 4 -snes_rtol 1.0e-6 -snes_converged_reason
#python3 test_helmholtz.py 2 -snes_converged_reason -ph_no_gradient -snes_fd_function -snes_fd_color

# The quadrature points and weight of numerical integration on [-1, 1]
gausslegendre = {
    1: [(0.0, 2.0)],
    2: [(-0.577350269189626, 1.0), (0.577350269189626, 1.0)],
    3: [(-0.774596669241483, 0.555555555555556), (0.0, 0.888888888888889), 
        (0.774596669241483, 0.555555555555556)]
}
symmgauss = {
    1: [(1.0/3.0, 1.0/3.0, 0.5)],
    3: [(1.0/6.0, 1.0/6.0, 1.0/6.0), (2.0/3.0, 1.0/6.0, 1.0/6.0), (1.0/6.0, 2.0/3.0, 1.0/6.0)],
    4: [(1.0/3.0, 1.0/3.0, -27.0/96.0), (1.0/5.0, 1.0/5.0, 25.0/96.0), 
        (3.0/5.0, 1.0/5.0, 25.0/96.0), (1.0/5.0, 3.0/5.0, 25.0/96.0)],
}

def f_constant(x, y, p, eps):
    return 1.0
def u_exact_cosines(x, y, p, eps):
    return math.cos(math.pi * x) * math.cos(math.pi * y)
def f_cosine(x, y, p, eps):
    uu = math.cos(math.pi*x) * math.cos(math.pi * y)
    pi2 = math.pi ** 2
    lapu = - 2 * pi2 * uu
    if p == 2.0:
        return - lapu + uu
    else:
        ux = - math.pi * math.sin(math.pi * x)* math.cos(math.pi * y)
        uy = - math.pi * math.cos(math.pi * x)* math.sin(math.pi * y)
        w = ux * ux + uy * uy + eps * eps
        pi3 = pi2 * math.pi
        wx = pi3 * math.sin(2 * math.pi * x)* math.cos(2 * math.pi * y)
        wy = pi3 * math.cos(2 * math.pi * x)* math.sin(2 * math.pi * y)
        s = (p - 2) / 2
        return - s * math.pow(w,s-1) * (wx * ux + wy * uy)- math.pow(w,s) * lapu + uu



class gradRef:
    def __init__(self, xi, eta):
        self.xi = xi
        self.eta = eta


class Helmholtz:
    def __init__(self):
        self.da = None
        self.local_vec = None 
        self.quadpts = 2
        self.f = None # a user defined function
        self.p = 2
        self.eps = 0

    def FormObjectiveLocal(self, sens, Vec_au):
        def ObjIntegrandRef(ff, uu, xi, eta):
            du = deval(uu, xi, eta)
            u = eval(uu, xi, eta)
            return GradPow(hx, hy, du, self.p, 0.0)/self.p + 0.5*u*u - eval(ff,xi,eta)*u

        mx, my = self.da.getSizes()

        self.da.globalToLocal(Vec_au, self.local_vec)
        au = self.da.getVecArray(self.local_vec)
        
        hx = 1.0 / (mx - 1)
        hy = 1.0 / (my - 1)
        points_and_weight = gausslegendre[self.quadpts]
        lobj = 0.0
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            if j == 0:
                continue
            y = j * hy
            for i in range(xs, xe):
                if i == 0:
                    continue
                x = i * hx
                ff = [self.f(x, y, self.p, self.eps), 
                      self.f(x-hx, y, self.p, self.eps),
                      self.f(x-hx, y-hy, self.p, self.eps),
                      self.f(x, y-hy, self.p, self.eps)]
                uu = [au[i, j], au[i-1,j],  au[i-1,j-1], au[i,j-1]  ]
                for gp_r, gw_r in points_and_weight:
                    for gp_s, gw_s in points_and_weight:
                        lobj += gw_r * gw_s * ObjIntegrandRef(ff, uu, gp_r, gp_s)
        lobj *= hx * hy / 4.0
        lobj = self.da.comm.tompi4py().allreduce(lobj, MPI.SUM)
        return lobj



    def FormFunctionLocal(self, sens, Vec_au, Vec_FF):
        def IntegrandRef(L, ff, uu, xi, eta):
            du = deval(uu, xi, eta)
            dchiL = dchi(L, xi, eta)
            return GradPow(hx,hy,du, self.p - 2.0, self.eps) \
                * GradInnerProd(hx,hy,du,dchiL) \
                + (eval(uu,xi,eta) - eval(ff, xi,eta)) * chi(L,xi,eta)

        mx, my = self.da.getSizes()

        self.da.globalToLocal(Vec_au, self.local_vec)
        au = self.da.getVecArray(self.local_vec)

        FF = self.da.getVecArray(Vec_FF)
        hx = 1.0 / (mx - 1)
        hy = 1.0 / (my - 1)
        points_and_weight = gausslegendre[self.quadpts]
        li = [0, -1, -1, 0]
        lj = [0, 0, -1, -1]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                FF[i,j] = 0.0
        
        for j in range(ys, ye):
            if j == 0 or j > my - 1:
                continue
            y = j * hy
            for i in range(xs, xe):
                if i == 0 or i > mx - 1:
                    continue
                x = i * hx
                ff = [self.f(x, y, self.p, self.eps), 
                      self.f(x-hx, y, self.p, self.eps),
                      self.f(x-hx, y-hy, self.p, self.eps), 
                      self.f(x, y-hy, self.p, self.eps)]
                uu = [au[i,j], au[i-1,j], au[i-1,j-1], au[i,j-1]]
                for l in range(4):
                    PP = i + li[l]
                    QQ = j + lj[l]
                    if PP >= xs and PP < xe and QQ >= ys and QQ < ye:
                        for gp_r,gw_r in points_and_weight:
                            for gp_s,gw_s in points_and_weight:
                                FF[PP,QQ] += 0.25 * hx * hy *gw_r* gw_s*IntegrandRef(l, ff, uu, gp_r, gp_s)

# The points of reference element
# [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]
xiL = [1.0, -1.0, -1.0, 1.0]
etaL = [1.0, 1.0, -1.0, -1.0]
# The basis function on the reference element
def chi(L, xi, eta):
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta)

def eval(v, xi, eta):
    return  v[0] * chi(0,xi,eta) + v[1] * chi(1,xi,eta) \
           + v[2] * chi(2,xi,eta) + v[3] * chi(3,xi,eta)
# the gradient of chi
def dchi(L, xi, eta):
    return gradRef(0.25 * xiL[L]  * (1.0 + etaL[L] * eta), 0.25 * etaL[L] * (1.0 + xiL[L]  * xi))
# evaluate partial derivs of v(xi,eta) on reference element
def deval(v, xi, eta):
    s = gradRef(0, 0)
    for L in range(4):
        tmp = dchi(L, xi, eta)
        s.xi += v[L] * tmp.xi
        s.eta += v[L] * tmp.eta
    return s

def GradInnerProd(hx:float, hy:float, du:gradRef, dv:gradRef):
    cx = 4.0 / (hx * hx)
    cy = 4.0 / (hy * hy)
    return cx * du.xi * dv.xi + cy * du.eta * dv.eta

def GradPow(hx:float, hy:float, du:gradRef, P:float, eps:float):
    return math.pow(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0)


def GetVecFromFunction(da, w, func, helm):
    mx, my = da.getSizes()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    aw = da.getVecArray(w)
    (xs, xe), (ys, ye) = da.getRanges()
    for j in range(ys, ye):
        y = j * hy 
        for i in range(xs, xe):
            x = i * hx 
            aw[i,j] = func(x, y, helm.p, helm.eps)
    return 0


helm = Helmholtz()

problem      = "COSINES"
OptDB        = PETSc.Options("")
exact_init   = OptDB.getBool("-ph_exact_init", False)
no_objective = OptDB.getBool("-ph_no_objective", False)
no_gradient  = OptDB.getBool("-ph_no_gradient", False)
helm.eps     = OptDB.getReal("-ph_eps", 0.0)
helm.p       = OptDB.getReal("-ph_p", 2.0)
helm.quadpts = OptDB.getInt("-quadpts", 2)
view_f       = OptDB.getBool("-view_f", False)


if helm.p < 1.0:
    raise RuntimeError("p >= 1 required")
if helm.p == 1.0:
    PETSc.Sys.Print("WARNING: well-posedness only known for p > 1\n")
if helm.quadpts < 1 or helm.quadpts > 3:
    raise RuntimeError("quadrature points n=1,2,3 only")

da = PETSc.DMDA()
da.create(comm = PETSc.COMM_WORLD,
        dim=2, 
        sizes=(2, 2), 
        proc_sizes=None, 
        boundary_type=(PETSc.DM.BoundaryType.NONE, PETSc.DM.BoundaryType.NONE),
        stencil_type=PETSc.DMDA.StencilType.BOX,
        stencil_width=1,
        dof=1,
        setup=False)
da.setFromOptions()
da.setUp()
da.setUniformCoordinates(0.0, 1.0, 0.0, 1.0)
helm.da = da
helm.local_vec = da.createLocalVec()

snes = PETSc.SNES().create(comm=PETSc.COMM_WORLD)
snes.setDM(da)
if not no_objective:
    snes.setObjective(helm.FormObjectiveLocal)
if no_gradient:
    OptDB.setValue("-snes_fd_function_eps", 0.0)
else:
    snes.setFunction(helm.FormFunctionLocal)
snes.setFromOptions()

u_initial = da.createGlobalVec()
u_initial.set(0.5)
if problem == "COSINES":
    if exact_init:
        GetVecFromFunction(da, u_initial, u_exact_cosines, helm)
    helm.f = f_cosine
elif problem == "CONSTANT":
    if exact_init:
        u_initial.set(1.0)
    helm.f = f_constant
else:
    raise RuntimeError("unknown problem type\n")

if view_f:
    vf = u_initial.copy()
    if problem == "CONSTANT":
        vf.set(1.0)
    elif problem == "COSINES":
        GetVecFromFunction(da, vf, f_cosine, helm)
    vf.view()
    vf.destroy()

snes.solve(None, u_initial)
u = snes.getSolution()
#u.view()

u_exact = u.copy()
if problem == "CONSTANT":
    u_exact.set(1.0)
elif problem == "COSINES":
    GetVecFromFunction(da, u_exact, u_exact_cosines, helm)
u.axpy(-1.0, u_exact)
err = u.norm(PETSc.NormType.NORM_INFINITY)
mx, my = da.getSizes()

PETSc.Sys.Print("done on %d x %d grid with p=%.3f ...\n  numerical error:  |u-u_exact|_inf = %.3e\n"%(mx, my, helm.p, err))

u_exact.destroy()
u_initial.destroy()
da.destroy()
snes.destroy()