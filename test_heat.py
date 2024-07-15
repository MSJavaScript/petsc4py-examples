# -*- coding:utf-8 -*-
import math
import sys 
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

class Heat2D:
    def __init__(self, da, D0):
        self.da = da
        self.D0 = D0
        self.rank = 0
        self.local_vec_u = da.createLocalVec()
        self.viewer = PETSc.Viewer()
    #Callable[[TS, float, Vec, Vec], None]
    def FormRHSFunctionLocal(self, ts, t, vec_u, vec_g):
        self.da.globalToLocal(vec_u, self.local_vec_u)
        au = self.da.getVecArray(self.local_vec_u)
        aG = self.da.getVecArray(vec_g)
        mx, my = self.da.getSizes()
        hx = 1.0 / (mx - 1)
        hy = 1.0 / my 
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            y = hy * j 
            for i in range(xs, xe):
                x = hx * i
                ul = au[i+1, j] + 2.0*hx*gamma_neumann(y) if i==0 else au[i-1, j]
                ur = au[i-1, j] if i == mx-1 else au[i+1, j]
                uxx = (ul - 2.0*au[i, j] + ur)/(hx*hx)
                uyy = (au[i, j-1] - 2.0*au[i, j] + au[i, j+1])/(hy*hy)
                aG[i, j] = self.D0*(uxx + uyy) + f_source(x,y)
        
    # Callable[[TS, float, Vec, Mat, Mat], None]
    def FormRHSJacobianLocal(self, ts, t, vec_u, Mat_J, Mat_P):
        row = PETSc.Mat.Stencil()
        col = [PETSc.Mat.Stencil() for i in range(5)]
        v = [0]*5
        mx, my = self.da.getSizes()
        hx = 1.0 / (mx - 1)
        hy = 1.0 / my 
        hx2 = hx * hx 
        hy2 = hy * hy
        D = self.D0
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            row.j = j
            col[0].j = j
            for i in range(xs, xe):
                row.i = i
                col[0].i = i
                v[0] = - 2.0 * D * (1.0 / hx2 + 1.0 / hy2)
                col[1].j = j-1;  col[1].i = i;    v[1] = D / hy2
                col[2].j = j+1;  col[2].i = i;    v[2] = D / hy2
                col[3].j = j;    col[3].i = i-1;  v[3] = D / hx2
                col[4].j = j;    col[4].i = i+1;  v[4] = D / hx2
                ncols = 5
                if i == 0:
                    ncols = 4
                    col[3].j = j;  col[3].i = i+1;  v[3] = 2.0 * D / hx2
                elif i == mx-1:
                    ncols = 4
                    col[3].j = j;  col[3].i = i-1;  v[3] = 2.0 * D / hx2
                
                for ic in range(ncols):
                    Mat_P.setValueStencil(row, col[ic], v[ic], PETSc.InsertMode.INSERT_VALUES)
        Mat_P.assemble()
        if Mat_J != Mat_P:
            Mat_J.assemble()
        return True
                
    #alias of Callable[[TS, int, float, Vec], None]
    def EnergyMonitor(self, ts, step, time, u):
        self.da.globalToLocal(u, self.local_vec_u)
        au = self.da.getVecArray(self.local_vec_u)
        mx, my = self.da.getSizes()
        lenergy = 0
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                if (i == 0 or i == mx-1):
                    lenergy += 0.5*au[i, j]
                else:
                    lenergy += au[i, j]
        hx = 1.0 / (mx-1)
        hy = 1.0 / my 
        lenergy *= (hx*hy)
        energy = da.comm.tompi4py().allreduce(lenergy, MPI.SUM)
        dt = ts.getTimeStep()
        PETSc.Sys.Print("  energy = %9.2e     nu = %8.4f"%(energy, self.D0*dt/(hx*hy)))

def f_source(x, y):
    return 3.0 * math.exp(-25.0*(x-0.6)*(x-0.6)) * math.sin(2.0*math.pi*y)
def gamma_neumann(y):
    return math.sin(6.0*math.pi*y)


OptDB = PETSc.Options("ht_")
D0      = OptDB.getReal("D0", 1.0)
monitor = OptDB.getBool("monitor", False)
mx      = OptDB.getInt("mx", 5)
my      = OptDB.getInt("my", 4)

# You can also :
da = PETSc.DMDA()
da.create(comm = PETSc.COMM_WORLD,
        dim=2, 
        sizes=(mx, my), 
        proc_sizes=None, #PETSC_DECIDE..
        boundary_type=(PETSc.DM.BoundaryType.NONE, PETSc.DM.BoundaryType.PERIODIC),
        stencil_type=PETSc.DMDA.StencilType.STAR,
        stencil_width=1,
        dof=1,
        setup=False)
da.setFromOptions()
da.setUp()
u = da.createGlobalVec()

heat = Heat2D(da, D0)
heat.rank = PETSc.COMM_WORLD.getRank()

ts = PETSc.TS().create(comm=PETSc.COMM_WORLD)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setDM(da)
ts.setRHSFunction(heat.FormRHSFunctionLocal)
ts.setRHSJacobian(heat.FormRHSJacobianLocal)

if monitor:
    ts.setMonitor(heat.EnergyMonitor)
ts.setType(PETSc.TS.Type.BDF)
ts.setTime(0.0)
ts.setMaxTime(0.1)
ts.setTimeStep(0.001)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setFromOptions()

t0 = ts.getTime()
tf = ts.getMaxTime()
mx, my = ts.getDM().getSizes()
PETSc.Sys.Print("solving on %d x %d grid for t0=%g to tf=%g ..."%(mx, my, t0, tf))

u.set(0.0)
ts.solve(u)

u.destroy()
ts.destroy()
da.destroy()