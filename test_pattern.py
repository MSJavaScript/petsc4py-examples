# -*- coding:utf-8 -*-
import math
import sys 
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

class Pattern():
    def __init__(self, da):
        self.da = da
        self.local_vec_Y = da.createLocalVec()

        self.L      = 2.5
        self.Du     = 8.0e-5
        self.Dv     = 4.0e-5
        self.phi    = 0.024
        self.kappa  = 0.06
        self.IFcn_called   = False
        self.IJac_called   = False
        self.RHSFcn_called = False
        self.RHSJac_called = False

    # in system form  F(t,Y,dot Y) = G(t,Y),  compute G():
    #     G^u(t,u,v) = - u v^2 + phi (1 - u)
    #     G^v(t,u,v) = + u v^2 - (phi + kappa) v
    def FormRHSFunctionLocal(self, ts, t, vec_Y, vec_G):
        self.RHSFcn_called = True 

        self.da.globalToLocal(vec_Y, self.local_vec_Y)
        aY = da.getVecArray(self.local_vec_Y)

        aG = da.getVecArray(vec_G)
        (xs, xe), (ys, ye) = da.getRanges()
        uv2 = aY[xs:xe, ys:ye, 0] * aY[xs:xe, ys:ye, 1] * aY[xs:xe, ys:ye, 1]
        aG[xs:xe, ys:ye, 0] = -uv2 + self.phi*(1.0 - aY[xs:xe, ys:ye,0])
        aG[xs:xe, ys:ye, 1] = uv2 - (self.phi+self.kappa)*aY[xs:xe, ys:ye, 1]
        # for j in range(ys, ye):
        #     for i in range(xs, xe):
        #         uv2 = aY[i,j,0]*aY[i,j,1]*aY[i,j,1]
        #         aG[i,j,0] = -uv2 + self.phi*(1.0 - aY[i,j,0])
        #         aG[i,j,1] = +uv2 - (self.phi+self.kappa)*aY[i,j,1]
        return True 

    def FormRHSJacobianLocal(self, ts, t, vec_Y, Mat_J, Mat_P):
        self.RHSJac_called = True
        aY = da.getVecArray(vec_Y)
        row = PETSc.Mat.Stencil()
        col = [PETSc.Mat.Stencil(), PETSc.Mat.Stencil()]
        v = [0, 0]

        (xs, xe), (ys, ye) = da.getRanges()
        for j in range(ys, ye):
            row.j = j;  col[0].j = j;  col[1].j = j 
            for i in range(xs, xe):
                row.i = i;  col[0].i = i;  col[1].i = i
                uv = aY[i, j, 0] * aY[i, j, 1]
                v2 = aY[i, j, 1] * aY[i, j, 1]
                # u equation
                row.c = 0;  col[0].c = 0;  col[1].c = 1
                v[0] = - v2 - self.phi
                v[1] = - 2.0 * uv
                Mat_P.setValueStencil(row, col[0], v[0], PETSc.InsertMode.INSERT_VALUES)
                Mat_P.setValueStencil(row, col[1], v[1], PETSc.InsertMode.INSERT_VALUES)
                # v equation
                row.c = 1;  col[0].c = 0;  col[1].c = 1
                v[0] = v2
                v[1] = 2.0 * uv - (self.phi + self.kappa)
                Mat_P.setValueStencil(row, col[0], v[0], PETSc.InsertMode.INSERT_VALUES)
                Mat_P.setValueStencil(row, col[1], v[1], PETSc.InsertMode.INSERT_VALUES)
        Mat_P.assemble()
        if Mat_J != Mat_P:
            Mat_J.assemble()
        return True
 
    # Callable[[TS, float, Vec, Vec, Vec], None]
    def FormIFunctionLocal(self, ts, t, vec_Y, vec_Ydot, vec_F):
        self.da.globalToLocal(vec_Y, self.local_vec_Y)
        aY    = self.da.getVecArray(self.local_vec_Y)
        aYdot = self.da.getVecArray(vec_Ydot)
        aF    = self.da.getVecArray(vec_F)

        mx, my = self.da.getSizes()
        h = self.L / mx 
        Cu = self.Du / (6.0*h*h)
        Cv = self.Dv / (6.0*h*h)
        self.IFcn_called = True 
        (xs, xe), (ys, ye) = da.getRanges()
        u = aY[xs:xe, ys:ye, 0]
        v = aY[xs:xe, ys:ye, 1]
        lapu = aY[(xs-1):(xe-1),(ys+1):(ye+1),0] \
            + 4.0*aY[xs:xe,(ys+1):(ye+1),0] + aY[(xs+1):(xe+1),(ys+1):(ye+1),0] \
            + 4.0*aY[(xs-1):(xe-1),ys:ye,0] - 20.0*u + 4.0*aY[(xs+1):(xe+1), ys:ye, 0] \
            + aY[(xs-1):(xe-1),(ys-1):(ye-1),0] + 4.0*aY[xs:xe,(ys-1):(ye-1),0] \
            + aY[(xs+1):(xe+1),(ys-1):(ye-1),0]
        lapv = aY[(xs-1):(xe-1),(ys+1):(ye+1),1] \
            + 4.0*aY[xs:xe,(ys+1):(ye+1),1] + aY[(xs+1):(xe+1),(ys+1):(ye+1),1] \
            + 4.0*aY[(xs-1):(xe-1),ys:ye,1] - 20.0*v + 4.0*aY[(xs+1):(xe+1), ys:ye, 1] \
            + aY[(xs-1):(xe-1),(ys-1):(ye-1),1] + 4.0*aY[xs:xe,(ys-1):(ye-1),1] \
            + aY[(xs+1):(xe+1),(ys-1):(ye-1),1]
        aF[xs:xe, ys:ye, 0] = aYdot[xs:xe, ys:ye, 0] - Cu*lapu
        aF[xs:xe, ys:ye, 1] = aYdot[xs:xe, ys:ye, 1] - Cv*lapv
        # for j in range(ys, ye):
        #     for i in range(xs, xe):
        #         u = aY[i, j, 0]
        #         v = aY[i, j, 1]
        #         lapu = aY[i-1,j+1,0] + 4.0*aY[i,j+1,0] + aY[i+1,j+1,0] + 4.0*aY[i-1,j,0] - 20.0*u + 4.0*aY[i+1, j, 0] + aY[i-1,j-1,0] + 4.0*aY[i,j-1,0] + aY[i+1,j-1,0]
        #         lapv = aY[i-1,j+1,1] + 4.0*aY[i,j+1,1] + aY[i+1,j+1,1] + 4.0*aY[i-1,j,1] - 20.0*v + 4.0*aY[i+1, j, 1] + aY[i-1,j-1,1] + 4.0*aY[i,j-1,1] + aY[i+1,j-1,1]
        #         aF[i,j,0] = aYdot[i,j,0] - Cu*lapu
        #         aF[i,j,1] = aYdot[i,j,1] - Cv*lapv
        return True
    
    # in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
    # Jacobian of F():
    #     J = (shift) dF/d(dot Y) + dF/dY
    # Callable[[TS, float, Vec, Vec, float, Mat, Mat], None]
    def FormIJacobianLocal(self, ts, t, vec_Y, vec_Ydot, shift, Mat_J, Mat_P):

        row = PETSc.Mat.Stencil()
        col = [PETSc.Mat.Stencil() for i in range(9)]
        val = [0]*9
        
        mx, my = self.da.getSizes()
        h = self.L / mx 
        Cu = self.Du / (6.0*h*h)
        Cv = self.Dv / (6.0*h*h)
        Mat_P.zeroEntries() # workaround to address PETSc issue #734
        self.IJac_called = True 
        (xs, xe), (ys, ye) = da.getRanges()
        for j in range(ys, ye):
            row.j = j 
            for i in range(xs, xe):
                row.i = i 
                for c in [0, 1]:
                    row.c = c
                    CC = Cu if c == 0 else Cv
                    for s in range(9):
                        col[s].c = c
                    col[0].i = i;   col[0].j = j
                    val[0] = shift + 20.0 * CC
                    col[1].i = i-1; col[1].j = j;    val[1] = - 4.0 * CC
                    col[2].i = i+1; col[2].j = j;    val[2] = - 4.0 * CC
                    col[3].i = i;   col[3].j = j-1;  val[3] = - 4.0 * CC
                    col[4].i = i;   col[4].j = j+1;  val[4] = - 4.0 * CC
                    col[5].i = i-1; col[5].j = j-1;  val[5] = - CC
                    col[6].i = i-1; col[6].j = j+1;  val[6] = - CC
                    col[7].i = i+1; col[7].j = j-1;  val[7] = - CC
                    col[8].i = i+1; col[8].j = j+1;  val[8] = - CC
                    for s in range(9):
                        Mat_P.setValueStencil(row, col[s], val[s], PETSc.InsertMode.INSERT_VALUES)
        Mat_P.assemble()
        if Mat_J != Mat_P:
            Mat_J.assemble()
        return True

def InitialState(da, Y, noiselevel, pattern_obj:Pattern):
    ledge = (pattern_obj.L - 0.5) / 2.0
    redge = pattern_obj.L - ledge
    Y.set(0.0)
    if noiselevel > 0.0:
        Y.setRandom()
        Y.scale(noiselevel)
    (xs, xe), (ys, ye) = da.getRanges()
    # getCoordinates returns a Vec which contains the (x,y)
    # coordinates of all the grid points on the mesh
    # It is one dimensional object and hard to use.
    # So we convert it to an array which has the same dimension 
    # as the mesh ...
    aC = da.getVecArray(da.getCoordinates())
    aY = da.getVecArray(Y)
    for j in range(ys, ye):
        for i in range(xs, xe):
            x = aC[i, j, 0]
            y = aC[i, j, 1]
            if (x >= ledge and x <= redge and y >= ledge and y <= redge):
                sx = math.sin(4.0 * math.pi * x)
                sy = math.sin(4.0 * math.pi * y)
                aY[i, j, 1] += 0.5*sx*sx*sy*sy # "u" 0, "v" 1
            aY[i, j, 0] += 1.0 - 2.0*aY[i, j, 1]
    return True


OptDB = PETSc.Options("ptn_")
call_back_report = OptDB.getBool("-call_back_report", False)
no_ijacobian     = OptDB.getBool("-no_ijacobian", False)
no_rhsjacobian   = OptDB.getBool("-no_rhsjacobian", False)
noiselevel       = OptDB.getReal("noisy_init", -1.0)

Du      = OptDB.getReal("Du", 8.0e-5)
Dv      = OptDB.getReal("Dv", 4.0e-5)
kappa   = OptDB.getReal("kappa", 0.06)
L       = OptDB.getReal("L", 2.5)
phi     = OptDB.getReal("phi", 0.024)

da = PETSc.DMDA()
da.create(comm = PETSc.COMM_WORLD,
        dim=2, 
        sizes=(3, 3), 
        proc_sizes=None, 
        boundary_type=(PETSc.DM.BoundaryType.PERIODIC, PETSc.DM.BoundaryType.PERIODIC),
        stencil_type=PETSc.DMDA.StencilType.BOX,
        stencil_width=1,
        dof=2,
        setup=False)
da.setFromOptions()
da.setUp()
#da.setFieldName(0, "u")
#da.setFieldName(1, "v")
(mx, my) = da.getSizes()
if mx != my:
    raise RuntimeError("mx must equals to my")
da.setUniformCoordinates(0.0, L, 0.0, L)
PETSc.Sys.Print("running on %d x %d grid with square cells of side h = %.6f ..."%(mx, my, L/mx))

pattern = Pattern(da)
pattern.Du = Du
pattern.Dv = Dv 
pattern.kappa = kappa
pattern.L = L 
pattern.phi = phi 

ts = PETSc.TS().create(comm=PETSc.COMM_WORLD)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setDM(da)
ts.setRHSFunction(pattern.FormRHSFunctionLocal)
if not no_rhsjacobian:
    ts.setRHSJacobian(pattern.FormRHSJacobianLocal)
ts.setIFunction(pattern.FormIFunctionLocal)
if not no_ijacobian:
    ts.setIJacobian(pattern.FormIJacobianLocal)
ts.setType(PETSc.TS.Type.ARKIMEX)
ts.setTime(0.0)
ts.setMaxTime(200.0)
ts.setTimeStep(5.0)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setFromOptions()

x = da.createGlobalVec()
InitialState(da, x, noiselevel, pattern)
ts.solve(x)

if call_back_report:
    ts_type = ts.getType()
    PETSc.Sys.Print("CALL-BACK REPORT\n  solver type: %s"%(ts_type))
    PETSc.Sys.Print("  IFunction:   %d  | IJacobian:   %d"%(int(pattern.IFcn_called), int(pattern.IJac_called)))
    PETSc.Sys.Print("  RHSFunction: %d  | RHSJacobian: %d"%(int(pattern.RHSFcn_called), int(pattern.RHSJac_called)))

x.destroy()
ts.destroy()
da.destroy()