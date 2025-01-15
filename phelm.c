static char help[] =
"Solves the p-Helmholtz equation in 2D using Q_1 FEM.  Option prefix -ph_.\n"
"Problem is posed as minimizing this objective functional over W^{1,p}\n"
"for p>1:\n"
"    I[u] = int_Omega (1/p) |grad u|^p + (1/2) u^2 - f u.\n"
"The strong form equation, namely setting the gradient to zero, is a PDE\n"
"    - div( |grad u|^{p-2} grad u ) + u = f\n"
"subject to homogeneous Neumann boundary conditions.  Implements objective\n"
"and gradient (residual) but no Hessian (Jacobian).  Defaults to linear\n"
"problem (p=2) and quadrature degree 2.  Can be run with only an objective\n" "function; use -ph_no_gradient -snes_fd_function.\n\n";

#include <petsc.h>
#include "../interlude/quadrature.h"

typedef struct {
    PetscReal  p, eps;
    PetscInt   quadpts;
    PetscReal  (*f)(PetscReal x, PetscReal y, PetscReal p, PetscReal eps);
} PHelmCtx;

static PetscReal f_constant(PetscReal x, PetscReal y, PetscReal p, PetscReal eps) {
    return 1.0;
}

static PetscReal u_exact_cosines(PetscReal x, PetscReal y, PetscReal p, PetscReal eps) {
    return PetscCosReal(PETSC_PI * x) * PetscCosReal(PETSC_PI * y);
}

static PetscReal f_cosines(PetscReal x, PetscReal y, PetscReal p, PetscReal eps) {
    const PetscReal uu = u_exact_cosines(x,y,p,eps),
                    pi2 = PETSC_PI * PETSC_PI,
                    lapu = - 2 * pi2 * uu;
    if (p == 2.0) {
        return - lapu + uu;
    } else {
        const PetscReal
            ux = - PETSC_PI * PetscSinReal(PETSC_PI * x)
                 * PetscCosReal(PETSC_PI * y),
            uy = - PETSC_PI * PetscCosReal(PETSC_PI * x)
                 * PetscSinReal(PETSC_PI * y),
            // note regularization changes f(x,y) but not u(x,y):
            w = ux * ux + uy * uy + eps * eps,
            pi3 = pi2 * PETSC_PI,
            wx = pi3 * PetscSinReal(2 * PETSC_PI * x)
                 * PetscCosReal(2 * PETSC_PI * y),
            wy = pi3 * PetscCosReal(2 * PETSC_PI * x)
                 * PetscSinReal(2 * PETSC_PI * y);
        const PetscReal s = (p - 2) / 2;  //  -1/2 <= s <= 0
        return - s * PetscPowScalar(w,s-1) * (wx * ux + wy * uy)
               - PetscPowScalar(w,s) * lapu + uu;
    }
}

typedef enum {CONSTANT, COSINES} ProblemType;
static const char* ProblemTypes[] = {"constant","cosines",
                                     "ProblemType", "", NULL};

extern PetscErrorCode GetVecFromFunction(DMDALocalInfo*, Vec,
                         PetscReal (*)(PetscReal, PetscReal, PetscReal, PetscReal), PHelmCtx*);
extern PetscErrorCode FormObjectiveLocal(DMDALocalInfo*, PetscReal**, PetscReal*, PHelmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal**, PetscReal**, PHelmCtx*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    PHelmCtx       user;
    DMDALocalInfo  info;
    ProblemType    problem = COSINES;
    PetscBool      no_objective = PETSC_FALSE,
                   no_gradient = PETSC_FALSE,
                   exact_init = PETSC_FALSE,
                   view_f = PETSC_FALSE;
    PetscReal      err;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    user.p = 2.0;
    user.eps = 0.0;
    user.quadpts = 2;
    PetscOptionsBegin(PETSC_COMM_WORLD,"ph_",
                  "p-Helmholtz solver options","");
    PetscCall(PetscOptionsReal("-eps",
                  "regularization parameter eps",
                  "phelm.c",user.eps,&(user.eps),NULL));
    PetscCall(PetscOptionsBool("-exact_init",
                  "use exact solution to initialize",
                  "phelm.c",exact_init,&(exact_init),NULL));
    PetscCall(PetscOptionsBool("-no_objective",
                  "do not set the objective evaluation function",
                  "phelm.c",no_objective,&(no_objective),NULL));
    PetscCall(PetscOptionsBool("-no_gradient",
                  "do not set the residual evaluation function",
                  "phelm.c",no_gradient,&(no_gradient),NULL));
    PetscCall(PetscOptionsReal("-p",
                  "exponent p > 1",
                  "phelm.c",user.p,&(user.p),NULL));
    if (user.p < 1.0) {
         SETERRQ(PETSC_COMM_SELF,1,"p >= 1 required");
    }
    if (user.p == 1.0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "WARNING: well-posedness only known for p > 1\n"));
    }
    PetscCall(PetscOptionsEnum("-problem",
                  "problem type determines right side f(x,y)",
                  "phelm.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,
                  NULL));
    PetscCall(PetscOptionsInt("-quadpts",
                  "number n of quadrature points in each direction (= 1,2,3 only)",
                  "phelm.c",user.quadpts,&(user.quadpts),NULL));
    if ((user.quadpts < 1) || (user.quadpts > 3)) {
        SETERRQ(PETSC_COMM_SELF,3,"quadrature points n=1,2,3 only");
    }
    PetscCall(PetscOptionsBool("-view_f",
                  "view right-hand side to STDOUT",
                  "phelm.c",view_f,&(view_f),NULL));
    PetscOptionsEnd();

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
           2,2,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0));
    PetscCall(DMDAGetLocalInfo(da,&info));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    if (!no_objective) {
        PetscCall(DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjectiveFn *)FormObjectiveLocal,&user));
    }
    if (no_gradient) {
        // why isn't this the default?  why no programmatic way to set?
        PetscCall(PetscOptionsSetValue(NULL,"-snes_fd_function_eps","0.0"));
    } else {
        PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunctionFn *)FormFunctionLocal,&user));
    }
    PetscCall(SNESSetFromOptions(snes));

    // set initial iterate and right-hand side
    PetscCall(DMCreateGlobalVector(da,&u_initial));
    PetscCall(VecSet(u_initial,0.5));
    switch (problem) {
        case CONSTANT:
            if (exact_init) {
                PetscCall(VecSet(u_initial,1.0));
            }
            user.f = &f_constant;
            break;
        case COSINES:
            if (exact_init) {
                PetscCall(GetVecFromFunction(&info,u_initial,&u_exact_cosines,&user));
            }
            user.f = &f_cosines;
            break;
        default:
            SETERRQ(PETSC_COMM_SELF,4,"unknown problem type\n");
    }

    // optionally view right-hand-side on initial grid
    if (view_f) {
        Vec vf;
        PetscCall(VecDuplicate(u_initial,&vf));
        switch (problem) {
            case CONSTANT:
                PetscCall(VecSet(vf,1.0));
                break;
            case COSINES:
                PetscCall(GetVecFromFunction(&info,vf,&f_cosines,&user));
                break;
        }
        PetscCall(VecView(vf,PETSC_VIEWER_STDOUT_WORLD));
        VecDestroy(&vf);
    }

    // solve and clean up
    PetscCall(SNESSolve(snes,NULL,u_initial));
    PetscCall(VecDestroy(&u_initial));
    PetscCall(DMDestroy(&da));
    PetscCall(SNESGetSolution(snes,&u));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));

    // evaluate numerical error
    PetscCall(VecDuplicate(u,&u_exact));
    switch (problem) {
        case CONSTANT:
            PetscCall(VecSet(u_exact,1.0));
            break;
        case COSINES:
            PetscCall(GetVecFromFunction(&info,u_exact,&u_exact_cosines,&user));
            break;
    }
    PetscCall(VecAXPY(u,-1.0,u_exact));    // u <- u + (-1.0) uexact
    PetscCall(VecNorm(u,NORM_INFINITY,&err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid with p=%.3f ...\n"
        "  numerical error:  |u-u_exact|_inf = %.3e\n",
        info.mx,info.my,user.p,err));

    PetscCall(VecDestroy(&u_exact));
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode GetVecFromFunction(DMDALocalInfo *info, Vec w,
         PetscReal (*fcn)(PetscReal, PetscReal, PetscReal, PetscReal),
         PHelmCtx *user) {
    const PetscReal hx = 1.0 / (info->mx - 1), hy = 1.0 / (info->my - 1);
    PetscReal       x, y, **aw;
    PetscInt        i, j;
    PetscCall(DMDAVecGetArray(info->da,w,&aw));
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            aw[j][i] = (*fcn)(x,y,user->p,user->eps);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da,w,&aw));
    return 0;
}

//STARTFEM
static PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
                 etaL[4] = { 1.0,  1.0, -1.0, -1.0};

static PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

// evaluate v(xi,eta) on reference element using local node numbering
static PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    return   v[0] * chi(0,xi,eta) + v[1] * chi(1,xi,eta)
           + v[2] * chi(2,xi,eta) + v[3] * chi(3,xi,eta);
}

typedef struct {
    PetscReal  xi, eta;
} gradRef;

static gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

// evaluate partial derivs of v(xi,eta) on reference element
static gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

static PetscReal GradInnerProd(PetscReal hx, PetscReal hy,
                               gradRef du, gradRef dv) {
    const PetscReal cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi * dv.xi + cy * du.eta * dv.eta;
}

static PetscReal GradPow(PetscReal hx, PetscReal hy,
                         gradRef du, PetscReal P, PetscReal eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0);
}
//ENDFEM

/* FLOPS:  (counting PetscPowScalar as 1)
     chi = 6
     eval = 4*6+7 = 31
     dchi = 8
     deval = 4*8+4 = 36
     GradInnerProd = 9
     GradPow = 9+4 = 13
     ObjIntegrandRef = deval + 2*eval + GradPow + 10 = 121
     FunIntegrandRef = chi + dchi + 2*eval + deval + GradPo + GradInnerProd + 9
                     = 143
*/

//STARTOBJECTIVE
static PetscReal ObjIntegrandRef(DMDALocalInfo *info,
                       const PetscReal ff[4], const PetscReal uu[4],
                       PetscReal xi, PetscReal eta, PHelmCtx *user) {
    const gradRef    du = deval(uu,xi,eta);
    const PetscReal  hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1),
                     u = eval(uu,xi,eta);
    return GradPow(hx,hy,du,user->p,0.0) / user->p + 0.5 * u * u
           - eval(ff,xi,eta) * u;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PHelmCtx *user) {
  const PetscReal hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1);
  const Quad1D    q = gausslegendre[user->quadpts-1];
  PetscReal       x, y, lobj = 0.0;
  PetscInt        i,j,r,s;
  MPI_Comm        com;

  // loop over all elements
  for (j = info->ys; j < info->ys + info->ym; j++) {
      if (j == 0)
          continue;
      y = j * hy;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if (i == 0)
              continue;
          x = i * hx;
          const PetscReal ff[4] = {user->f(x,y,user->p,user->eps),
                                   user->f(x-hx,y,user->p,user->eps),
                                   user->f(x-hx,y-hy,user->p,user->eps),
                                   user->f(x,y-hy,user->p,user->eps)};
          const PetscReal uu[4] = {au[j][i],au[j][i-1],
                                   au[j-1][i-1],au[j-1][i]};
          // loop over quadrature points on this element
          for (r = 0; r < q.n; r++) {
              for (s = 0; s < q.n; s++) {
                  lobj += q.w[r] * q.w[s]
                          * ObjIntegrandRef(info,ff,uu,
                                            q.xi[r],q.xi[s],user);
              }
          }
      }
  }
  lobj *= hx * hy / 4.0;  // from change of variables formula
  PetscCall(PetscObjectGetComm((PetscObject)(info->da),&com));
  PetscCall(MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,com));
  PetscCall(PetscLogFlops(129*info->xm*info->ym));
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
static PetscReal IntegrandRef(DMDALocalInfo *info, PetscInt L,
                     const PetscReal ff[4], const PetscReal uu[4],
                     PetscReal xi, PetscReal eta, PHelmCtx *user) {
  const gradRef    du    = deval(uu,xi,eta),
                   dchiL = dchi(L,xi,eta);
  const PetscReal  hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1);
  return GradPow(hx,hy,du,user->p - 2.0,user->eps)
           * GradInnerProd(hx,hy,du,dchiL)
         + (eval(uu,xi,eta) - eval(ff,xi,eta)) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PHelmCtx *user) {
  const PetscReal hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1);
  const Quad1D    q = gausslegendre[user->quadpts-1];
  const PetscInt  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
  PetscReal       x, y;
  PetscInt        i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < info->ys + info->ym; j++)
      for (i = info->xs; i < info->xs + info->xm; i++)
          FF[j][i] = 0.0;

  // loop over all elements
  for (j = info->ys; j <= info->ys + info->ym; j++) {
      if ((j == 0) || (j > info->my-1))
          continue;
      y = j * hy;
      for (i = info->xs; i <= info->xs + info->xm; i++) {
          if ((i == 0) || (i > info->mx-1))
              continue;
          x = i * hx;
          const PetscReal ff[4] = {user->f(x,y,user->p,user->eps),
                                   user->f(x-hx,y,user->p,user->eps),
                                   user->f(x-hx,y-hy,user->p,user->eps),
                                   user->f(x,y-hy,user->p,user->eps)};
          const PetscReal uu[4] = {au[j][i],au[j][i-1],
                                   au[j-1][i-1],au[j-1][i]};
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // only update residual if we own node
              if (PP >= info->xs && PP < info->xs + info->xm
                  && QQ >= info->ys && QQ < info->ys + info->ym) {
                  // loop over quadrature points
                  for (r = 0; r < q.n; r++) {
                      for (s = 0; s < q.n; s++) {
                         FF[QQ][PP]
                             += 0.25 * hx * hy * q.w[r] * q.w[s]
                                * IntegrandRef(info,l,ff,uu,
                                               q.xi[r],q.xi[s],user);
                      }
                  }
              }
          }
      }
  }
  PetscCall(PetscLogFlops((5+q.n*q.n*149)*(info->xm+1)*(info->ym+1)));
  return 0;
}
//ENDFUNCTION
