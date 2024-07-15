static char help[] =
"Coupled reaction-diffusion equations (Pearson 1993).  Option prefix -ptn_.\n"
"Demonstrates form  F(t,Y,dot Y) = G(t,Y)  where F() is IFunction and G() is\n"
"RHSFunction().  Implements IJacobian() and RHSJacobian().  Defaults to\n"
"ARKIMEX (= adaptive Runge-Kutta implicit-explicit) TS type.\n\n";

#include <petsc.h>

typedef struct {
  PetscReal u, v;
} Field;

typedef struct {
  PetscReal  L,     // domain side length
             Du,    // diffusion coefficient: u equation
             Dv,    //                        v equation
             phi,   // "dimensionless feed rate" (F in Pearson 1993)
             kappa; // "dimensionless rate constant" (k in Pearson 1993)
  PetscBool  IFcn_called, IJac_called, RHSFcn_called, RHSJac_called;
} PatternCtx;

extern PetscErrorCode InitialState(DM, Vec, PetscReal, PatternCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal, Field**,
                                           Field**, PatternCtx*);
extern PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo*, PetscReal, Field**,
                                           Mat, Mat, PatternCtx*);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*, PetscReal, Field**, Field**,
                                         Field **, PatternCtx*);
extern PetscErrorCode FormIJacobianLocal(DMDALocalInfo*, PetscReal, Field**, Field**,
                                         PetscReal, Mat, Mat, PatternCtx*);

int main(int argc,char **argv)
{
  PatternCtx     user;
  TS             ts;
  Vec            x;
  DM             da;
  DMDALocalInfo  info;
  PetscReal      noiselevel = -1.0;  // negative value means no initial noise
  PetscBool      no_rhsjacobian = PETSC_FALSE,
                 no_ijacobian = PETSC_FALSE,
                 call_back_report = PETSC_FALSE;
  TSType         type;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  // parameter values from pages 21-22 in Hundsdorfer & Verwer (2003)
  user.L      = 2.5;
  user.Du     = 8.0e-5;
  user.Dv     = 4.0e-5;
  user.phi    = 0.024;
  user.kappa  = 0.06;
  user.IFcn_called   = PETSC_FALSE;
  user.IJac_called   = PETSC_FALSE;
  user.RHSFcn_called = PETSC_FALSE;
  user.RHSJac_called = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "options for patterns", "");
  PetscCall(PetscOptionsBool("-call_back_report","report on which user-supplied call-backs were actually called",
           "pattern.c",call_back_report,&(call_back_report),NULL));
  PetscCall(PetscOptionsReal("-Du","diffusion coefficient of first equation",
           "pattern.c",user.Du,&user.Du,NULL));
  PetscCall(PetscOptionsReal("-Dv","diffusion coefficient of second equation",
           "pattern.c",user.Dv,&user.Dv,NULL));
  PetscCall(PetscOptionsReal("-kappa","dimensionless rate constant (=k in (Pearson, 1993))",
           "pattern.c",user.kappa,&user.kappa,NULL));
  PetscCall(PetscOptionsReal("-L","square domain side length; recommend L >= 0.5",
           "pattern.c",user.L,&user.L,NULL));
  PetscCall(PetscOptionsBool("-no_ijacobian","do not set call-back DMDATSSetIJacobian()",
           "pattern.c",no_ijacobian,&(no_ijacobian),NULL));
  PetscCall(PetscOptionsBool("-no_rhsjacobian","do not set call-back DMDATSSetRHSJacobian()",
           "pattern.c",no_rhsjacobian,&(no_rhsjacobian),NULL));
  PetscCall(PetscOptionsReal("-noisy_init",
           "initialize u,v with this much random noise (e.g. 0.2) on top of usual initial values",
           "pattern.c",noiselevel,&noiselevel,NULL));
  PetscCall(PetscOptionsReal("-phi","dimensionless feed rate (=F in (Pearson, 1993))",
           "pattern.c",user.phi,&user.phi,NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
               DMDA_STENCIL_BOX,  // for 9-point stencil
               3,3,PETSC_DECIDE,PETSC_DECIDE,
               2, 1,              // degrees of freedom, stencil width
               NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da,0,"u"));
  PetscCall(DMDASetFieldName(da,1,"v"));
  PetscCall(DMDAGetLocalInfo(da,&info));
  if (info.mx != info.my) {
      SETERRQ(PETSC_COMM_SELF,1,"pattern.c requires mx == my");
  }
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.L, 0.0, user.L, -1.0, -1.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
           "running on %d x %d grid with square cells of side h = %.6f ...\n",
           info.mx,info.my,user.L/(PetscReal)(info.mx)));

//STARTTSSETUP
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetApplicationContext(ts,&user));
  PetscCall(DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user));
  if (!no_rhsjacobian) {
      PetscCall(DMDATSSetRHSJacobianLocal(da,
               (DMDATSRHSJacobianLocal)FormRHSJacobianLocal,&user));
  }
  PetscCall(DMDATSSetIFunctionLocal(da,INSERT_VALUES,
           (DMDATSIFunctionLocal)FormIFunctionLocal,&user));
  if (!no_ijacobian) {
      PetscCall(DMDATSSetIJacobianLocal(da,
               (DMDATSIJacobianLocal)FormIJacobianLocal,&user));
  }
  PetscCall(TSSetType(ts,TSARKIMEX));
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetMaxTime(ts,200.0));
  PetscCall(TSSetTimeStep(ts,5.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
//ENDTSSETUP

  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(InitialState(da,x,noiselevel,&user));
  PetscCall(TSSolve(ts,x));

  // optionally report on call-backs
  if (call_back_report) {
      PetscCall(TSGetType(ts,&type));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"CALL-BACK REPORT\n  solver type: %s\n",type));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  IFunction:   %d  | IJacobian:   %d\n",
                                          (int)user.IFcn_called,(int)user.IJac_called));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  RHSFunction: %d  | RHSJacobian: %d\n",
                                          (int)user.RHSFcn_called,(int)user.RHSJac_called));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

// Formulas from page 22 of Hundsdorfer & Verwer (2003).  Interpretation here is
// to always generate 0.5 x 0.5 non-trivial patch in (0,L) x (0,L) domain.
PetscErrorCode InitialState(DM da, Vec Y, PetscReal noiselevel, PatternCtx* user) {
  DMDALocalInfo    info;
  PetscInt         i,j;
  PetscReal        sx,sy;
  const PetscReal  ledge = (user->L - 0.5) / 2.0, // nontrivial initial values on
                   redge = user->L - ledge;       //   ledge < x,y < redge
  DMDACoor2d       **aC;
  Field            **aY;

  PetscCall(VecSet(Y,0.0));
  if (noiselevel > 0.0) {
      // noise added to usual initial condition is uniform on [0,noiselevel],
      //     independently for each location and component
      PetscCall(VecSetRandom(Y,NULL));
      PetscCall(VecScale(Y,noiselevel));
  }
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAGetCoordinateArray(da,&aC));
  PetscCall(DMDAVecGetArray(da,Y,&aY));
  for (j = info.ys; j < info.ys+info.ym; j++) {
    for (i = info.xs; i < info.xs+info.xm; i++) {
      if ((aC[j][i].x >= ledge) && (aC[j][i].x <= redge)
              && (aC[j][i].y >= ledge) && (aC[j][i].y <= redge)) {
          sx = PetscSinReal(4.0 * PETSC_PI * aC[j][i].x);
          sy = PetscSinReal(4.0 * PETSC_PI * aC[j][i].y);
          aY[j][i].v += 0.5 * sx * sx * sy * sy;
      }
      aY[j][i].u += 1.0 - 2.0 * aY[j][i].v;
    }
  }
  PetscCall(DMDAVecRestoreArray(da,Y,&aY));
  PetscCall(DMDARestoreCoordinateArray(da,&aC));
  return 0;
}

// in system form  F(t,Y,dot Y) = G(t,Y),  compute G():
//     G^u(t,u,v) = - u v^2 + phi (1 - u)
//     G^v(t,u,v) = + u v^2 - (phi + kappa) v
//STARTRHSFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info,
                   PetscReal t, Field **aY, Field **aG, PatternCtx *user) {
  PetscInt   i, j;
  PetscReal  uv2;

  user->RHSFcn_called = PETSC_TRUE;
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          uv2 = aY[j][i].u * aY[j][i].v * aY[j][i].v;
          aG[j][i].u = - uv2 + user->phi * (1.0 - aY[j][i].u);
          aG[j][i].v = + uv2 - (user->phi + user->kappa) * aY[j][i].v;
      }
  }
  return 0;
}
//ENDRHSFUNCTION

PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo *info,
                                    PetscReal t, Field **aY,
                                    Mat J, Mat P, PatternCtx *user) {
    PetscInt    i, j;
    PetscReal   v[2], uv, v2;
    MatStencil  col[2],row;

    user->RHSJac_called = PETSC_TRUE;
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;  col[0].j = j;  col[1].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;  col[0].i = i;  col[1].i = i;
            uv = aY[j][i].u * aY[j][i].v;
            v2 = aY[j][i].v * aY[j][i].v;
            // u equation
            row.c = 0;  col[0].c = 0;  col[1].c = 1;
            v[0] = - v2 - user->phi;
            v[1] = - 2.0 * uv;
            PetscCall(MatSetValuesStencil(P,1,&row,2,col,v,INSERT_VALUES));
            // v equation
            row.c = 1;  col[0].c = 0;  col[1].c = 1;
            v[0] = v2;
            v[1] = 2.0 * uv - (user->phi + user->kappa);
            PetscCall(MatSetValuesStencil(P,1,&row,2,col,v,INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}

// in system form  F(t,Y,dot Y) = G(t,Y),  compute F():
//     F^u(t,u,v,u_t,v_t) = u_t - D_u Laplacian u
//     F^v(t,u,v,u_t,v_t) = v_t - D_v Laplacian v
//STARTIFUNCTION
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info, PetscReal t,
                                  Field **aY, Field **aYdot, Field **aF,
                                  PatternCtx *user) {
  PetscInt         i, j;
  const PetscReal  h = user->L / (PetscReal)(info->mx),
                   Cu = user->Du / (6.0 * h * h),
                   Cv = user->Dv / (6.0 * h * h);
  PetscReal        u, v, lapu, lapv;

  user->IFcn_called = PETSC_TRUE;
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          u = aY[j][i].u;
          v = aY[j][i].v;
          lapu =     aY[j+1][i-1].u + 4.0*aY[j+1][i].u +   aY[j+1][i+1].u
                 + 4.0*aY[j][i-1].u -    20.0*u        + 4.0*aY[j][i+1].u
                 +   aY[j-1][i-1].u + 4.0*aY[j-1][i].u +   aY[j-1][i+1].u;
          lapv =     aY[j+1][i-1].v + 4.0*aY[j+1][i].v +   aY[j+1][i+1].v
                 + 4.0*aY[j][i-1].v -    20.0*v        + 4.0*aY[j][i+1].v
                 +   aY[j-1][i-1].v + 4.0*aY[j-1][i].v +   aY[j-1][i+1].v;
          aF[j][i].u = aYdot[j][i].u - Cu * lapu;
          aF[j][i].v = aYdot[j][i].v - Cv * lapv;
      }
  }
  return 0;
}
//ENDIFUNCTION

// in system form  F(t,Y,dot Y) = G(t,Y),  compute combined/shifted
// Jacobian of F():
//     J = (shift) dF/d(dot Y) + dF/dY
//STARTIJACOBIAN
PetscErrorCode FormIJacobianLocal(DMDALocalInfo *info,
                   PetscReal t, Field **aY, Field **aYdot,
                   PetscReal shift, Mat J, Mat P,
                   PatternCtx *user) {
    PetscInt         i, j, s, c;
    const PetscReal  h = user->L / (PetscReal)(info->mx),
                     Cu = user->Du / (6.0 * h * h),
                     Cv = user->Dv / (6.0 * h * h);
    PetscReal        val[9], CC;
    MatStencil       col[9], row;

    PetscCall(MatZeroEntries(P));  // workaround to address PETSc issue #734
    user->IJac_called = PETSC_TRUE;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        row.j = j;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            row.i = i;
            for (c = 0; c < 2; c++) { // u,v equations are c=0,1
                row.c = c;
                CC = (c == 0) ? Cu : Cv;
                for (s = 0; s < 9; s++)
                    col[s].c = c;
                col[0].i = i;   col[0].j = j;
                val[0] = shift + 20.0 * CC;
                col[1].i = i-1; col[1].j = j;    val[1] = - 4.0 * CC;
                col[2].i = i+1; col[2].j = j;    val[2] = - 4.0 * CC;
                col[3].i = i;   col[3].j = j-1;  val[3] = - 4.0 * CC;
                col[4].i = i;   col[4].j = j+1;  val[4] = - 4.0 * CC;
                col[5].i = i-1; col[5].j = j-1;  val[5] = - CC;
                col[6].i = i-1; col[6].j = j+1;  val[6] = - CC;
                col[7].i = i+1; col[7].j = j-1;  val[7] = - CC;
                col[8].i = i+1; col[8].j = j+1;  val[8] = - CC;
                PetscCall(MatSetValuesStencil(P,1,&row,9,col,val,INSERT_VALUES));
            }
        }
    }

    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
//ENDIJACOBIAN
