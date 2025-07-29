# petsc4py-examples
Some examples for petsc4py.
These examples are from the book: PETSc for Partial Differential Equations.
I convert some C codes in this book to the python version.

To run the python code, you should install petsc. You can install it by conda
```bash
conda -c conda-forge install petsc petsc4py mpi4py matplotlib 
```
run the code by
```bash
mpiexec -np 4 python3 test_heat.py
```

The correspondin C codes are also given. You can compile the C code by the makefile.

Below are the summary of these files:
1. **heat.cpp, test_heat.py**: Solve the time-dependent heat equation in the Chapter 5 of the book.
2. **pattern.c, test_pattern.py**: Solve the coupled reaction-diffusion equations in the Chapter 5 of the book.
3. **phelm.c, interclude, test_helmholtz.py**: Solve the Helmholtz equation in the Chapter 9 of the book.
4. **fenics_deal.II_step9.py**: Solve the convection equation (step-9 of deal.II tutorial) by fenics.
