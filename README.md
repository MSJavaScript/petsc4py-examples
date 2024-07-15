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
