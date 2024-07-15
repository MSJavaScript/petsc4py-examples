export PETSC_DIR=/mnt/d/software_install/fenics
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CFLAGS += -pedantic

pattern: pattern.c
	-${CLINKER} -o pattern pattern.c  ${PETSC_LIB}

distclean:
	@rm -f *~ pattern *tmp
