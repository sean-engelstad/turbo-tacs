
TACS_LIB = ${TACS_DIR}/lib/liba2dshells.a

TACS_INCLUDE = -I${TACS_DIR}/src \
	-I${TACS_DIR}/src/bpmat \
	-I${TACS_DIR}/src/elements \
	-I${TACS_DIR}/src/elements/basis \
	-I${TACS_DIR}/src/elements/shell \
	-I${TACS_DIR}/src/constitutive \
	-I${TACS_DIR}/src/io \
	-I${TACS_DIR}/src/utils

# Set the command line flags to use for compilation
TACS_OPT_CC_FLAGS = ${GPU_CC_FLAGS} ${TACS_DEF} ${EXTRA_CC_FLAGS} ${METIS_INCLUDE} ${AMD_INCLUDE} ${TECIO_INCLUDE} ${A2D_INCLUDE} ${TACS_INCLUDE} ${MPI_INC}
TACS_DEBUG_CC_FLAGS = ${GPU_CC_FLAGS} ${TACS_DEF} ${EXTRA_DEBUG_CC_FLAGS} ${METIS_INCLUDE} ${AMD_INCLUDE} ${TECIO_INCLUDE} ${A2D_INCLUDE} ${TACS_INCLUDE} ${MPI_INC}

# By default, use the optimized flags
TACS_CC_FLAGS = ${TACS_OPT_CC_FLAGS}

# Set the linking flags to use
TACS_EXTERN_LIBS = ${AMD_LIBS} ${METIS_LIB} ${LAPACK_LIBS} ${TECIO_LIBS}
TACS_LD_FLAGS = ${EXTRA_LD_FLAGS} ${TACS_LD_CMD} ${TACS_EXTERN_LIBS} ${MPI_LIB} -lmpi

# This is the one rule that is used to compile all the
# source code in TACS
# change back and force to std=c++11 vs std=c++17
%.o: %.cpp
	${CXX} ${TACS_CC_FLAGS} -std=c++17 -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo
