include ../../Makefile.in
include ../../A2D_SHELLS.mk

# LOCAL_OBJS = RunCRM.o RunCRM_orig.o
# LOCAL_EXECS = runcrm.x runcrm_orig.x

LOCAL_OBJS = RunCRM.o
LOCAL_EXECS = runcrm.x

default: ${LOCAL_OBJS} ${LOCAL_EXECS}

runcrm.x:
	${CXX} -o runcrm.x RunCRM.o ${TACS_LD_FLAGS}

runcrm_orig.x:
	${CXX} -o runcrm_orig.x RunCRM_orig.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

complex_debug: TACS_DEF="-DTACS_USE_COMPLEX"
complex_debug: debug


new:
	make clean
	make debug

run:
	./runcrm.x >> new.txt
	./runcrm_orig.x >> orig.txt

newrun:
	make new
	make run

clean:
	rm -f *.o || echo "no .o files"
	rm -f *.x || echo "no .x files"
	rm -f *.f5 || echo "no .f5 files"
	rm -f *.vtk || echo "no .vtk files"
	rm *.txt || echo "no *.txt files to remove"