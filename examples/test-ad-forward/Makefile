include ../../Makefile.in
include ../../A2D_SHELLS.mk

LOCAL_OBJS = TestADForward.o
LOCAL_EXECS = test_ad_forward.x

default: ${LOCAL_OBJS} ${LOCAL_EXECS}

test_ad_forward.x:
	${CXX} -o test_ad_forward.x TestADForward.o ${TACS_LD_FLAGS}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

complex: TACS_DEF="-DTACS_USE_COMPLEX"
complex: default

complex_debug: TACS_DEF="-DTACS_USE_COMPLEX"
complex_debug: debug


new:
	$(MAKE) clean
	$(MAKE) debug

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.f5
	rm -f *.vtk
