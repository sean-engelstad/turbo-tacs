# ============================================
#
# Make file for TACS_DIR/
#
# ============================================

include Makefile.in
include A2D_SHELLS.mk

TACS_SUBDIRS = src \
	src/bpmat \
	src/elements \
	src/elements/basis \
	src/constitutive \
	src/io \
	src/utils

TACS_OBJS := $(addsuffix /*.o, ${TACS_SUBDIRS})

default:
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
	   echo "Building Complex TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) TACS_DIR=${TACS_DIR} TACS_DEF="${TACS_DEF} -DTACS_USE_COMPLEX") || exit 1; \
            done \
	else \
	   echo "Building Real TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) TACS_DIR=${TACS_DIR} TACS_DEF=${TACS_DEF}) || exit 1; \
            done \
	fi
	${CXX} ${SO_LINK_FLAGS} ${TACS_OBJS} ${TACS_EXTERN_LIBS} -o ${TACS_DIR}/lib/liba2dshells.${SO_EXT}

debug:
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
	   echo "Building Complex TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) debug TACS_DIR=${TACS_DIR} TACS_DEF="${TACS_DEF} -DTACS_USE_COMPLEX") || exit 1; \
            done \
	else \
	   echo "Building Real TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) debug TACS_DIR=${TACS_DIR} TACS_DEF=${TACS_DEF}) || exit 1; \
            done \
	fi
	${CXX} ${SO_LINK_FLAGS} ${TACS_OBJS} ${TACS_EXTERN_LIBS} -o ${TACS_DIR}/lib/liba2dshells.${SO_EXT}

complex: TACS_IS_COMPLEX=true
complex: default

complex_debug: TACS_IS_COMPLEX=true
complex_debug: debug

clean:
	${RM} lib/libtacs.a lib/liba2dshells.a lib/liba2dshells.${SO_EXT}
	@for subdir in $(TACS_SUBDIRS) ; do \
	  echo "making $@ in $$subdir"; \
	  echo; \
	     (cd $$subdir && $(MAKE) $@ TACS_DIR=${TACS_DIR}) || exit 1; \
	done
