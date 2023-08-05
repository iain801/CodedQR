EXECS= codedqr_base codedqr_main pbmgs_mpi
MPICC?=mpiicc 
CC = icc
IDIR = ${MKLROOT}/include
LDIR = ${MKLROOT}/lib/intel64
CFLAGS= -Wall -m64 -Ofast -ipo -fp-model precise -I$(IDIR) -L$(LDIR)
LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

MAKEDIR = ./out

codedqr: codedqr_base.o codedqr_main.o
	${MPICC} ${MAKEDIR}/codedqr_base.o ${MAKEDIR}/codedqr_main.o -o ${MAKEDIR}/codedqr_main $(CFLAGS) $(LIBS)

codedqr_main.o: codedqr_base.o codedqr_main.c
	${MPICC} ${MAKEDIR}/codedqr_base.o -c codedqr_main.c -o ${MAKEDIR}/codedqr_main.o $(CFLAGS)

codedqr_base.o: codedqr_base.c
	${MPICC} -c codedqr_base.c -o ${MAKEDIR}/codedqr_base.o $(CFLAGS)

.PHONY: clean codedqr

clean:
	rm ${MAKEDIR}/*.o
	rm ${MAKEDIR}/codedqr_main