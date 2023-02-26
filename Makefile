EXECS= codedqr_base codedqr_main pbmgs_mpi
MPICC?=mpiicc
CC = icc
IDIR = ${MKLROOT}/include
LDIR = ${MKLROOT}/lib/intel64
CFLAGS=-m64 -I$(IDIR) -L$(LDIR)
LIBS = -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl

codedqr: codedqr_base.o codedqr_main.o
	${MPICC} codedqr_base.o codedqr_main.o -o codedqr_main $(CFLAGS) $(LIBS)

codedqr_main.o: codedqr_base.o codedqr_main.c
	${MPICC} codedqr_base.o -c codedqr_main.c -o codedqr_main.o $(CFLAGS) $(LIBS)

codedqr_base.o: codedqr_base.c
	${MPICC} -c codedqr_base.c -o codedqr_base.o $(CFLAGS) $(LIBS)

.PHONY: clean codedqr

clean:
	rm *.o
	rm codedqr_main