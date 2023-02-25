EXECS= codedqr_on_sq pbmgs_mpi
MPICC?=mpiicc
CC = icc
IDIR = ${MKLROOT}/include
LDIR = ${MKLROOT}/lib/intel64
CFLAGS=-m64 -I$(IDIR) -L$(LDIR)
LIBS = -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
all: ${EXECS}
$(EXECS): %: %.c
	${MPICC} -o $@ $< $(CFLAGS) $(LIBS)
clean:
	rm ${EXECS}