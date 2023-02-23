EXECS= codedqr_on_sq pbmgs_mpi
MPICC?=mpicc
CC = gcc
IDIR = ${MKLROOT}/include
LDIR = ${MKLROOT}/lib/intel64
CFLAGS=-m64 -I$(IDIR) -L$(LDIR)
LIBS = -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
all: ${EXECS}
$(EXECS): %: %.c
	${MPICC} -o $@ $< $(CFLAGS) $(LIBS)
clean:
	rm ${EXECS}