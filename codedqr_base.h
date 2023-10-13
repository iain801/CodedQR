#ifndef CODEDQR_BASE   /* Include guard */
#define CODEDQR_BASE

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>
#include <mkl.h>

#define MASTER 0        /* taskid of first task */
#define DIST_Q 1        /* code for mpi send/recv */
#define COMP_Q 2        /* code for mpi send/recv */
#define COMP_R 3        /* code for mpi send/recv */

#define DEBUG 0         /* run in debug mode */
#define TEST_FAILIURE 1 /* test failiure and reconstruction */
#define SET_SEED 0      /* choose random seed, time-based if 0 */
#define DO_FINAL_SOLVE 0 /* solve for x after QR */ 

extern VSLStreamStatePtr stream;
extern MPI_Comm glob_comm, row_comm, col_comm;

struct ReconInfo {
    double *Gv_tilde, *Gh_tilde, t_decode;
    int p_rank, proc_cols, proc_rows, 
        max_fails, loc_cols, loc_rows;
};

extern struct ReconInfo recon_info;

void printMatrix(double* matrix, int cols, int rows);

void fprintMatrix(FILE* out, double* matrix, int cols, int rows);

void iprintMatrix(int* matrix, int cols, int rows);

void randMatrixR(double* A, int n, int m, int rowsize);

void randMatrix(double* A, int n, int m);

double checkError(double* A, double* Q, double* R, double* E, 
    int loc_cols, int loc_rows, int glob_cols, int glob_rows);

void scatterA(double* A, double* Q, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows, int max_fails);

void gatherA(double** A,  int p_rank, int proc_cols, 
    int proc_rows, int loc_cols, int loc_rows);

void constructGv(double* Gv, int proc_rows, int f);

void constructGh(double* Gh, int proc_cols, int f);

void checksumV(double *Q, int p_rank);

void checksumH(double *Q, int p_rank);

void genFail(double* Q, double* R, int* col_status, int* row_status, int p_rank);

void reconstructQ(double* Q, int* node_status, int p_rank);

void reconstructR(double* R, int* node_status, int p_rank);

void pbmgs(double* Q, double* R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows);

void postOrthogonalize(double* Q, double* Gv_tilde, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows, int max_fails);

#endif // CODEDQR_BASE