/* Coded Parallel Block Modified Gram Schmidt
 * implementation with MPI using [INSERT] checksum storage
 * for random square matrix A(n,n)
 * Using MGS in place of ICGS 
 *
 * Iain Weissburg 2023
 */

#ifndef CODEDQR_BASE   /* Include guard */
#define CODEDQR_BASE

#define MKL_INT int
#define MKL_DOUBLE double
#define cblas_d double

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#include <mpi.h>
#include <mkl.h>

#define MASTER 0        /* taskid of first task */
#define DIST_Q 1        /* code for mpi send/recv */
#define COMP_Q 2        /* code for mpi send/recv */
#define COMP_R 3        /* code for mpi send/recv */

#define DEBUG 0         /* run in debug mode */
#define SET_SEED 42     /* choose random seed, time-based if 0 */

VSLStreamStatePtr stream;
MPI_Comm glob_comm, row_comm, col_comm;
MPI_Group row_group, col_group;

struct ReconInfo {
    double* Gv_tilde;
    int p_rank, proc_cols, proc_rows, 
        max_fails, loc_cols, loc_rows;
};

struct ReconInfo recon_inf;

void printMatrix(double* matrix, int cols, int rows);

void iprintMatrix(int* matrix, int cols, int rows);

void randMatrix(double* A, int n, int m);

double checkError(double* A, double* Q, double* R, double* E, 
    int loc_cols, int loc_rows, int glob_cols, int glob_rows);

void scatterA(double* A, double* Q, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows);

void gatherA(double** A,  int p_rank, int proc_cols, 
    int proc_rows, int loc_cols, int loc_rows);

void constructGv(double* Gv, int proc_rows, int f);

void constructGh(double* Gh, int proc_cols, int f);

void checksumQ(double *Q, int p_rank);

void genFail(double* Q, int target_rank, int p_rank, int loc_cols, int loc_rows);

void reconstructQ(double* Q, int* node_status, int p_rank);

void pbmgs(double* Q, double* R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows);

#endif // CODEDQR_H