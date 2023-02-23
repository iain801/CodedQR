/* Coded Parallel Block Modified Gram Schmidt
 * implementation with MPI using [INSERT] checksum storage
 * for random square matrix A(n,n)
 * Using MGS in place of ICGS 
 * Takes arguements of: 
 *      np = (From MPI) number of submatrices
 *      n = global matrix dimension (n x n matrix) 
 *      f = number of tolerable faults
 *
 * Iain Weissburg 2023
 */

#define MKL_INT int
#define MKL_DOUBLE double
#define cblas_d double

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <mkl.h>
#include <time.h>
#include <stdio.h>

#define MASTER 0    /* taskid of first task */
#define DIST_Q 1    /* code for mpi send/recv */
#define COMP_Q 2    /* code for mpi send/recv */
#define COMP_R 3    /* code for mpi send/recv */

#define DEBUG 0     /* run in debug mode */
#define SET_SEED 1  /* whether to set srand to 0 */

FILE *fp_log;
VSLStreamStatePtr stream;

void printMatrix(double* matrix, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(fp_log,"%.3f ", matrix[i*cols + j]);
        }
        fprintf(fp_log,"\n");
    }
    fprintf(fp_log,"\n");
}

/* Fill Matrix with standard normal randoms */
void randMatrix(double* A, int n, int m) {
    vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n*m, A, 0, 1 );
}

/* sets B = QR and returns 1-norm of A - B */
double checkError(double* A, double* Q, double* R, double* B, int glob_cols, int glob_rows) {
    LAPACKE_dlacpy(CblasRowMajor,'A', glob_rows, glob_cols, Q, glob_cols, B, glob_cols);
    cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    glob_rows, glob_cols, 1, R, glob_cols, B, glob_cols);
    cblas_daxpy(glob_cols*glob_rows, -1, A, 1, B, 1);
    return cblas_dnrm2(glob_cols*glob_rows, B, 1);
}

void scatterA(double* A, double* Q, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows) {
    
    int i, j, k, l;
    int glob_cols = loc_cols * proc_cols;

    /* Master distributes A across process Q matrices */
    if (p_rank == MASTER) {
        /* NOTE: after all operations, Q will be 
            left as local Q for master process */

        /* i = process col */
        for (i = proc_cols - 1; i >= 0; --i) {
            /* j = process row*/
            for (j = proc_rows - 1; j >= 0; --j) {

                int r_off = loc_rows * j;
                int c_off = loc_cols * i;
                for (k = 0; k < loc_cols; ++k) {
                    for (l = 0; l < loc_rows; ++l) {
                        Q[l*loc_cols + k] = 
                            A[ (r_off + l) * glob_cols + (c_off + k) ];
                    }
                }

                /* If target not master, send Q */
                int target = j * proc_cols + i;
                if (target != p_rank)
                    MPI_Send(Q, loc_cols * loc_rows, MPI_DOUBLE,
                        target, DIST_Q, MPI_COMM_WORLD);
            }
        }
    }

    /* If not master recieve Q */
    if (p_rank > MASTER) {
        MPI_Recv(Q, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, DIST_Q, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
    }
    
    if(DEBUG) {
        int p_col = p_rank % proc_cols;
        int p_row = p_rank / proc_cols;
        fprintf(fp_log,"Q_initial (%d,%d)\n", p_col, p_row);
        printMatrix(Q, loc_cols, loc_rows);
    }
}

void gatherQR(double** Q, double** R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows) {

    int i, j, k, l;
    int glob_cols = loc_cols * proc_cols;
    int glob_rows = loc_rows * proc_rows;

    /* Master compiles Q and R from across processes */
    if (p_rank == MASTER) {
        double *Q_global = (double*) malloc(glob_cols * glob_rows * sizeof(double));
        double *R_global = (double*) malloc(glob_cols * glob_rows * sizeof(double));
        for (i = 0; i < proc_cols; ++i) {
            for (j = 0; j < proc_rows; ++j) {

                /* If target not master, recieve Q and R */
                int target = j * proc_cols + i;
                if (target != MASTER) {
                    MPI_Recv(*Q, loc_cols * loc_rows, MPI_DOUBLE,
                        target, COMP_Q, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(*R, loc_cols * loc_rows, MPI_DOUBLE,
                        target, COMP_R, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                int r_off = loc_rows * j;
                int c_off = loc_cols * i;
                for (k = 0; k < loc_cols; ++k) {
                    for (l = 0; l < loc_rows; ++l) {
                        Q_global[ (r_off + l) * glob_cols + (c_off + k) ] 
                            = (*Q)[l*loc_cols + k];
                        R_global[ (r_off + l) * glob_cols + (c_off + k) ] 
                            = (*R)[l*loc_cols + k];
                    }
                }
            }
        }
        free(*Q);
        free(*R);
        *(Q) = Q_global;
        *(R) = R_global;
    }

    /* If not master send Q and R*/
    if (p_rank > MASTER) {
        MPI_Send(*Q, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, COMP_Q, MPI_COMM_WORLD);
        MPI_Send(*R, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, COMP_R, MPI_COMM_WORLD);
    }  
}

/* Actually Gv for Q-Factor protection, Construction 1 */
void constructGv(double* Gv, int proc_rows, int f) {
    int i, j, k;

    int p_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);  

    double* V = calloc(((proc_rows - f) - f) * f, sizeof(double));
    randMatrix(V, (proc_rows - f) - f, f);
    
    /* G_pre = [-1/2*VV^T V]*/
    for (i = 0; i < f; ++i) {

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, f, f, proc_rows - 2*f, 
            -0.5, V, proc_rows - 2*f, V, proc_rows - 2*f, 0, Gv, proc_rows - f);
        
        for (j = f; j < (proc_rows - f); ++j) {
            Gv[i*(proc_rows - f) + j] = V[i*((proc_rows - f) - f) + j - f];
        }
    }
    if (DEBUG) {
        fprintf(fp_log, "V:\n");
        printMatrix(V, (proc_rows - f) - f, f);
        fprintf(fp_log, "G_pre:\n");
        printMatrix(Gv, (proc_rows - f), f);
    }

    free(V);
}

/* Actually Gh for R-factor protection, random */
void constructGh(double* Gh, int proc_cols, int f) {
    randMatrix(Gh, proc_cols-f, f);
}

void genFail(double* Q, int target_row, int p_row, int loc_cols, int loc_rows) {
    int dim = loc_cols * loc_rows;
    if (target_row == p_row)
        for (int i=0; i < dim; ++i) {
            Q[i] = 0;
        }
}

void reconstructQ(double* Q, double* Gv_tilde, int target_row, int p_rank, int proc_cols, int proc_rows, int max_fails,
                    int loc_cols, int loc_rows) {

    int i;
    MPI_Comm col_comm;
    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;

    MPI_Comm_split(MPI_COMM_WORLD, p_col, p_rank, &col_comm);

    double* part_recon = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    
    if (target_row >= proc_rows - max_fails) { //if target node is checksum
        if (p_row != target_row) { //if not target
            for (i=0; i < loc_cols * loc_rows; ++i) {
                part_recon[i] = Gv_tilde[(target_row - proc_rows + max_fails) * (proc_rows - max_fails) + p_row] * Q[i];
            }
            MPI_Reduce(part_recon, NULL, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, target_row, col_comm);
        }
        else { //reduce checksums to target node
            MPI_Reduce(part_recon, Q, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, target_row, col_comm);
        }
    }

    else { //if target node is regular node
        //TODO: support multiple failiures, maybe use a send-recieve scheme instead of reduce
        int checksum_row = 0;

        if (p_row != target_row) { //if not target
            if (p_row < proc_rows - max_fails) { //if regular node
                for (i=0; i < loc_cols * loc_rows; ++i) {
                    part_recon[i] = -1 * Gv_tilde[checksum_row * (proc_rows - max_fails) + p_row] * Q[i];
                }
            }
            else if (p_row == checksum_row + proc_rows - max_fails) { //if selected checksum
                for (i=0; i < loc_cols * loc_rows; ++i) {
                    part_recon[i] = Q[i];
                }
            }
            MPI_Reduce(part_recon, NULL, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, target_row, col_comm);
        }
        else { //reduce to target node
            MPI_Reduce(part_recon, Q, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, target_row, col_comm);
            
            /* divide by correct G term */
            for (i=0; i < loc_cols * loc_rows; ++i) { 
                Q[i] *= 1 / Gv_tilde[checksum_row * (proc_rows - max_fails) + p_row];
            }
        }
    }
}

void pbmgs(double* Q, double* R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows) {

    int APC,                        /* active process column */
        i, j, k;                    /* iterators */
    double  Qnorm, Qdot,            /* operation variables */
            Qnorm_loc, Qdot_loc;    /* operation local matrices */

    MPI_Comm row_comm, col_comm;
    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;

    MPI_Comm_split(MPI_COMM_WORLD, p_col, p_rank, &col_comm);
    MPI_Comm_split(MPI_COMM_WORLD, p_row, p_rank, &row_comm);

    double* Qbar = (double*) malloc(loc_cols * loc_rows * sizeof(double));
    double* Rbar = (double*) malloc(loc_cols * loc_rows * sizeof(double));

    /* For each block */
    for (APC = 0; APC < proc_cols; ++APC) {
        i = APC * loc_cols;

        /* ICGS Step */
        if (p_col == APC) {
            for (j = 0; j < loc_cols; ++j) {
                Qnorm = 0;

                Qnorm_loc = cblas_ddot(loc_rows, Q + j, loc_cols, Q + j, loc_cols);
            
                /* Allreduce to find squared sum of Qbar */
                MPI_Allreduce(&Qnorm_loc, &Qnorm, 1, MPI_DOUBLE,
                    MPI_SUM, col_comm);

                /* Set zero values to near-zero */
                Qnorm = sqrt(Qnorm);

                /* Normalize local portions of Qbar */
                cblas_dscal(loc_rows, 1 / Qnorm, Q + j, loc_cols);

                /* Set R to Qnorm in the correct row in the correct node */                    
                if (p_row == (i + j) / loc_rows) {
                    if(DEBUG) fprintf(fp_log,"Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qnorm, (i + j) % loc_rows, k);
                    R[((i + j) % loc_rows)*loc_cols + j] = Qnorm;
                }     

                // For upper triangular R[j, j+1:n]
                for (k = j+1; k < loc_cols; ++k) {

                    Qdot = 0;

                    /* Perform local parts of Qbar * Q[:,k]*/
                    Qdot_loc = cblas_ddot(loc_rows, Q + k, loc_cols, Q + j, loc_cols);

                    /* Reduce to form Qdot = Qbar * Q[:,k] */
                    MPI_Allreduce(&Qdot_loc, &Qdot, 1, 
                        MPI_DOUBLE, MPI_SUM, col_comm);

                    if(DEBUG) {
                        fprintf(fp_log,"Qdot_%d: %.3f\n\n", p_col*loc_cols + k, Qdot);
                    }

                    // Q[:,k] = Q[:,k] - Qdot * Qbar
                    cblas_daxpy(loc_rows, -1 * Qdot, Q + j, loc_cols, Q + k, loc_cols);


                    /* Set R to Qdot in the correct row in the correct node */                    
                    if (p_row == (i + j) / loc_rows) {
                        if(DEBUG) fprintf(fp_log,"Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qdot, (i + j) % loc_rows, k);
                        R[((i + j) % loc_rows)*loc_cols + k] = Qdot;
                    }
                }
            }

            if(DEBUG) {
                fprintf(fp_log,"Q_reduced (%d,%d)\n", p_col, p_row);
                printMatrix(Q, loc_cols, loc_rows);
            }

            /* Copy Q into Qbar for broadcast */
            LAPACKE_dlacpy(CblasRowMajor, 'A', loc_cols, loc_rows, Q, loc_cols, Qbar, loc_cols);
        }

        if(DEBUG && p_col == APC) {
            fprintf(fp_log,"Qbar_broadcast (%d,%d)\n", p_col, p_row);
            printMatrix(Qbar, loc_cols, loc_rows);
        }

        MPI_Bcast(Qbar, loc_cols * loc_rows, MPI_DOUBLE,
            APC, row_comm);

        /* In remaining blocks */
        if (p_col > APC) {
            /* Local matrix multiply R = Qbar^T * Q */
            if(DEBUG) {
                fprintf(fp_log,"Block (%d,%d) is being transformed at APC=%d\n\n", p_col, p_row, APC);
            }

            /* j = cols of Q in local block */
            for (j = 0; j < loc_cols; ++j) {

                /* k = cols of Qbar */
                for (k = 0; k < loc_cols; ++k) {

                    /* R[j,k] = Q[:,j] * Qbar[:,k]*/
                    Qdot_loc = cblas_ddot(loc_rows, Q + j, loc_cols, Qbar + k, loc_cols);

                    /* Column wise All-reduce*/
                    MPI_Allreduce(&Qdot_loc, &Qdot, 1, 
                        MPI_DOUBLE, MPI_SUM, col_comm);
                    
                    Rbar[k*loc_cols + j] = Qdot;
                }

                if(DEBUG) {
                    fprintf(fp_log,"R_summed (%d,%d) j=%d\n", p_col, p_row, j);
                    printMatrix(R, loc_cols, loc_cols);
                }

                /* k = cols of Qbar */
                for (k = 0; k < loc_cols; ++k) {

                    /*  Q[:,j] reduced by Q[:,i] * R[i,j] */
                    cblas_daxpy(loc_rows, -1 * Rbar[k*loc_cols + j], Qbar + k, loc_cols, Q + j, loc_cols);

                    /* Set R in the correct row in the correct node */                    
                    if (p_row == (i + k) / loc_rows) {
                        R[((i + k) % loc_rows)*loc_cols + j] 
                            = Rbar[k*loc_cols + j];
                    }
                }

                if(DEBUG) {
                    fprintf(fp_log,"Q_reduced2 (%d,%d) j=%d\n", p_col, p_row, j);
                    printMatrix(R, loc_cols, loc_cols);
                }

            }
        }
    }
}

int main(int argc, char** argv) {
    
    int	proc_size,              /* number of tasks in partition */
        p_rank,                 /* a task identifier */ 
        glob_cols, glob_rows,   /* global matrix dimensions */
        proc_cols, proc_rows,   /* processor grid dimensions */
        loc_cols, loc_rows,     /* local block dimensions */
        max_fails,              /* maximum tolerable failiures (f in literature) */
        check_cols, check_rows; /* checksum rows/columns */
    double  *A, *Q, *R,         /* main i/o matrices */
            *Gv_tilde,          /* Q Factor generator matrix */
            *Gh_tilde;          /* R Factor generator matrix */

    double t1, t2;              /* timer */

    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    char fname[99];
    sprintf(fname, "log_%d.txt", MPI_Wtime());
    fp_log = fopen(fname, "a");

    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);  

    proc_cols = 4;
    proc_rows = proc_size / 4;
    max_fails = proc_rows - proc_cols;

    glob_cols = glob_rows = atoi(argv[1]);
    loc_cols = loc_rows = atoi(argv[2]);
    check_cols = 0;
    check_rows = loc_rows * max_fails;

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        A = (double*) calloc(glob_cols * (glob_rows + check_rows), sizeof(double));
        fprintf(fp_log,"codedqr_on_sq has started with %d tasks in %d rows and %d columns\n", proc_size, proc_rows, proc_cols);
        fprintf(fp_log,"Each process has %d rows and %d columns\n\n", loc_rows, loc_cols);

        /* Generate random matrix */
        if(SET_SEED) vslNewStream(&stream, VSL_BRNG_SFMT19937, 1);
        else vslNewStream(&stream, VSL_BRNG_SFMT19937, MPI_Wtime());
        randMatrix(A, glob_cols, glob_rows);
        if(DEBUG) {
            fprintf(fp_log,"Initializing array A: \n");
            printMatrix(A, glob_cols, glob_rows);
        }

    }
    /* TODO: 
     *  - Generate V and broadcast
     *  - Each checksum node uses one row of G_pre times column of A nodes
     *  - calculate grid -> distribute A -> calculate checksums.
     *  - extract Q and R
     *  - Test failiure and reconstruction
     */

    /************* Distribute A across process Q *********************/

    Q = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    R = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    
    scatterA(A, Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    
    /* Start timer*/
    t1 = MPI_Wtime();

    /***************** Q-Factor Checksums ****************************/
    
    int i, j;
    Gv_tilde = (double*) calloc((proc_rows - max_fails) * max_fails, sizeof(double));

    /* Construct Gv in master node */
    if (p_rank == MASTER)
        constructGv(Gv_tilde, proc_rows, max_fails);

    /* Broadcast Gv from master node */
    MPI_Bcast(Gv_tilde, (proc_rows - max_fails) * max_fails, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Comm col_comm;
    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;
    int cs_row = p_row - (proc_rows - max_fails);
    int cs_col = p_col - proc_cols;// + max_fails;

    MPI_Comm_split(MPI_COMM_WORLD, p_col, p_rank, &col_comm);

    double* part_checksum = (double*) calloc(loc_cols * loc_rows, sizeof(double));

    if (cs_row < 0) { //if non-checksum node
        for (j=0; j < max_fails; ++j) {
            for (i=0; i < loc_cols * loc_rows; ++i) {
                part_checksum[i] = Gv_tilde[j * (proc_rows - max_fails) + p_row] * Q[i];
            }
            MPI_Reduce(part_checksum, NULL, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, j + proc_rows - max_fails, col_comm);
        }
    }
    else { //reduce checksums to checksum nodes
        for (j=0; j < max_fails; ++j) {
            MPI_Reduce(part_checksum, Q, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, j + proc_rows - max_fails, col_comm);
            if (DEBUG && j==cs_row) {
                fprintf(fp_log, "Checkum node %d Q is now:\n", p_rank);
                printMatrix(Q, loc_cols, loc_rows);
            }
        }
    }

    /***************** R-Factor Checksums ****************************/

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, p_row, p_rank, &row_comm);

    /******************** Test Reconstruction ************************/

    int failed_row = 4;
    genFail(Q, failed_row, p_row, loc_cols, loc_rows);
    reconstructQ(Q, Gv_tilde, failed_row, p_rank, proc_cols, proc_rows, max_fails, loc_cols, loc_rows);
    
    /************************ PBMGS **********************************/
    
    pbmgs(Q, R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /***************** Q-Factor Checksums ****************************/

    /* End timer */
    t2 = MPI_Wtime() - t1;

    /*********** Compile Q and R from local blocks *******************/

    gatherQR(&Q, &R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /********************* Check Results *****************************/

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t2, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    t2 = 1000 * exec_time / proc_size;
    }

    if (p_rank == MASTER) {

        if (glob_cols < 100 && glob_rows < 100) {
            fprintf(fp_log,"\n");
            fprintf(fp_log,"Matrix A:\n");
            printMatrix(A, glob_cols, glob_rows);

            fprintf(fp_log,"Matrix Q1:\n");
            printMatrix(Q, glob_cols, glob_rows);

            fprintf(fp_log,"Matrix Q2:\n");
            printMatrix(Q + (glob_cols * glob_rows), glob_cols, check_rows);

            fprintf(fp_log,"Matrix R:\n");
            printMatrix(R, glob_cols, glob_rows);
        }
    
        //Check error = A - QR (should be near 0)
        if (glob_cols < 1000 && glob_rows < 1000) {
            double* B = malloc(glob_cols * glob_rows * sizeof(double));
            double sum = checkError(A, Q, R, B, glob_cols, glob_rows);          
            if (sum > 0 && glob_cols < 100 && glob_rows < 100) {
                fprintf(fp_log,"Matrix B:\n");
                printMatrix(B, glob_cols, glob_rows);
            }
            fprintf(fp_log,"Roundoff Error: %f\n", sum); 
        }
        fprintf(fp_log,"Execution Time: %.3f ms\n", t2);
    }

    if(p_rank == MASTER) fprintf(fp_log,"\n\n");
    fclose(fp_log);
    MPI_Finalize();
}