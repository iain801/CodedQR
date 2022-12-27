/* Parallel Block Modified Gram Schmidt
 * implementation with MPI
 * for random rectangular matrix A(n,m) 
 * Using MGS in place of ICGS 
 * Takes arguements of: 
 *      np = (From MPI) number of submatrices
 *      n = global column count 
 *      m = global row count 
 *      b = submatrix column count
 *
 * Iain Weissburg 2022
 */

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <stdio.h>

void printMatrix(double* matrix, int n, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.3f ", matrix[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void randMatrix(double* A, int n, int m) {
    for (int i = 0; i < n * m; ++i) {
        A[i] = (rand() % 10 - 5) / 1.0;
    }
}

double checkError(double* A, double* Q, double* R, double* B, int glob_cols, int glob_rows) {
    int i, j, k;
    double sum = 0;
    for (i = 0; i < glob_rows; i++) {
        for (j = 0; j < glob_cols; j++) {
            B[i*glob_cols + j] = 0;
            for (k = 0; k < glob_cols; k++) {
                B[i*glob_cols + j] += Q[i*glob_cols + k] * R[k*glob_cols + j];
            }

            sum += fabs(B[i*glob_cols+j] - A[i*glob_cols+j]);
        }
    }

    return sum;
}

#define MASTER 0                /* taskid of first task */
#define DIST_Q 1
#define COMP_Q 2
#define COMP_R 3

#define DEBUG 0

void distributeQ(double* A, double* Q, int p_rank, 
    int proc_cols, int proc_rows, int glob_cols, int glob_rows) {
    
    int i, j, k, l;
    int loc_cols = glob_cols / proc_cols;
    int loc_rows = glob_rows / proc_rows;

    /* Master distributes Q across processes */
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
        printf("Q_initial (%d,%d)\n", p_col, p_row);
        printMatrix(Q, loc_cols, loc_rows);
    }
}

void gatherQR(double** Q, double** R, int p_rank, 
    int proc_cols, int proc_rows, int glob_cols, int glob_rows) {

    int i, j, k, l;
    int loc_cols = glob_cols / proc_cols;
    int loc_rows = glob_rows / proc_rows;

    /* Master compiles Q and R from across processes */
    if (p_rank == MASTER) {
        double *Q_global = (double*) malloc(glob_cols * glob_rows * sizeof(double));
        double *R_global = (double*) malloc(glob_cols * glob_rows * sizeof(double));
        for (i = 0; i < proc_cols; ++i) {
            for (j = 0; j < proc_rows; ++j) {

                /* If target not master, recieve Q and R */
                int target = j * proc_cols + i;
                if (target != p_rank) {
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


int main(int argc, char** argv) {
    
    int	proc_size,              /* number of tasks in partition */
        p_rank,                 /* a task identifier */
        i, j, k, l,             /* iterators */
        glob_cols, glob_rows,   /* global matrix dimensions */
        block_size,             /* width of each local block */
        proc_cols, proc_rows,   /* processor grid dimensions */
        p_col, p_row,           /* processor coordinates in grid (aka color) */
        loc_cols, loc_rows,     /* local block dimensions */
        APC;                    /* active process column */
    double *A, *Q, *R,          /* main i/o matrices */
        *Qbar, *Rbar,           /* broadcast matrix */
        Qnorm, Qdot,            /* operation variables */
        Qnorm_loc, Qdot_loc;    /* operation local matrices */

    MPI_Comm row_comm, col_comm;

    double t1, t2;              /* timer */

    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    
    if (argc != 4) {
        if (p_rank == MASTER) printf ("Invalid Input, must have arguements: m=rows n=cols b=blocksize \n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    glob_cols = atoi(argv[1]);
    glob_rows = atoi(argv[2]);
    block_size = atoi(argv[3]);

    if (glob_cols > glob_rows) {
        if (p_rank == MASTER) printf ("Invalid Input, n cannot be greater than m\n");
    }
    if (block_size > glob_rows) {
        if (p_rank == MASTER) printf ("Invalid Input, b cannot be greater than m\n");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

    proc_cols = glob_cols / block_size;
    proc_rows = proc_size / proc_cols;
    p_col = p_rank % proc_cols;
    p_row = p_rank / proc_cols;
    
    /* NOTE: loc_cols = glob_cols / proc_cols */
    loc_cols = block_size; 
    loc_rows = glob_rows / proc_rows;

    MPI_Comm_split(MPI_COMM_WORLD, p_col, p_rank, &col_comm);
    MPI_Comm_split(MPI_COMM_WORLD, p_row, p_rank, &row_comm);

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        A = (double*) malloc(glob_rows * glob_cols * sizeof(double));
        printf("pbmgs_mpi has started with %d tasks in %d rows and %d columns\n", proc_size, proc_rows, proc_cols);
        printf("Each process has %d rows and %d columns\n\n", loc_rows, loc_cols);

        /* Generate random matrix */
        srand(0);
        randMatrix(A, glob_cols, glob_rows);
        if(DEBUG) {
            printf("Initializing array A: \n");
            printMatrix(A, glob_cols, glob_rows);
        }
    }

    /* Start timer*/
    t1 = MPI_Wtime();

    /************* Distribute A across processes *********************/

    Q = (double*) malloc(loc_cols * loc_rows * sizeof(double));
    R = (double*) malloc(loc_cols * loc_rows * sizeof(double));
    
    distributeQ(A, Q, p_rank, proc_cols, proc_rows, glob_cols, glob_rows);

    /************************ PBMGS **********************************/

    Qbar = (double*) malloc(loc_cols * loc_rows * sizeof(double));
    Rbar = (double*) malloc(loc_cols * loc_rows * sizeof(double));

    /* For each block */
    for (APC = 0; APC < proc_cols; ++APC) {
        i = APC * loc_cols;

        /* ICGS Step */
        if (p_col == APC) {
            for (j = 0; j < loc_cols; ++j) {
                Qnorm_loc = 0;
                Qnorm = 0;

                for (k = 0; k < loc_rows; ++k) {
                    Qnorm_loc += Q[k*loc_cols + j] * Q[k*loc_cols + j];
                }
            
                /* Allreduce to find squared sum of Qbar */
                MPI_Allreduce(&Qnorm_loc, &Qnorm, 1, MPI_DOUBLE,
                    MPI_SUM, col_comm);

                /* Set zero values to near-zero */
                if (Qnorm == 0) 
                    Qnorm = 2.3e-308;
                else 
                    Qnorm = sqrt(Qnorm);

                /* Normalize local portions of Qbar */
                for (k = 0; k < loc_rows; ++k)
                    Q[k*loc_cols + j] /= Qnorm;

                /* Set R to Qnorm in the correct row in the correct node */                    
                if (p_row == (i + j) / loc_rows) {
                    if(DEBUG) printf("Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qnorm, (i + j) % loc_rows, l);
                    R[((i + j) % loc_rows)*loc_cols + j] = Qnorm;
                }     

                // For upper triangular R[j, j+1:n]
                for (l = j+1; l < loc_cols; ++l) {

                    Qdot_loc = 0;
                    Qdot = 0;

                    /* Perform local parts of Qbar * Q[:,l]*/
                    for (k = 0; k < loc_rows; ++k) {
                        Qdot_loc += Q[k*loc_cols + l] * Q[k*loc_cols + j];
                    }

                    /* Reduce to form Qdot = Qbar * Q[:,l] */
                    MPI_Allreduce(&Qdot_loc, &Qdot, 1, 
                        MPI_DOUBLE, MPI_SUM, col_comm);

                    if(DEBUG) {
                        printf("Qdot_%d: %.3f\n\n", p_col*block_size + l, Qdot);
                    }

                    // Q[:,l] = Q[:,l] - Qdot * Qbar
                    for (k = 0; k < loc_rows; ++k) {
                        Q[k*loc_cols + l] -= Qdot * Q[k*loc_cols + j];
                    }

                    /* Set R to Qdot in the correct row in the correct node */                    
                    if (p_row == (i + j) / loc_rows) {
                        if(DEBUG) printf("Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qdot, (i + j) % loc_rows, l);
                        R[((i + j) % loc_rows)*loc_cols + l] = Qdot;
                    }
                }
            }

            if(DEBUG) {
                printf("Q_reduced (%d,%d)\n", p_col, p_row);
                printMatrix(Q, loc_cols, loc_rows);
            }

            /* Copy Q into Qbar for broadcast */
            for (j = 0; j < loc_cols * loc_rows; ++j)
                Qbar[j] = Q[j];
        }

        if(DEBUG && p_col == APC) {
            printf("Qbar_broadcast (%d,%d)\n", p_col, p_row);
            printMatrix(Qbar, loc_cols, loc_rows);
        }

        MPI_Bcast(Qbar, loc_cols * loc_rows, MPI_DOUBLE,
            APC, row_comm);

        /* In remaining blocks */
        if (p_col > APC) {
            /* Local matrix multiply R = Qbar^T * Q */
            if(DEBUG) {
                printf("Block (%d,%d) is being transformed at APC=%d\n\n", p_col, p_row, APC);
            }

            /* j = cols of Q in local block */
            for (j = 0; j < loc_cols; ++j) {

                /* k = cols of Qbar */
                for (k = 0; k < loc_cols; ++k) {

                    /* R[j,k] = Q[:,j] * Qbar[:,k]*/
                    Qdot_loc = 0;
                    /* l = all rows in block */
                    for (l = 0; l < loc_rows; ++l) {
                        Qdot_loc += Q[l*loc_cols + j] * Qbar[l*loc_cols + k];
                    }

                    /* Column wise All-reduce*/
                    MPI_Allreduce(&Qdot_loc, &Qdot, 1, 
                        MPI_DOUBLE, MPI_SUM, col_comm);
                    
                    Rbar[k*loc_cols + j] = Qdot;
                }

                if(DEBUG) {
                    printf("R_summed (%d,%d) j=%d\n", p_col, p_row, j);
                    printMatrix(R, loc_cols, loc_cols);
                }

                /* k = cols of Qbar */
                for (k = 0; k < loc_cols; ++k) {

                    /*  Q[:,j] reduced by Q[:,i] * R[i,j] */
                    for (l = 0; l < loc_rows; ++l) {
                        Q[l*loc_cols + j] -= Rbar[k*loc_cols + j] * Qbar[l*loc_cols + k];
                    }

                    /* Set R in the correct row in the correct node */                    
                    if (p_row == (i + k) / loc_rows)
                    {
                        R[((i + k) % loc_rows)*loc_cols + j] 
                            = Rbar[k*loc_cols + j];
                    }

                }

                if(DEBUG) {
                    printf("Q_reduced2 (%d,%d) j=%d\n", p_col, p_row, j);
                    printMatrix(R, loc_cols, loc_cols);
                }

            }
        }
    }

    /*********** Compile Q and R from local blocks *******************/

    gatherQR(&Q, &R, p_rank, proc_cols, proc_rows, glob_cols, glob_rows);

    /********************* Check Results *****************************/

    /* End timer */
    t2 = MPI_Wtime() - t1;

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t2, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    if (p_rank == MASTER) t2 = 1000 * exec_time / proc_size;
    }

    if (p_rank == MASTER) {

        if (glob_rows <= 25) {
            printf("\n");
            printf("Matrix A:\n");
            printMatrix(A, glob_cols, glob_rows);

            printf("Matrix Q:\n");
            printMatrix(Q, glob_cols, glob_cols);

            printf("Matrix R:\n");
            printMatrix(R, glob_cols, glob_cols);
        }
    
        // Check error = A - QR (should be near 0)
        if (glob_cols < 1000 && glob_rows < 1000) {
            double* B = malloc(glob_cols * glob_rows * sizeof(double));
            double sum = checkError(A, Q, R, B, glob_cols, glob_rows);
            printf("Roundoff Error: %f\n", sum);            
            if (sum > 0 && glob_rows <= 25) {
                printf("Matrix B:\n");
                printMatrix(B, glob_cols, glob_rows);
            }
        }
        printf("Execution Time: %.3f ms\n", t2);
    }

    MPI_Finalize();
}