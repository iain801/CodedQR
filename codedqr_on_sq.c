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

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <stdio.h>

#define MASTER 0    /* taskid of first task */
#define DIST_Q 1    /* code for mpi send/recv */
#define COMP_Q 2    /* code for mpi send/recv */
#define COMP_R 3    /* code for mpi send/recv */

#define DEBUG 0     /* run in debug mode */
#define SET_SEED 1  /* whether to set srand to 0 */

FILE *fp_log;

void printMatrix(double* matrix, int n, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(fp_log,"%.3f ", matrix[i*n + j]);
        }
        fprintf(fp_log,"\n");
    }
    fprintf(fp_log,"\n");
}

/* uniform random in range [rangeLow, rangeHigh] */
double uniform(int rangeLow, int rangeHigh) {
    return ((double)rand()/(double)RAND_MAX) * (rangeHigh - rangeLow) + rangeLow;
}

void randMatrix(double* A, int n, int m) {
    int size = n * m;
    for (int i = 0; i < size; ++i) {
        A[i] = uniform(0,1);
    }
}

/* sets C = AB 
    a_cols must equal b_rows 
    C will be b_cols * a_rows */
void matrixMultiply(double* A, double* B, double* C, int a_cols, int a_rows, int b_cols, int b_rows) {
    int i, j, k;
    for (i = 0; i < a_rows; ++i) {
        for (j = 0; j < b_cols; ++j) {
            C[i*b_cols + j] = 0;
            for (k = 0; k < a_cols; ++k) {
                C[i*b_cols + j] += A[i*a_cols + k] * B[k*b_cols + j];
            }
        }
    }
}

/* sets B = QR and returns 1-norm of A - B */
double checkError(double* A, double* Q, double* R, double* B, int glob_cols, int glob_rows) {
    double sum = 0;
    matrixMultiply(Q, R, B, glob_cols, glob_rows, glob_cols, glob_rows);

    for (int i = 0; i < glob_rows * glob_cols; ++i) {
        sum += fabs(B[i] - A[i]);
    }

    return sum;
}

/* Actually Gv for Q-Factor protection, Construction 1 */
void constructGv(double* Gv, int proc_cols, int f, int loc_cols) {
    int i, j, k;

    double* V = calloc((proc_cols - f) * f, sizeof(double));
    randMatrix(V, proc_cols - f, f);
    
    /* G_pre = [-1/2*VV^T V]*/
    double* G_pre = calloc(proc_cols * f, sizeof(double));
    for (i = 0; i < f; ++i) {
        for (j = 0; j < f; ++j) {
            for (k = 0; k < (proc_cols - f); ++k) {
                G_pre[i*proc_cols + j] += -0.5 * V[i*f + k] * V[j*f + k];
            }
        }
        for (j = f; j < proc_cols; ++j) {
            G_pre[i*proc_cols + j] = V[i*(proc_cols - f) + j - f];
        }
    }

    free(V);

    int glob_cols = loc_cols * proc_cols;
    int r_off, c_off;
    for (i = 0; i < f; ++i) {
        r_off = i * loc_cols;
        for (j = 0; j < proc_cols; ++j) {
            c_off = j * loc_cols;
            for (k = 0; k < loc_cols; ++k) {
                Gv[(r_off + k) * glob_cols + (c_off + k)] = G_pre[i*proc_cols + j];
            }
        }
    }
}

/* Actually Gh for R-factor protection, random */
void constructGh(double* Gh, int proc_rows, int f, int loc_rows) {
    int i, j, k;
    double* G_pre = malloc(f * proc_rows * sizeof(double));
    randMatrix(G_pre, proc_rows, f);
    
    int glob_cols = loc_rows * f;
    int r_off, c_off;
    for (i = 0; i < proc_rows; ++i) {
        r_off = i * loc_rows;
        for (j = 0; j < f; ++j) {
            c_off = j * loc_rows;
            for (k = 0; k < loc_rows; ++k) {
                Gh[(r_off + k) * glob_cols + (c_off + k)] = G_pre[i*proc_rows + j];
            }
        }
    }
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

void pbmgs(double* Q, double* R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows) {

    int APC,                        /* active process column */
        i, j, k, l;                 /* iterators */
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
                Qnorm_loc = 0;
                Qnorm = 0;

                for (k = 0; k < loc_rows; ++k) {
                    Qnorm_loc += Q[k*loc_cols + j] * Q[k*loc_cols + j];
                }
            
                /* Allreduce to find squared sum of Qbar */
                MPI_Allreduce(&Qnorm_loc, &Qnorm, 1, MPI_DOUBLE,
                    MPI_SUM, col_comm);

                /* Set zero values to near-zero */
                Qnorm = sqrt(Qnorm);

                /* Normalize local portions of Qbar */
                for (k = 0; k < loc_rows; ++k)
                    Q[k*loc_cols + j] /= Qnorm;

                /* Set R to Qnorm in the correct row in the correct node */                    
                if (p_row == (i + j) / loc_rows) {
                    if(DEBUG) fprintf(fp_log,"Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qnorm, (i + j) % loc_rows, l);
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
                        fprintf(fp_log,"Qdot_%d: %.3f\n\n", p_col*loc_cols + l, Qdot);
                    }

                    // Q[:,l] = Q[:,l] - Qdot * Qbar
                    for (k = 0; k < loc_rows; ++k) {
                        Q[k*loc_cols + l] -= Qdot * Q[k*loc_cols + j];
                    }

                    /* Set R to Qdot in the correct row in the correct node */                    
                    if (p_row == (i + j) / loc_rows) {
                        if(DEBUG) fprintf(fp_log,"Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qdot, (i + j) % loc_rows, l);
                        R[((i + j) % loc_rows)*loc_cols + l] = Qdot;
                    }
                }
            }

            if(DEBUG) {
                fprintf(fp_log,"Q_reduced (%d,%d)\n", p_col, p_row);
                printMatrix(Q, loc_cols, loc_rows);
            }

            /* Copy Q into Qbar for broadcast */
            for (j = 0; j < loc_cols * loc_rows; ++j)
                Qbar[j] = Q[j];
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
                    fprintf(fp_log,"R_summed (%d,%d) j=%d\n", p_col, p_row, j);
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
            *Gv, *Gh;           /* checksum generator matrices */

    double t1, t2;              /* timer */

    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    fp_log = fopen("log.txt", "a");
    
    if (argc != 3) {
        if (p_rank == MASTER) fprintf(fp_log, "Invalid Input, must have arguements: n, f\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    glob_cols = glob_rows = atoi(argv[1]);
    max_fails = atoi(argv[2]);

    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);    

    proc_cols = proc_rows = (int) sqrt(proc_size);

    if (proc_size != proc_cols * proc_rows) {
        if (p_rank == MASTER) fprintf(fp_log, "Invalid Input, np must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    
    loc_cols = glob_cols / (proc_cols);
    loc_rows = glob_rows / (proc_rows - max_fails); // To accomidate checksum additions
    check_cols = loc_cols * max_fails;
    check_rows = loc_rows * max_fails;

    if (!loc_cols) {
        if (p_rank == MASTER) fprintf(fp_log, "Invalid Input, n^2 must be greater than np\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    else if (glob_rows != loc_rows * (proc_rows - max_fails)) {
        if (p_rank == MASTER) fprintf(fp_log, "Invalid Input, n must be divisible by b=sqrt(np)\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        A = (double*) calloc(glob_cols * (glob_rows + check_rows), sizeof(double));
        fprintf(fp_log,"codedqr_on_sq has started with %d tasks in %d rows and %d columns\n", proc_size, proc_rows, proc_cols);
        fprintf(fp_log,"Each process has %d rows and %d columns\n\n", loc_rows, loc_cols);

        /* Generate random matrix */
        if(SET_SEED) srand(0);
        else srand(MPI_Wtime());
        randMatrix(A, glob_cols, glob_rows);
        if(DEBUG) {
            fprintf(fp_log,"Initializing array A: \n");
            printMatrix(A, glob_cols, glob_rows);
        }

        /* Construct Gv and build checksum under A */
        Gv = (double*) calloc(glob_cols * check_rows, sizeof(double));
        constructGv(Gv, proc_cols, max_fails, loc_cols);
        matrixMultiply(Gv, A, A + (glob_cols * glob_rows), glob_cols, check_cols, glob_cols, glob_rows);
        //free(Gv);
        
        // /* Construct Gh and build AGh and GvAGh checksum in seperate matrix */
        // double* r_checksums = (double*) calloc(check_cols * (glob_rows + check_rows), sizeof(double));
        // Gh = (double*) calloc(check_cols * glob_rows, sizeof(double));
        // constructGh(Gh, proc_rows, max_fails, loc_rows);
        // matrixMultiply(A, Gh, r_checksums, glob_cols, glob_rows, check_cols, glob_rows);
        // matrixMultiply(A + (glob_cols * glob_rows), Gh, r_checksums + (check_cols * glob_rows), 
        //     glob_cols, check_rows, check_cols, glob_rows);

        // /* Append r_checksums to right of A */
        // double* A_temp = (double*) malloc((glob_cols + check_cols) * (glob_rows + check_rows) * sizeof(double));
        // for (int i = 0; i < glob_rows + check_rows; ++i) {
        //     for (int j = 0; j < glob_cols; ++j) {
        //         A_temp[i * (glob_cols + check_cols) + j] = A[i * glob_cols + j];
        //     }
        //     for (int j = 0; j < check_cols; ++j) {
        //         A_temp[i * (glob_cols + check_cols) + (glob_cols + j)] = r_checksums[i * check_cols + j];
        //     }
        // }
        
        //free(r_checksums);
        //free(A);
        //A = A_temp;
    }
    /* TODO: 
     *  - extract Q and R
     */

    /* Start timer*/
    t1 = MPI_Wtime();

    /************* Distribute A across process Q *********************/

    Q = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    R = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    
    scatterA(A, Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /************************ PBMGS **********************************/

    pbmgs(Q, R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /*********** Compile Q and R from local blocks *******************/

    gatherQR(&Q, &R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

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


        fprintf(fp_log,"\n");
        fprintf(fp_log,"Matrix A with Checksums:\n");
        printMatrix(A, glob_cols, glob_rows + check_rows);

        fprintf(fp_log,"Matrix Q with Checksums:\n");
        printMatrix(Q, glob_cols, glob_rows + check_rows);

        fprintf(fp_log,"Matrix R:\n");
        printMatrix(R, glob_cols, glob_rows + check_rows);
    
        //Check error = A - QR (should be near 0)
        if (glob_cols < 1000 && glob_rows < 1000) {
            double* B = malloc(glob_cols * glob_rows * sizeof(double));
            double sum = checkError(A, Q, R, B, glob_cols, glob_rows);          
            if (sum > 0 && glob_cols <= 25) {
                fprintf(fp_log,"Matrix B:\n");
                printMatrix(B, glob_cols, glob_rows);
            }
            fprintf(fp_log,"Roundoff Error: %f\n", sum); 
        }
        fprintf(fp_log,"Execution Time: %.3f ms\n", t2);
    }
    
    fclose(fp_log);
    MPI_Finalize();
}