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
        A[i] = (rand() % 10000 - 5000) / 100.0;
    }
}

#define MASTER 0                /* taskid of first task */
#define DIST_Q 1
#define COMP_Q 2
#define COMP_R 3

#define DEBUG 1

int main(int argc, char** argv) {
    
    int	numtasks,               /* number of tasks in partition */
        taskid,                 /* a task identifier */
        i, j, k, l,             /* misc */
        globalcols, globalrows, /* matrix dimensions */
        blocksize,              /* width of each local block */
        PN, PM,                 /* processor grid dimensions */
        pcol, prow,             /* processor coordinates in grid (aka color) */
        localcols, localrows,   /* local block dimensions */
        colid, rowid,           /* process rank in col_ and row_comm*/
        APC,                    /* active process column */
        r_off, c_off;           /* offsets between block and global matrices */
    double *A, *Q, *R,          /* main matrices */
        *Qnorm, *Qdot, *Qbar,   /* operation matrices */
        *Qnorm_loc, *Qdot_loc;  /* operation local matrices */
    MPI_Status status;
    MPI_Comm row_comm, col_comm;

    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // if (argc != 4) {
    //     if (taskid == MASTER) printf ("Invalid Input, must have arguements: m=rows n=cols b=blocksize \n");
    //     MPI_Abort(MPI_COMM_WORLD, 4);
    // }
    // if (globalcols > globalrows) {
    //     if (taskid == MASTER) printf ("Invalid Input, n cannot be greater than m\n");
    //     MPI_Abort(MPI_COMM_WORLD, 2);    }
    // if (blocksize > globalrows) {
    //     if (taskid == MASTER) printf ("Invalid Input, b cannot be greater than m\n");
    //     MPI_Abort(MPI_COMM_WORLD, 3);
    // }

    globalcols = 4;//atoi(argv[1]);
    globalrows = 4;//atoi(argv[2]);
    blocksize = 2;//atoi(argv[3]);

    PN = globalcols / blocksize;
    PM = numtasks / PN;
    pcol = taskid % PN;
    prow = taskid / PN;

    MPI_Comm_split(MPI_COMM_WORLD, pcol, taskid, &col_comm);
    MPI_Comm_split(MPI_COMM_WORLD, prow, taskid, &row_comm);

    /* NOTE: colid should equal prow, and vice versa */
    MPI_Comm_rank(col_comm, &colid);
    MPI_Comm_rank(row_comm, &rowid);


    /******************* Initialize arrays ***************************/

    if (taskid == MASTER)
    {
        A = (double*) calloc(globalrows * globalcols, sizeof(double));
        printf("pbmgs_mpi has started with %d tasks in %d rows and %d columns.\n", numtasks, PM, PN);
        printf("Initializing arrays...\n");

        /* Generate random matrix */
        srand(0);
        randMatrix(A, globalcols, globalrows);
    }

    /************* Distribute Q across processes *********************/
    
    /* NOTE: localcols = globalcols / PN */
    localcols = blocksize; 
    localrows = globalrows / PM;

    Q = (double*) calloc(localcols * localrows, sizeof(double));
    R = (double*) calloc(localcols * localcols, sizeof(double));
    
    /* Master distributes Q across processes */
    if (taskid == MASTER) {
        /* NOTE: after all operations, Q will be 
            left as local Q for master process */
        for (i = PN - 1; i >= 0; --i) {
            for (j = PM - 1; j >= 0; --j) {

                r_off = localrows * j;
                c_off = localcols * i;
                for (k = 0; k < localcols; ++k) {
                    for (l = 0; l < localrows; ++l) {
                        Q[l*localcols + k] = 
                            A[ (r_off + l) * globalcols + (c_off + k) ];
                    }
                }

                /* If target not master, send Q */
                if (i != 0 && j != 0)
                    MPI_Send(Q, localcols * localrows, MPI_DOUBLE,
                        j * PN + i, DIST_Q, MPI_COMM_WORLD);
            }
        }
    }

    /* If not master recieve Q */
    else {
        MPI_Recv(Q, localcols * localrows, MPI_DOUBLE,
            MASTER, DIST_Q, MPI_COMM_WORLD, &status);
    }

    /************************ PBMGS **********************************/

    Qbar = (double*) calloc(localcols * localrows, sizeof(double));

    /* For each block */
    for (i = 0; i < globalcols; i += blocksize) {

        APC = (i / globalcols) % PN;

        /* ICGS Step */
        if (pcol == APC) {

            Qnorm_loc = (double*) calloc(localcols, sizeof(double));
            Qdot_loc = (double*) calloc(localcols, sizeof(double));
            
            Qnorm = (double*) calloc(localcols, sizeof(double));
            Qdot = (double*) calloc(localcols, sizeof(double));

            /* Find squared sum of local portions of Qbar columns */
            for (j = 0; j < localcols; ++j) {
                for (k = 0; k < localrows; ++k) {
                    Qnorm_loc[j] += Q[k*localcols + j] * Q[k*localcols + j];
                }
            }

            /* Allreduce to find squared sums of Qbar columns */
            MPI_Allreduce(Qnorm_loc, Qnorm, localcols, MPI_DOUBLE,
                MPI_SUM, col_comm);

            /* Finish forming vector norms */
            for (j = 0; j < localcols; ++j) {
                /* Set zero values to near-zero */
                if (Qnorm[j] == 0) 
                    R[j*localcols + j] = 2.3e-308;
                else 
                    Qnorm[j] = sqrt(Qnorm[j]);

                /* Normalize local portions of Qbar */
                for (k = 0; k < localrows; ++k)
                    Q[k*localcols + j] /= Qnorm[j];
            }
            
            /* for each Qbar = Q[:,j] */
            for (j = 0; j < localcols; ++j) {
                
                // For upper triangular R[j, j+1:n]
                for (l = j+1; l < localcols; ++l) {

                    /* Perform local parts of Qbar * Q[:,l]*/ 
                    Qdot_loc[l] = 0;
                    Qdot[l] = 0;
                    for (k = 0; k < localrows; ++k) {
                        Qdot_loc[l] += Q[k*localcols + l] * Q[k*localcols + j];
                    }
                }

                /* Reduce to form Qdot = Qbar * Q[:,l] */
                MPI_Allreduce(Qdot_loc + (j+1), Qdot + (j+1), localcols - (j+1), 
                    MPI_DOUBLE, MPI_SUM, col_comm);
                
                for (l = j+1; l < localcols; ++l) {

                    // Q[:,l] = Q[:,l] - Qdot * Qbar
                    for (k = 0; k < localcols; ++k) {
                        Q[k*localcols + l] -= Qdot[l] * Q[k*localcols + j];
                    }
                }

                /* Copy Qnorm and Qdot into R for 
                    diagonal block (prow == pcol) */
                if (prow == pcol) {
                    R[j*localcols + j] = Qnorm[j];
                    for (l = j+1; l < localcols; ++l) {
                        R[j*localcols + l] = Qdot[l];
                    }
                }
            }

            free(Qdot_loc);
            free(Qnorm_loc);
            free(Qdot);
            free(Qnorm);
            free(Qbar);
            Qbar = Q;
        }

        MPI_Bcast(Qbar, localcols * localrows, MPI_DOUBLE,
            APC, row_comm);
        
        /* In remaining blocks */
        if (pcol > APC) {
            /* Local matrix multiply R = Qbar^T * Q */

            /* j = cols of Q in local block */
            for (j = 0; j < localcols; ++j) {
                /* k = cols of Qbar */
                for (k = 0; k < localcols; ++k) {

                    /* R[j,k] = Q[:,j] * Qbar[:,k]*/
                    R[j*localcols + k] = 0;
                    /* l = all rows in block */
                    for (l = 0; l < localrows; ++l) {
                        R[k*localcols + l] += Q[l*localcols + j] * Qbar[l*localcols + k];
                    }

                    /*  Q[:,j] reduced by Q[:,i] * R[i,j] */
                    for (l = 0; l < localrows; ++l) {
                        Q[l*localcols + j] -= R[k*localcols + j] * Q[l*localcols + k];
                    }
                }
            }
        }
    }

    /*********** Compile Q and R from local blocks *******************/

    double *Q_global = (double*) calloc(globalcols * globalrows, sizeof(double));
    double *R_global = (double*) calloc(globalcols * globalcols, sizeof(double));

    /* Master compiles Q and R from across processes */
    if (taskid == MASTER) {
        for (i = 0; i < PN; ++i) {
            for (j = 0; j < PM; ++j) {

                /* If target not master, recieve Q and R */
                if (i != 0 && j != 0) {
                    MPI_Recv(Q, localcols * localrows, MPI_DOUBLE,
                        j * PN + i, COMP_Q, MPI_COMM_WORLD, &status);
                    MPI_Recv(R, localcols * localcols, MPI_DOUBLE,
                        j * PN + i, COMP_R, MPI_COMM_WORLD, &status);
                }

                r_off = localrows * j;
                int R_r_off = localcols * j;
                c_off = localcols * i;
                for (k = 0; k < localcols; ++k) {
                    for (l = 0; l < localrows; ++l) {
                        Q_global[ (r_off + l) * globalcols + (c_off + k) ] 
                            = Q[l*localcols + k];
                    }
                    for (l = 0; l < localcols; ++l) {
                        R_global[ (R_r_off + l) * globalcols + (c_off + k) ] 
                            = R[l*localcols + k];
                    }
                }
            }
        }
        free(Q);
        free(R);
        Q = Q_global;
        R = R_global;
    }

    /* If not master send Q and R*/
    else {
        MPI_Send(Q, localcols * localrows, MPI_DOUBLE,
            MASTER, COMP_Q, MPI_COMM_WORLD);
        MPI_Send(R, localcols * localcols, MPI_DOUBLE,
            MASTER, COMP_R, MPI_COMM_WORLD);

        free(Q);
        free(R);
    }

    MPI_Finalize();

    
    // Check error = A - QR (should be near 0)
    double* B = calloc(globalcols * globalrows, sizeof(double));
    double sum = 0;
    for (i = 0; i < globalrows; i++) {
        for (j = 0; j < globalcols; j++) {
            B[i*globalcols + j] = 0;
            for (k = 0; k < globalcols; k++) {
                B[i*globalcols + j] += Q[i*globalcols + k] * R[k*globalcols + j];
            }

            sum += fabs(B[i*globalcols+j] - A[i*globalcols+j]);
        }
    }
    free(B);
    printf("Roundoff Error: %f\n\n", sum);

    if (globalcols <= 10) {
        printf("Matrix A:\n");
        printMatrix(A, globalcols, globalrows);

        printf("Matrix Q:\n");
        printMatrix(Q, globalcols, globalrows);

        printf("Matrix R:\n");
        printMatrix(R, globalcols, globalcols);
    }

    free(A);
    free(Q);
    free(R);

    return 0;
}