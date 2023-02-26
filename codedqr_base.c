/* Coded Parallel Block Modified Gram Schmidt
 * implementation with MPI using Off node checksum storage
 * Core functions
 * 
 * Iain Weissburg 2023
 */

#include "codedqr_base.h"

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
double checkError(double* A, double* Q, double* R, double* B, 
    int loc_cols, int loc_rows, int glob_cols, int glob_rows) {
    
    double out_norm = 0;    

    double  *Q_bar = (double*) calloc(loc_rows * glob_cols, sizeof(double)), 
            *R_bar = (double*) calloc(loc_cols * glob_rows, sizeof(double));

    MPI_Allgather(Q, loc_rows * loc_cols, MPI_DOUBLE, Q_bar, 
        loc_rows * loc_cols, MPI_DOUBLE, row_comm);
    MPI_Allgather(R, loc_rows * loc_cols, MPI_DOUBLE, R_bar, 
        loc_rows * loc_cols, MPI_DOUBLE, col_comm);

    int proc_cols = glob_cols / loc_cols;
    for (int pc =0; pc < proc_cols; ++pc) {

        int offset = pc * loc_cols * loc_rows;
        for (int i=0; i < loc_cols; ++i) {
            for (int j=0; j < loc_rows; ++j) {
                for (int k=0; k < loc_cols; ++k) {
                    B[j * loc_rows + i] += Q_bar[offset + j * loc_cols + k] * R_bar[offset + k * loc_cols + i];
                }
            }
        }

    }

    cblas_daxpy(loc_cols*loc_rows, -1, A, 1, B, 1);

    double part_norm = cblas_dnrm2(loc_cols*loc_rows, B, 1);
    
    MPI_Reduce(&part_norm, &out_norm, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);

    free(Q_bar);
    free(R_bar);

    return out_norm;
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
                        target, DIST_Q, glob_comm);
            }
        }
    }

    /* If not master recieve Q */
    if (p_rank > MASTER) {
        MPI_Recv(Q, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, DIST_Q, glob_comm, MPI_STATUS_IGNORE);
        
    }
    
    if(DEBUG) {
        int p_col = p_rank % proc_cols;
        int p_row = p_rank / proc_cols;
        fprintf(fp_log,"Q_initial (%d,%d)\n", p_col, p_row);
        printMatrix(Q, loc_cols, loc_rows);
    }
}

void gatherA(double** A,  int p_rank, int proc_cols, 
    int proc_rows, int loc_cols, int loc_rows) {

    int i, j, k, l;
    int glob_cols = loc_cols * proc_cols;
    int glob_rows = loc_rows * proc_rows;

    /* Master compiles matrix A from across processes */
    if (p_rank == MASTER) {
        double *A_glob = (double*) malloc(glob_cols * glob_rows * sizeof(double));
        for (i = 0; i < proc_cols; ++i) {
            for (j = 0; j < proc_rows; ++j) {

                /* If target not master, recieve A */
                int target = j * proc_cols + i;
                if (target != MASTER) {
                    MPI_Recv(*A, loc_cols * loc_rows, MPI_DOUBLE,
                        target, COMP_Q, glob_comm, MPI_STATUS_IGNORE);
                }

                int r_off = loc_rows * j;
                int c_off = loc_cols * i;
                for (k = 0; k < loc_cols; ++k) {
                    for (l = 0; l < loc_rows; ++l) {
                        A_glob[ (r_off + l) * glob_cols + (c_off + k) ] 
                            = (*A)[l*loc_cols + k];
                    }
                }
            }
        }
        free(*A);
        *(A) = A_glob;
    }

    /* If not master send A*/
    if (p_rank > MASTER) {
        MPI_Send(*A, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, COMP_Q, glob_comm);
    }  
}

/* Actually Gv for Q-Factor protection, Construction 1 */
void constructGv(double* Gv, int proc_rows, int f) {
    int i, j, k;

    int p_rank;
    MPI_Comm_rank(glob_comm, &p_rank);  

    double* V = calloc(((proc_rows - f) - f) * f, sizeof(double));
    randMatrix(V, (proc_rows - f) - f, f);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, f, f, proc_rows - 2*f, 
        -0.5, V, proc_rows - 2*f, V, proc_rows - 2*f, 0, Gv, proc_rows - f);
    
    LAPACKE_dlacpy(CblasRowMajor, 'A', f, proc_rows - 2*f, V, proc_rows - 2*f, Gv + f, proc_rows - f);

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

void genFail(double* Q, int target_rank, int p_rank, int loc_cols, int loc_rows) {
    int dim = loc_cols * loc_rows;
    if (target_rank == p_rank)
        for (int i=0; i < dim; ++i) {
            Q[i] = 0;
        }
}

void reconstructQ(double* Q, double* Gv_tilde, int target_row, int p_rank, int proc_cols, int proc_rows, int max_fails,
                    int loc_cols, int loc_rows) {

    int i;
    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;

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

    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;

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

                    /*  Q[:,j] reduced by Q[:,k] * R[k,j] */
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
