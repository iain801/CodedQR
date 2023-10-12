/* Coded Parallel Block Modified Gram Schmidt
 * implementation with MPI using Off node checksum storage
 * Core functions
 * 
 * Iain Weissburg 2023
 */

#include "codedqr_base.h"

struct ReconInfo recon_info;

VSLStreamStatePtr stream;
MPI_Comm glob_comm, row_comm, col_comm;

void printMatrix(double* matrix, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%-+6.3f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fprintMatrix(FILE* out, double* matrix, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(out, "%-+6.3f ", matrix[i*cols + j]);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "\n");
}

void iprintMatrix(int* matrix, int cols, int rows) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", matrix[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Fill Matrix with standard normal randoms, with row major size rowsize */
void randMatrixR(double* A, int n, int m, int rowsize) {
    for (int i=0; i < m; ++i)
        vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n, A + i*rowsize, 0, 1 );
}

/* Fill Matrix with standard normal randoms */
void randMatrix(double* A, int n, int m) {
    randMatrixR(A, n, m, n);
}

/* sets E = QR and returns 1-norm of A - E */
double checkError(double* A, double* Q, double* R, double* E, 
    int loc_cols, int loc_rows, int glob_cols, int glob_rows) {
    
    double out_norm = 0;    

    double  *Q_bar = mkl_calloc(loc_rows * glob_cols, sizeof(double), 16), 
            *R_bar = mkl_calloc(loc_cols * glob_rows, sizeof(double), 64);

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
                    E[j * loc_rows + i] += Q_bar[offset + j * loc_cols + k] * R_bar[offset + k * loc_cols + i];
                }
            }
        }
    }

    cblas_daxpy(loc_cols*loc_rows, -1, A, 1, E, 1);

    double part_norm = cblas_dnrm2(loc_cols*loc_rows, E, 1);
    
    MPI_Reduce(&part_norm, &out_norm, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);

    mkl_free(Q_bar);
    mkl_free(R_bar);

    return out_norm;
}

void scatterA(double* A, double* Q, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows, int max_fails) {
    
    int i, j;
    int glob_cols = loc_cols * proc_cols;

    /* Master distributes A across process Q matrices */
    if (p_rank == MASTER) {

        /* i = process col */
        for (i = proc_cols - 1; i >= 0; --i) {
            /* j = process row*/
            for (j = proc_rows - 1; j >= 0; --j) {
                int target = j * proc_cols + i;
                LAPACKE_dlacpy(CblasRowMajor, 'A', loc_rows, loc_cols, 
                    A + (j * proc_cols * loc_rows + i) * loc_cols, glob_cols, Q, loc_cols);

                /* If target not master, send Q */
                if (target != MASTER) {
                    MPI_Send(Q, loc_cols * loc_rows, MPI_DOUBLE,
                        target, DIST_Q, glob_comm);
                }
            }
        }
    }

    /* If not master recieve Q */
    else {
        MPI_Recv(Q, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, DIST_Q, glob_comm, MPI_STATUS_IGNORE);
    }
}

void gatherA(double** A,  int p_rank, int proc_cols, 
    int proc_rows, int loc_cols, int loc_rows) {

    int i, j;
    int glob_cols = loc_cols * proc_cols;
    int glob_rows = loc_rows * proc_rows;

    /* Master compiles matrix A from across processes */
    if (p_rank == MASTER) {
        double *A_glob = mkl_calloc(glob_cols * glob_rows, sizeof(double), 64);
        for (i = 0; i < proc_cols; ++i) {
            for (j = 0; j < proc_rows; ++j) {

                /* If target not master, recieve A */
                int target = j * proc_cols + i;
                if (target != MASTER) {
                    MPI_Recv(*A, loc_cols * loc_rows, MPI_DOUBLE,
                        target, COMP_Q, glob_comm, MPI_STATUS_IGNORE);
                }
                
                LAPACKE_dlacpy(CblasRowMajor, 'A', loc_cols, loc_rows, *A, loc_cols, 
                    A_glob + (j * proc_cols * loc_rows + i) * loc_cols, glob_cols);

                // printf("A_glob after (%d):\n", target);
                // printMatrix(A_glob, glob_cols, glob_rows);
                // // printMatrix(Q, loc_cols * proc_cols, loc_rows * proc_rows);
                // // fflush(stdout);
            }
        }
        mkl_free(*A);
        *(A) = A_glob;
    }

    /* If not master send A*/
    else {
        // *A = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);

        MPI_Send(*A, loc_cols * loc_rows, MPI_DOUBLE,
            MASTER, COMP_Q, glob_comm);
    }  
}

/* Actually Gv for Q-Factor protection, Construction 1 */
void constructGv(double* Gv, int proc_rows, int f) {
    if(f == 0) return;

    int p_rank;
    MPI_Comm_rank(glob_comm, &p_rank);  

    double* V = mkl_calloc(((proc_rows - f) - f) * f, sizeof(double), 64);
    randMatrix(V, (proc_rows - f) - f, f);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, f, f, proc_rows - 2*f, 
        -0.5, V, proc_rows - 2*f, V, proc_rows - 2*f, 0, Gv, proc_rows - f);
    
    LAPACKE_dlacpy(CblasRowMajor, 'A', f, proc_rows - 2*f, V, proc_rows - 2*f, Gv + f, proc_rows - f);

    if (DEBUG) {
        printf( "V:\n");
        printMatrix(V, (proc_rows - f) - f, f);
        printf( "G_pre:\n");
        printMatrix(Gv, (proc_rows - f), f);
    }

    mkl_free(V);
}

/* Actually Gh for R-factor protection, random */
void constructGh(double* Gh, int proc_cols, int f) {
    if(f == 0) return;
    randMatrix(Gh, f, proc_cols - f);
}

void checksumV(double *Q, int p_rank) {
    if(recon_info.max_fails == 0) return;
    int j;

    int p_row = p_rank / recon_info.proc_cols;
    int cs_row = p_row - recon_info.proc_rows + recon_info.max_fails;
    
    double* Q_bar = mkl_calloc(recon_info.loc_cols * recon_info.loc_rows, sizeof(double), 64);

    for (j=0; j < recon_info.max_fails; ++j) {
        if (cs_row < 0) { //if non-checksum node

            /* Copy corrosponding Gv_tilde term * Q -> Qbar*/
            cblas_daxpby(recon_info.loc_cols * recon_info.loc_rows, 
                recon_info.Gv_tilde[j * (recon_info.proc_rows - recon_info.max_fails) + p_row], 
                Q, 1, 0, Q_bar, 1);

            // if(p_rank == MASTER) printf("Marker Qbar scale\n");
        }

        //reduce checksums to checksum nodes
        MPI_Reduce(Q_bar, Q, recon_info.loc_cols * recon_info.loc_rows, MPI_DOUBLE, 
            MPI_SUM, j + recon_info.proc_rows - recon_info.max_fails, col_comm);

        // if(p_rank == MASTER) printf("Marker Reduce\n");
    }
    mkl_free(Q_bar);
}

void checksumH(double *Q, int p_rank) {
    if(recon_info.max_fails == 0) return;
    int j;

    int p_col = p_rank % recon_info.proc_cols;
    int cs_col = p_col - recon_info.proc_cols + recon_info.max_fails;
    
    double* Q_bar = mkl_calloc(recon_info.loc_cols * recon_info.loc_rows, sizeof(double), 64);

    for (j=0; j < recon_info.max_fails; ++j) {
        if (cs_col < 0) { //if non-checksum node

            /* Copy corrosponding Gh_tilde term * Q -> Qbar*/
            cblas_daxpby(recon_info.loc_cols * recon_info.loc_rows, 
                recon_info.Gh_tilde[p_col * recon_info.max_fails + j], 
                Q, 1, 0, Q_bar, 1);
        }

        //reduce checksums to checksum nodes
        MPI_Reduce(Q_bar, Q, recon_info.loc_cols * recon_info.loc_rows, MPI_DOUBLE, 
            MPI_SUM, j + recon_info.proc_cols - recon_info.max_fails, row_comm);

    }
    mkl_free(Q_bar);
}

void genFail(double* Q, double* R, int* col_status, int* row_status, int p_rank) {
    int p_col = p_rank % recon_info.proc_cols;
    int p_row = p_rank / recon_info.proc_cols;

    if (col_status[p_row] && row_status[p_col]) {
        for (int i=0; i < recon_info.loc_cols * recon_info.loc_rows; ++i) {
            Q[i] = NAN;
            R[i] = NAN;
        }
    }
}

void reconstructQ(double* Q, int* node_status, int p_rank) {
    if (recon_info.max_fails == 0) return;

    int i, j;
    int p_row = p_rank / recon_info.proc_cols;
    int m = recon_info.proc_rows - recon_info.max_fails;
    int n = recon_info.proc_cols - recon_info.max_fails;
    double* Q_bar = mkl_calloc(recon_info.loc_cols * recon_info.loc_rows, sizeof(double), 64);
    double* Gv_succ = mkl_calloc(n * m, sizeof(double), 64);

    
    /* first m active nodes map, first_m_nodes[index] = node */
    int* first_m_nodes = mkl_malloc(m * sizeof(int), 64);

    /* inverse active node map, first_m_nodes_i[node] = index */
    int* first_m_nodes_i = mkl_malloc(recon_info.proc_rows * sizeof(int), 64);
    
    int reg_fails = 0;
    for (i=0; i < m; ++i) {
        if (node_status[i])
            reg_fails = 1;
    }  
    
    if (reg_fails) {
        /* fill node maps with first m active nodes */
        for (i=0, j=0; j < m; ++i) {
            if (!node_status[i]) {
                first_m_nodes_i[i] = j;
                first_m_nodes[j++] = i;
            }
            /* if node is failed, set inverse map to -1 */
            else {
                first_m_nodes_i[i] = -1;
            }
        }

        /* finish filling inverse map with -1 */
        for (;i<recon_info.proc_rows;++i) {
            first_m_nodes_i[i] = -1;
        }

        /* if node is active */
        if (!node_status[p_row]) {

            /*********** Compute Success Matrix ****************/

            for (i=0; i < m; ++i) {
                /* if regular node, set Gv_succ to 1 */
                if (first_m_nodes[i] < m) { 
                    Gv_succ[i * n + first_m_nodes[i]] = 1;
                }
                /* if checksum node, set Gv_succ to Gv_tidle */
                else {
                    cblas_dcopy(n, recon_info.Gv_tilde + (first_m_nodes[i] - m) * n, 1, Gv_succ + i * n, 1);
                }
            }

            /* Take inverse of success matrix */
            int* ipiv = mkl_malloc(m * sizeof(int), 64);
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, Gv_succ, n, ipiv);
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, m, Gv_succ, n, ipiv);
            mkl_free(ipiv);
        }

        /************ Perform Matrix Reductions ***********/
        
        for(i=0; i<m; ++i) {
            /* if node i is failed*/
            if(node_status[i]) { 
                /* if chosen node */
                if(first_m_nodes_i[p_row] > -1) {

                    /* Copy Gv_succ * Q to Qbar*/
                    cblas_daxpby(recon_info.loc_rows * recon_info.loc_cols, 
                        Gv_succ[i * n + first_m_nodes_i[p_row]], 
                        Q, 1, 0, Q_bar, 1);                   
    
                }
                MPI_Reduce(Q_bar, Q, recon_info.loc_cols * recon_info.loc_rows, MPI_DOUBLE, MPI_SUM, i, col_comm);
                for (j=recon_info.loc_cols * recon_info.loc_rows-1; j >=0; --j) { Q_bar[j] = 0; }
            }
        }
    }

    /* Reconstruct Checksums if failed*/
    for (i=m; i < recon_info.proc_rows; ++i) {
        if(node_status[i]) {
            checksumV(Q, p_rank);
            break;
        }   
    }

    mkl_free(Gv_succ);
    mkl_free(first_m_nodes);
    mkl_free(first_m_nodes_i);
    mkl_free(Q_bar);
}

void reconstructR(double* R, int* node_status, int p_rank) {
    if (recon_info.max_fails == 0) return;

    int i, j;
    int p_col = p_rank % recon_info.proc_cols;
    int m = recon_info.proc_rows - recon_info.max_fails;
    int n = recon_info.proc_cols - recon_info.max_fails;
    double* R_bar = mkl_calloc(recon_info.loc_cols * recon_info.loc_rows, sizeof(double), 64);
    double* Gh_succ = mkl_calloc(n * m, sizeof(double), 64);

    int reg_fails = 0;
    for (i=0; i < n; ++i) {
        if (node_status[i])
            reg_fails = 1;
    }  
    /* first n active nodes map, first_n_nodes[index] = node */
    int* first_n_nodes = mkl_malloc(n * sizeof(int), 64);

    /* inverse active node map, first_n_nodes_i[node] = index */
    int* first_n_nodes_i = mkl_malloc(recon_info.proc_cols * sizeof(int), 64);

    /* if node is active */
    if (reg_fails) {
        /* fill node maps with first n active nodes */
        for (i=0, j=0; j < n; ++i) {
            if (!node_status[i]) {
                first_n_nodes_i[i] = j;
                first_n_nodes[j++] = i;
            }
            /* if node is failed, set inverse map to -1 */
            else {
                first_n_nodes_i[i] = -1;
            }
        }

        /* finish filling inverse map with -1 */
        for (;i<recon_info.proc_cols;++i) {
            first_n_nodes_i[i] = -1;
        }

        if (!node_status[p_col]) {

            /*********** Compute Success Matrix ****************/

            for (i=0; i < n; ++i) {
                /* if regular node, set Gh_succ to 1 */
                if (first_n_nodes[i] < n) { 
                    Gh_succ[n * first_n_nodes[i] + i] = 1;
                }
                /* if checksum node, set Gh_succ to Gh_tidle */
                else {
                    cblas_dcopy(m, recon_info.Gh_tilde + (first_n_nodes[i] - n), recon_info.max_fails, Gh_succ + i, n);
                }
            }

            /* Take inverse of success matrix */
            int* ipiv = mkl_malloc(n * sizeof(int), 64);
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, m, Gh_succ, m, ipiv);
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, Gh_succ, m, ipiv);
            mkl_free(ipiv);
        }

        /************ Perform Matrix Reductions ***********/

        for(i=0; i<n; ++i) {
            /* if node i is failed*/
            if(node_status[i]) { 
                /* if chosen node */
                if(first_n_nodes_i[p_col] > -1) {

                    /* Copy Gh_succ * R to Rbar*/
                    cblas_daxpby(recon_info.loc_rows * recon_info.loc_cols, 
                        Gh_succ[n * first_n_nodes_i[p_col] + i], 
                        R, 1, 0, R_bar, 1);
                } 
                MPI_Reduce(R_bar, R, recon_info.loc_cols * recon_info.loc_rows, MPI_DOUBLE, MPI_SUM, i, row_comm); 
                for (j=recon_info.loc_cols * recon_info.loc_rows-1; j >=0; --j) { R_bar[j] = 0; }
            }
        }
    }

    /* Reconstruct Checksums if failed*/
    for (i=n; i < recon_info.proc_cols; ++i) {
        if(node_status[i]) {
            checksumH(R, p_rank);
            break;
        }        
    }

    mkl_free(Gh_succ);
    mkl_free(first_n_nodes);
    mkl_free(first_n_nodes_i);
    mkl_free(R_bar);
}

void pbmgs(double* Q, double* R, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows) {

    int APC,                        /* active process column */
        i, j, k;                    /* iterator */
    double  Qnorm, Qdot,            /* operation variable */
            Qnorm_loc, Qdot_loc;    /* operation local variable */

    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;

    double* Qbar = mkl_malloc(loc_cols * loc_rows * sizeof(double), 64);
    double* Rbar = mkl_malloc(loc_cols * loc_rows * sizeof(double), 64);

    int *row_status = mkl_calloc(proc_cols, sizeof(int), 64);
    int *col_status = mkl_calloc(proc_rows, sizeof(int), 64);

    /* For each block */
    for (APC = 0; APC < proc_cols; ++APC) {
        i = APC * loc_cols;

        /* Test Reconstruction */
        if (TEST_FAILIURE) {
            MPI_Barrier(glob_comm);
            double t_temp = MPI_Wtime();

            for (int z=0; z < proc_cols; ++z) {
                row_status[z] = 0;
                col_status[z] = 0;
            }

            for (int z=0; z < recon_info.max_fails; ++z) {
                if (proc_cols - p_row - z > 0)
                    row_status[proc_cols - p_row - z - 1] = 1;

                if (proc_rows - p_col - z > 0)
                    col_status[proc_rows - p_col - z - 1] = 1;
            }

            genFail(Q, R, col_status, row_status, p_rank);

            reconstructR(R, row_status, p_rank);
            reconstructQ(Q, col_status, p_rank);

            t_temp = MPI_Wtime() - t_temp;  

            /* Take average recovery time */
            {
            double exec_time;
            MPI_Reduce(&t_temp, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
            recon_info.t_decode += exec_time / (proc_rows * proc_cols);
            }  
        }
        /* Back to algorithm */

        /* ICGS Step */
        if (p_col == APC) {
            for (j = 0; j < loc_cols; ++j) {
                Qnorm = 0;

                Qnorm_loc = cblas_ddot(loc_rows, Q + j, loc_cols, Q + j, loc_cols);
            
                /* Allreduce to find squared sum of Qbar */
                MPI_Allreduce(&Qnorm_loc, &Qnorm, 1, MPI_DOUBLE,
                    MPI_SUM, col_comm);

                Qnorm = sqrt(Qnorm);

                /* Normalize local portions of Qbar */
                cblas_dscal(loc_rows, 1 / Qnorm, Q + j, loc_cols);

                /* Set R to Qnorm in the correct row in the correct node */                    
                if (p_row == (i + j) / loc_rows) {
                    if(DEBUG) printf("Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qnorm, (i + j) % loc_rows, k);
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
                        printf("Qdot_%d: %.3f\n\n", p_col*loc_cols + k, Qdot);
                    }

                    // Q[:,k] = Q[:,k] - Qdot * Qbar
                    cblas_daxpy(loc_rows, -1 * Qdot, Q + j, loc_cols, Q + k, loc_cols);


                    /* Set R to Qdot in the correct row in the correct node */                    
                    if (p_row == (i + j) / loc_rows) {
                        if(DEBUG) printf("Process (%d,%d) is setting %.3f at (%d,%d)", p_row, p_col, Qdot, (i + j) % loc_rows, k);
                        R[((i + j) % loc_rows)*loc_cols + k] = Qdot;
                    }
                }
            }

            if(DEBUG) {
                printf("Q_reduced (%d,%d)\n", p_col, p_row);
                printMatrix(Q, loc_cols, loc_rows);
            }

            /* Copy Q into Qbar for broadcast */
            LAPACKE_dlacpy(CblasRowMajor, 'A', loc_cols, loc_rows, Q, loc_cols, Qbar, loc_cols);
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
                    Qdot_loc = cblas_ddot(loc_rows, Q + j, loc_cols, Qbar + k, loc_cols);

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

                    /*  Q[:,j] reduced by Q[:,k] * R[k,j] */
                    cblas_daxpy(loc_rows, -1 * Rbar[k*loc_cols + j], Qbar + k, loc_cols, Q + j, loc_cols);

                    /* Set R in the correct row in the correct node */                    
                    if (p_row == (i + k) / loc_rows) {
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
    mkl_free(Qbar);
    mkl_free(Rbar);
    mkl_free(row_status);
    mkl_free(col_status);
}

/* finds matrix Q^T = (G0Q1)^T * G0 
    outputs Q^T into Q */
void postOrthogonalize(double* Q, double* Gv_tilde, int p_rank, 
    int proc_cols, int proc_rows, int loc_cols, int loc_rows, int max_fails) {

    if (max_fails == 0) {
        return;
    }
    
    int i;
    int p_col = p_rank % proc_cols;
    int p_row = p_rank / proc_cols;
    int m = proc_rows - max_fails;
    int n = proc_cols - max_fails;
    int is_check = p_col / n + p_row / m;
    int reg_rank;
    
    /* Split communicators for regular and checksum nodes */
    MPI_Comm reg_comm, reg_col, reg_row;
    MPI_Comm_split(glob_comm, is_check, p_rank, &reg_comm);
    MPI_Comm_split(reg_comm, p_col, p_row, &reg_col);
    MPI_Comm_split(reg_comm, p_row, p_col, &reg_row);
    MPI_Comm_rank(reg_comm, &reg_rank);
    
    double* G0 = mkl_calloc(n * m, sizeof(double), 64);
    
    /* Copy Gv into G0 */
    cblas_dcopy(n * max_fails, Gv_tilde, 1, G0, 1);

    /* Copy V^T from Gv into G0 */
    mkl_domatcopy2('R', 'T', max_fails, n - max_fails, 1, Gv_tilde + max_fails, n, 1,
        G0 + n * max_fails, n, 1);

    /* Identity matrix addition/subtraction */
    for(i=0; i < max_fails; ++i) {
        G0[i*n+i] += 1;
    }
    for(; i < n; ++i) {
        G0[i*n+i] -= 1;
    }
    
    /* Perform operation if regular node 
        operation (G0Q1)^T G0 */
    if (!is_check) {

        double* Q_bar = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);
        double* Q_res = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);
        
        /* Perform Q_res = G0 * Q1 */
        for (i=0; i < m; ++i) {
            cblas_daxpby(loc_cols * loc_rows, G0[i * n + p_row], Q, 1, 0, Q_bar, 1);

            MPI_Reduce(Q_bar, Q_res, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, i, reg_col);
        }
        
        /* Transpose Q_res*/
        if(p_row != p_col) { 
            /* Distributed transpose */
            int target_rank = p_col * n + p_row;
            MPI_Sendrecv_replace(Q_res, loc_cols * loc_rows, MPI_DOUBLE, 
                target_rank, 0, target_rank, 0, reg_comm, MPI_STATUS_IGNORE);
        }
        
        /* Local transpose */
        mkl_dimatcopy('R', 'T', loc_rows, loc_cols, 1, Q_res, loc_cols, loc_rows);

        /* Q_res * G0 */
        for (i=0; i < n; ++i) {
            cblas_daxpby(loc_cols * loc_rows, G0[p_col * n + i], Q_res, 1, 0, Q_bar, 1);

            MPI_Reduce(Q_bar, Q, loc_cols * loc_rows, MPI_DOUBLE, MPI_SUM, i, reg_row);
        }

        mkl_free(Q_bar);
        mkl_free(Q_res);
    }

    mkl_free(G0);

    MPI_Comm_free(&reg_row);
    MPI_Comm_free(&reg_col);
    MPI_Comm_free(&reg_comm);
}
