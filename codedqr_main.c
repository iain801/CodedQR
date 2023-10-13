/* Main Runner file for Coded QR PBMGS
 * for random square matrix A(n,n)
 * Using MGS in place of ICGS 
 * Takes arguements of: 
 *      np = (From MPI) number of processes/submatrices
 *      n = global matrix dimension (n x n matrix)
 *      f = maximum tolerable faults (f <= sqrt(np)/2)
 *      log = filename of timing output log (OPTIONAL)
 *
 * NOTE: Fault-Tolerance tested on Intel MPI 2021.5
 * Iain Weissburg 2023
 */

#include "codedqr_base.h"

#define CHECK_ERROR 0   /* check fp error */
#define USE_MASTER 0   /* generate and collect global on MASTER */ 

int main(int argc, char** argv) {
    
    int	proc_size,              /* total number of processes */
        p_rank,                 /* a task identifier */ 
        p_row, p_col,           /* process coordinates in matrix */
        glob_cols, glob_rows,   /* global matrix dimensions */
        proc_cols, proc_rows,   /* processor grid dimensions */
        loc_cols, loc_rows,     /* local block dimensions */
        max_fails,              /* maximum tolerable failiures (f in literature) */
        check_cols, check_rows; /* checksum rows/columns */

    double  *A = NULL, *Q, *R,  /* main i/o matrices */
            *Gv_tilde,          /* Q Factor generator matrix */
            *Gh_tilde;          /* R Factor generator matrix */

    double  t_solve, t_qr,      /* timer */
            t_postortho,        /* timer */
            t_valid, t_temp,    /* timer */
            t_decode, t_encode, /* timer */
            error_norm = 0,     /* norm of error matrix */
            *X, *B,             /* linear system results */
            *E;                 /* error matrix */

    int seed = SET_SEED;

    /******************** Initialize MPI *****************************/

    MPI_Init(&argc, &argv);

    /**************** Shuffle process ranks **************************/

    MPI_Group glob_group, glob_group_reordered;
    MPI_Comm_group(MPI_COMM_WORLD, &glob_group);
    MPI_Group_size(glob_group, &proc_size);
    MPI_Group_rank(glob_group, &p_rank); 

    if (!seed && p_rank == MASTER) {
        seed = MPI_Wtime() * 8888.8;
    }
 
    MPI_Bcast(&seed, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    vslNewStream(&stream, VSL_BRNG_SFMT19937, seed);

    int* shuffled_ranks = mkl_malloc(proc_size * sizeof(int), 64);
    for (int i=0; i < proc_size; ++i) shuffled_ranks[i] = i;

    int swap = 0;
    for (int i=0; i < proc_size; ++i) {
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &swap, i, proc_size);
        int temp = shuffled_ranks[swap];
        shuffled_ranks[swap] = shuffled_ranks[i];
        shuffled_ranks[i] = temp;
    }

    MPI_Group_incl(glob_group, proc_size, shuffled_ranks, &glob_group_reordered);
    MPI_Comm_create(MPI_COMM_WORLD, glob_group_reordered, &glob_comm);

    MPI_Comm_size(glob_comm, &proc_size);
    MPI_Comm_rank(glob_comm, &p_rank); 

    MPI_Group_free(&glob_group);
    MPI_Group_free(&glob_group_reordered);

    mkl_free(shuffled_ranks);

    /************ Set Configuration Variables ************************/

    glob_cols = glob_rows = atoi(argv[1]);
    max_fails = atoi(argv[2]);
    
    proc_cols = proc_rows = (int) floor(sqrt(proc_size));
    
    loc_cols = glob_cols / (proc_cols - max_fails);
    loc_rows = glob_rows / (proc_rows - max_fails);

    check_cols = loc_cols * max_fails;
    check_rows = loc_rows * max_fails;
    
    p_col = p_rank % proc_cols;
    p_row = p_rank / proc_cols;

    MPI_Comm_split(glob_comm, p_col, p_rank, &col_comm);
    MPI_Comm_split(glob_comm, p_row, p_rank, &row_comm);

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        printf("mpirun -np %d ./out/codedqr_main %d %d\n", proc_size, glob_cols, max_fails);
        printf("There are %d tasks in %d rows and %d columns\n", proc_size, proc_rows, proc_cols);
        printf("Local matrix dimensions: %d x %d\n\n", loc_cols, loc_rows);
        fflush(stdout);

        if (USE_MASTER) {
            A = mkl_calloc((glob_cols + check_cols) * (glob_rows + check_rows), sizeof(double), 64);
            randMatrixR(A, glob_cols, glob_rows, glob_cols + check_cols);
            
            /* Limit generated A precision to 5 decimal places */
            for (int i=0; i < glob_rows * glob_cols; i++) {
                A[i] = roundf(A[i] * 1e5);
                A[i] = A[i] * 1e-5;
            }
            
            // for (int i = 0; i < glob_cols; ++i) {
            //     for (int j = 0; j < glob_rows; j++) {
            //         A[i * (glob_cols + check_cols) + j] = (i == j) ? 1 : 0;
            //     }
            // }

            if(DEBUG) {
                printf("Initializing array A: \n");
                printMatrix(A, glob_cols + check_cols, glob_rows + check_rows);
            }
        }
        
    }

    /************* Distribute A across process Q *********************/

    Q = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);
    R = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);

    if(USE_MASTER) {
        scatterA(A, Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows, max_fails);
        if (p_rank == MASTER) mkl_free(A);
    }
    else if (p_col < proc_cols - max_fails && p_row < proc_rows - max_fails) {
        randMatrix(Q, loc_cols, loc_rows);
    }
    A = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);
    

    /***************** Setup recon_info Struct ************************/

    recon_info.loc_cols = loc_cols;
    recon_info.loc_rows = loc_rows;
    recon_info.max_fails = max_fails;
    recon_info.p_rank = p_rank;
    recon_info.proc_cols = proc_cols;
    recon_info.proc_rows = proc_rows;
    
    Gh_tilde = mkl_calloc(max_fails * (proc_cols - max_fails), sizeof(double), 64);
    recon_info.Gh_tilde = Gh_tilde; 

    Gv_tilde = mkl_calloc((proc_rows - max_fails) * max_fails, sizeof(double), 64);
    recon_info.Gv_tilde = Gv_tilde;

    recon_info.t_decode = 0.0;

    /******************* Generate Gv and Gh **************************/
    /* Start timer*/
    MPI_Barrier(glob_comm);
    t_temp = MPI_Wtime(); 

    /* Construct Gh and Gv in master node and broadcast */
    if (p_rank == MASTER) {
        constructGh(Gh_tilde, proc_cols, max_fails);
        constructGv(Gv_tilde, proc_rows, max_fails);
    }
    
    MPI_Bcast(Gh_tilde, max_fails * (proc_cols - max_fails), MPI_DOUBLE, MASTER, glob_comm);
    MPI_Bcast(Gv_tilde, (proc_rows - max_fails) * max_fails, MPI_DOUBLE, MASTER, glob_comm); 
        
    /******************* R-Factor Checksums **************************/

    if(p_row < proc_rows - max_fails) {
        checksumH(Q, p_rank);  
    }  

    glob_cols += max_fails * loc_cols;

    /******************* Q-Factor Checksums **************************/

    checksumV(Q, p_rank);

    /* End timer */
    t_encode = MPI_Wtime() - t_temp;

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t_encode, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t_encode = exec_time / proc_size;
    }

    /****************** Copy A to Local Parts ************************/
    
    LAPACKE_dlacpy(CblasRowMajor,'A', loc_rows, loc_cols, Q, loc_cols, A, loc_cols);
    
    /************************ PBMGS **********************************/
    /* Start timer*/
    MPI_Barrier(glob_comm);
    t_temp = MPI_Wtime();

    pbmgs(Q, R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /* End timer */
    t_qr = MPI_Wtime() - t_temp;    

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t_qr, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t_qr = exec_time / proc_size;
    }

    /********************* Check A = QR ******************************/

    E = mkl_calloc(loc_cols * loc_rows, sizeof(double), 64);
    
    t_temp = MPI_Wtime();

    if (CHECK_ERROR) {
        error_norm = checkError(A, Q, R, E, loc_cols, loc_rows, 
            glob_cols, glob_rows + check_rows);  
    }

    t_valid = MPI_Wtime() - t_temp;   

    /* Take average checking time */
    {
    double exec_time;
    MPI_Reduce(&t_valid, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t_valid = exec_time / proc_size;
    }

    /***************** Post-Orthogonalization ************************/

    
    MPI_Barrier(glob_comm);
    t_temp = MPI_Wtime();
    postOrthogonalize(Q, Gv_tilde, p_rank, proc_cols, proc_rows, loc_cols, loc_rows, max_fails);   
    t_postortho = MPI_Wtime() - t_temp;   

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t_postortho, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t_postortho = exec_time / proc_size;
    }  

        
    /*********** Compile Q and R from local blocks *******************/

    if (USE_MASTER) {
        gatherA(&A, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
        gatherA(&Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
        gatherA(&R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
        gatherA(&E, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    }

    /******************* Solve linear system *************************/
    if (p_rank == MASTER) {
        if (USE_MASTER) {
            int nrhs = 1;

            X = mkl_malloc(glob_rows * nrhs * sizeof(double), 64);
            B = mkl_malloc(glob_rows * nrhs * sizeof(double), 64);

            randMatrix(B, glob_rows, nrhs);

            /* Limit generated B precision to 5 decimal places */
            for (int i=0; i < glob_rows * nrhs; i++) {
                B[i] = roundf(B[i] * 1e5);
                B[i] = B[i] * 1e-5;
            }
            
            if (DO_FINAL_SOLVE) {
                t_temp = MPI_Wtime();

                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, glob_rows, nrhs, glob_rows, 1, Q, glob_cols, B, nrhs, 0, X, nrhs);
                LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', glob_rows, nrhs, R, glob_cols, X, nrhs);

                t_solve = MPI_Wtime() - t_temp;
            }

            /* Print all matrices */
            if (glob_cols < 100 && glob_rows < 100) {
                printf("\n");
                printf("Matrix A:\n");
                printMatrix(A, glob_cols, glob_rows + check_rows);

                if (fabs(error_norm) > 1e-4) {
                    printf("Matrix E:\n");
                    printMatrix(E, glob_cols, glob_rows + check_rows);
                }

                printf("Matrix Q:\n");
                printMatrix(Q, glob_cols, glob_rows + check_rows);

                printf("Matrix R:\n");
                printMatrix(R, glob_cols, glob_rows + check_rows);

                printf("Matrix B:\n");
                printMatrix(B, nrhs, glob_rows);

                printf("Matrix X:\n");
                printMatrix(X, nrhs, glob_rows);

                fflush(stdout);
            }
            
            mkl_free(B);
            mkl_free(X);
        }
    
        /* Print Stats */

        t_decode = recon_info.t_decode;
        t_qr -= t_decode;

        printf("CS Construct Time: %.3g s\n", t_encode);
        printf("PBMGS Time: %.3g s\n", t_qr);
        printf("Node Recovery Time: %.3g s\n", t_decode);
        printf("QR Checking Time: %.3g s\n", t_valid);
        printf("Post-Ortho Time: %.3g s\n", t_postortho);
        printf("Serial Solve Time: %.3g s\n", t_solve);
        printf("Roundoff Error: %.5g\n", error_norm); 
        if (error_norm > 1e-4) {
            printf("WARNING: HIGH ERROR \n");
        }
        printf("\n\n");
        fflush(stdout);

        if (argc == 4) {
            char fname[30];
            sprintf(fname, "%s", argv[3]);
            FILE *log = fopen(fname,"a");
            fprintf(log, "%d,%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g\n", 
                proc_rows-max_fails, glob_rows, max_fails, t_decode, t_solve, t_postortho, t_encode, t_qr);
            fclose(log);
        }
        
    }

    mkl_free(Gv_tilde);
    mkl_free(Gh_tilde);
    mkl_free(A);
    mkl_free(Q);
    mkl_free(R);
    mkl_free(E);
    
    mkl_free_buffers();

    MPI_Comm_free(&glob_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();

    return 0;
}