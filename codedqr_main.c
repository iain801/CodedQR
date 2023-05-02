/* Main Runner file for Coded QR PBMGS
 * for random square matrix A(n,n)
 * Using MGS in place of ICGS 
 * Takes arguements of: 
 *      np = (From MPI) number of processes/submatrices
 *      n = global matrix dimension (n x n matrix) 
 *      l = local matrix dimension (l x l matrices)
 *      f = maximum tolerable faults (f <= sqrt(np)/2)
 *
 * NOTE: Fault-Tolerance tested on Intel MPI 2021.5
 * Iain Weissburg 2023
 */

#include "codedqr_base.h"

int main(int argc, char** argv) {
    
    int	proc_size,              /* total number of processes */
        p_rank,                 /* a task identifier */ 
        p_row, p_col,           /* process coordinates in matrix */
        glob_cols, glob_rows,   /* global matrix dimensions */
        proc_cols, proc_rows,   /* processor grid dimensions */
        loc_cols, loc_rows,     /* local block dimensions */
        max_fails,              /* maximum tolerable failiures (f in literature) */
        check_cols, check_rows; /* checksum rows/columns */

    double  *A, *Q, *R,         /* main i/o matrices */
            *Gv_tilde,          /* Q Factor generator matrix */
            *Gh_tilde;          /* R Factor generator matrix */

    double  t1, t2, t3, t4, t5, t6, /* timer */
            error_norm,         /* norm of error matrix */
            *X, *B,             /* linear system results */
            *E;                 /* error matrix */
    
    int nrhs = 1;
    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &glob_comm);

    MPI_Comm_size(glob_comm, &proc_size);
    MPI_Comm_rank(glob_comm, &p_rank);  

    glob_cols = glob_rows = atoi(argv[1]);
    loc_cols = loc_rows = atoi(argv[2]);
    max_fails = atoi(argv[3]);
    
    proc_cols = glob_cols / loc_cols + max_fails;
    proc_rows = proc_size / proc_cols;

    check_cols = loc_cols * max_fails;
    check_rows = loc_rows * max_fails;
    
    p_col = p_rank % proc_cols;
    p_row = p_rank / proc_cols;

    MPI_Comm_split(glob_comm, p_col, p_rank, &col_comm);
    MPI_Comm_split(glob_comm, p_row, p_rank, &row_comm);
    MPI_Comm_group(row_comm, &row_group);
    MPI_Comm_group(col_comm, &col_group);

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        A = mkl_calloc((glob_cols + check_cols) * (glob_rows + check_rows), sizeof(double), 16);
        printf("mpirun -np %d ./out/codedqr_main %d %d %d\n", proc_size, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        printf("There are %d tasks in %d rows and %d columns\n\n", proc_size, proc_rows, proc_cols);

        /* Generate random matrix */
        if(SET_SEED) vslNewStream(&stream, VSL_BRNG_SFMT19937, SET_SEED);
        else vslNewStream(&stream, VSL_BRNG_SFMT19937, MPI_Wtime());
        randMatrixR(A, glob_cols, glob_rows, glob_cols + check_cols);

        B = mkl_malloc(glob_rows * nrhs * sizeof(double), 16);
        randMatrix(B, glob_rows, nrhs);

        if(DEBUG) {
            printf("Initializing array A: \n");
            printMatrix(A, glob_cols + check_cols, glob_rows + check_rows);
        }
    }

    /************* Distribute A across process Q *********************/

    Q = mkl_calloc(loc_cols * loc_rows, sizeof(double), 16);
    R = mkl_calloc(loc_cols * loc_rows, sizeof(double), 16);
    
    scatterA(A, Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows, max_fails);

    if (p_rank == MASTER) mkl_free(A);
    A = mkl_calloc(loc_cols * loc_rows, sizeof(double), 16);

    /* Start timer*/
    t1 = MPI_Wtime();

    /***************** Setup recon_inf Struct ************************/

    recon_inf.loc_cols = loc_cols;
    recon_inf.loc_rows = loc_rows;
    recon_inf.max_fails = max_fails;
    recon_inf.p_rank = p_rank;
    recon_inf.proc_cols = proc_cols;
    recon_inf.proc_rows = proc_rows;

    /******************* R-Factor Checksums **************************/

    Gh_tilde = mkl_calloc(max_fails * (proc_cols - max_fails), sizeof(double), 16);
    recon_inf.Gh_tilde = Gh_tilde;

    /* Construct Gh in master node and broadcast */
    if (p_rank == MASTER)
        constructGh(Gh_tilde, proc_cols, max_fails);

    MPI_Bcast(Gh_tilde, max_fails * (proc_cols - max_fails), MPI_DOUBLE, MASTER, glob_comm);

    checksumH(Q, p_rank);    

    glob_cols += max_fails * loc_cols;

    /******************* Q-Factor Checksums **************************/

    Gv_tilde = mkl_calloc((proc_rows - max_fails) * max_fails, sizeof(double), 16);
    recon_inf.Gv_tilde = Gv_tilde;

    /* Construct Gv in master node and broadcast */
    if (p_rank == MASTER)
        constructGv(Gv_tilde, proc_cols, max_fails);

    MPI_Bcast(Gv_tilde, (proc_rows - max_fails) * max_fails, MPI_DOUBLE, MASTER, glob_comm);    

    checksumV(Q, p_rank);

    /* End timer */
    t6 = MPI_Wtime() - t1;    

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t6, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t6 = exec_time / proc_size;
    }

    /****************** Copy A to Local Parts ************************/
    
    LAPACKE_dlacpy(CblasRowMajor,'A', loc_rows, loc_cols, Q, loc_cols, A, loc_cols);
    
    /************************ PBMGS **********************************/
    
    t1 = MPI_Wtime();

    pbmgs(Q, R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /* End timer */
    t2 = MPI_Wtime() - t1;    

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t2, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t2 = exec_time / proc_size;
    }
    
    /******************** Test Reconstruction ************************/
    t3 = MPI_Wtime();
    
    if(max_fails > 0) 
        genFail(Q, R, proc_cols, p_rank, loc_cols, loc_rows);
    if(max_fails > 1) {
        genFail(Q, R, proc_cols + 2, p_rank, loc_cols, loc_rows);
        genFail(Q, R, 2 * proc_cols, p_rank, loc_cols, loc_rows);
    }

    int *row_status = mkl_malloc(proc_rows * sizeof(int), 16);
    int *col_status = mkl_malloc(proc_rows * sizeof(int), 16);

    /* NOTE: Assuming proc_rows = proc_cols */
    for (int i=0;i<proc_rows;++i) {
        col_status[i] = 1;
        row_status[i] = 1;
    }

    if (max_fails > 0 && p_row == 1) {
        row_status[0] = 0;
        if(max_fails > 1) row_status[2] = 0;
    }
    if (max_fails > 0 && p_col == 0) {
        col_status[1] = 0;
        if(max_fails > 1) col_status[2] = 0;
    }
    if (max_fails > 1 && p_col == 2) {
        col_status[1] = 0;
    }
    if (max_fails > 1 && p_row == 2) {
        row_status[0] = 0;
    }

    reconstructR(R, row_status, p_rank);
    reconstructQ(Q, col_status, p_rank);

    mkl_free(row_status);
    mkl_free(col_status);

    t5 = MPI_Wtime() - t3;   

    /* Take average recovery time */
    {
    double exec_time;
    MPI_Reduce(&t5, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t5 = exec_time / proc_size;
    }   

    /****************** Check Results of QR **************************/

    E = mkl_calloc(loc_cols * loc_rows, sizeof(double), 16);

    t1 = MPI_Wtime();
    
    error_norm = checkError(A, Q, R, E, loc_cols, loc_rows, glob_cols, glob_rows + check_rows);  

    t4 = MPI_Wtime() - t1;   

    /* Take average checking time */
    {
    double exec_time;
    MPI_Reduce(&t4, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t4 = exec_time / proc_size;
    }
    
    t1 = MPI_Wtime();
    postOrthogonalize(Q, Gv_tilde, p_rank, proc_cols, proc_rows, loc_cols, loc_rows, max_fails);   
    t3 = MPI_Wtime() - t1;   

    /* Take average checking time */
    {
    double exec_time;
    MPI_Reduce(&t3, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t3 = exec_time / proc_size;
    }  

        
    /*********** Compile Q and R from local blocks *******************/

    gatherA(&A, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&E, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /******************* Solve linear system *************************/
    if (p_rank == MASTER) {
        t1 = MPI_Wtime();
        X = mkl_malloc(glob_rows * nrhs * sizeof(double), 16);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, glob_rows, nrhs, glob_rows, 1, Q, glob_cols, B, nrhs, 0, X, nrhs);
        LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', glob_rows, nrhs, R, glob_cols, X, nrhs);
        t1 = MPI_Wtime() - t1;

        /* Print all matrices */
        if (glob_cols < 100 && glob_rows < 100) {
            printf("\n");
            printf("Matrix A:\n");
            printMatrix(A, glob_cols, glob_rows + check_rows);

            if (fabs(error_norm) > 1e-9) {
                printf("Matrix B:\n");
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
        }
    
        /* Print Stats */
        printf("CS Construct Time: %.3g s\n", t6);
        printf("PBMGS Time: %.3g s\n", t2);
        printf("Node Recovery Time: %.3g s\n", t5);
        printf("QR Checking Time: %.3g s\n", t4);
        printf("Post-Ortho Time: %.3g s\n", t3);
        printf("Serial Solve Time: %.3g s\n", t1);
        printf("Roundoff Error: %.5g\n", error_norm); 

        char fname[30];
        sprintf(fname, "codedqr-test.csv");
        FILE *log = fopen(fname,"a");
        fprintf(log, "%d, %d, %d, %.8g, %.8g, %.8g, %.8g, %.8g, %.8g, %.8g\n", 
            proc_rows, glob_rows, max_fails, t6, t2, t5, t4, t3, t1, error_norm);
        fclose(log);

        mkl_free(B);
        mkl_free(X);
    }

    mkl_free(Gv_tilde);
    mkl_free(Gh_tilde);
    mkl_free(A);
    mkl_free(Q);
    mkl_free(R);
    mkl_free(E);

    if(p_rank == MASTER) printf("\n\n");

    MPI_Comm_free(&glob_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Group_free(&row_group);
    MPI_Group_free(&col_group);

    MPI_Finalize();
}