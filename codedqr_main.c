/* Main Runner file for Coded QR PBMGS
 * for random square matrix A(n,n)
 * Using MGS in place of ICGS 
 * Takes arguements of: 
 *      np = (From MPI) number of processes/submatrices
 *      n = global matrix dimension (n x n matrix) 
 *      l = local matrix dimension (l x l matrices)
 *
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

    double  t1, t2, t3, t4,     /* timer */
            error_norm,         /* norm of error matrix */
            *E;                 /* error matrix */
    

    /*********************** Initialize MPI *****************************/

    MPI_Init (&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &glob_comm);

    MPI_Comm_size(glob_comm, &proc_size);
    MPI_Comm_rank(glob_comm, &p_rank);  

    proc_cols = (int) sqrt(proc_size);
    proc_rows = proc_size / proc_cols;
    max_fails = proc_rows - proc_cols;

    glob_cols = glob_rows = atoi(argv[1]);
    loc_cols = loc_rows = atoi(argv[2]);
    check_cols = 0;
    check_rows = loc_rows * max_fails;

    char fname[99];
    sprintf(fname, "log_%d_%dx_%d.txt", proc_size, glob_cols, loc_cols);
    fp_log = fopen(fname, "a");

    p_col = p_rank % proc_cols;
    p_row = p_rank / proc_cols;

    MPI_Comm_split(glob_comm, p_col, p_rank, &col_comm);
    MPI_Comm_split(glob_comm, p_row, p_rank, &row_comm);
    MPI_Comm_group(row_comm, &row_group);
    MPI_Comm_group(col_comm, &col_group);

    /******************* Initialize arrays ***************************/

    if (p_rank == MASTER)
    {
        A = (double*) calloc(glob_cols * (glob_rows + check_rows), sizeof(double));
        printf("codedqr_on_sq has started with %d tasks in %d rows and %d columns\n", proc_size, proc_rows, proc_cols);
        printf("Each process has %d rows and %d columns\n\n", loc_rows, loc_cols);

        /* Generate random matrix */
        if(SET_SEED) vslNewStream(&stream, VSL_BRNG_SFMT19937, SET_SEED);
        else vslNewStream(&stream, VSL_BRNG_SFMT19937, MPI_Wtime());
        randMatrix(A, glob_cols, glob_rows);
        if(DEBUG) {
            printf("Initializing array A: \n");
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

    if (p_rank == MASTER) free(A);
    A = (double*) calloc(loc_cols * loc_rows, sizeof(double));
    
    /* Start timer*/
    t1 = MPI_Wtime();

    /******** Build Checksum Generator Matrices **********************/
    
    Gv_tilde = (double*) calloc((proc_rows - max_fails) * max_fails, sizeof(double));
    Gh_tilde = (double*) calloc(max_fails * (proc_cols - max_fails), sizeof(double));

    /* Construct Gv and Gh in master node */
    if (p_rank == MASTER)
        constructGv(Gv_tilde, proc_rows, max_fails);
        // constructGh(Gh_tilde, proc_cols, max_fails);

    /* Broadcast Gv and Gh from master node */
    MPI_Bcast(Gv_tilde, (proc_rows - max_fails) * max_fails, MPI_DOUBLE, MASTER, glob_comm);
    MPI_Bcast(Gh_tilde, max_fails * (proc_cols - max_fails), MPI_DOUBLE, MASTER, glob_comm);
    
    /***************** Q-Factor Checksums ****************************/
    int i, j;
    int cs_row = p_row - (proc_rows - max_fails);
    int cs_col = p_col - proc_cols;// + max_fails;
    
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
                printf( "Checkum node %d Q is now:\n", p_rank);
                printMatrix(Q, loc_cols, loc_rows);
            }
        }
    }

    /***************** R-Factor Checksums ****************************/

    /****************** Copy A to Local Parts ************************/
    
    t1 += MPI_Wtime();
    LAPACKE_dlacpy(CblasRowMajor,'A', loc_rows, loc_cols, Q, loc_cols, A, loc_cols);
    t1 -= MPI_Wtime();

    /******************** Test Reconstruction ************************/

    genFail(Q, 1, p_rank, loc_cols, loc_rows);
    if (p_col == 0) {
        int node_status[3] = {1, 0, 1};
        reconstructQ(Q, Gv_tilde, node_status, p_rank, proc_cols, proc_rows, max_fails, loc_cols, loc_rows);
    }
    else if (p_col == 1) {
        int node_status[3] = {1, 0, 1};
        reconstructQ(Q, Gv_tilde, node_status, p_rank, proc_cols, proc_rows, max_fails, loc_cols, loc_rows);
    }

    if(p_rank == MASTER) {
        printf("Recon A1:\n");
        printMatrix(Q, glob_cols, glob_rows);

        printf("Recon A2:\n");
        printMatrix(Q + (glob_cols * glob_rows), glob_cols, check_rows);
    }
    
    /************************ PBMGS **********************************/
    
    pbmgs(Q, R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    /***************** Get average timing ****************************/

    /* End timer */
    t2 = MPI_Wtime() - t1;    

    /* Take average execution time */
    {
    double exec_time;
    MPI_Reduce(&t2, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t2 = exec_time / proc_size;
    }

    /********************* Check Results *****************************/

    E = (double*) calloc(loc_cols * loc_rows, sizeof(double));

    t3 = MPI_Wtime();
    
    error_norm = checkError(A, Q, R, E, loc_cols, loc_rows, glob_cols + check_cols, glob_rows + check_rows);  
    
    t4 = MPI_Wtime() - t3;   

    /* Take average checking time */
    {
    double exec_time;
    MPI_Reduce(&t4, &exec_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, glob_comm);
    t4 = exec_time / proc_size;
    }     
        
    /*********** Compile Q and R from local blocks *******************/

    gatherA(&A, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&Q, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&R, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);
    gatherA(&E, p_rank, proc_cols, proc_rows, loc_cols, loc_rows);

    if (p_rank == MASTER) {

        /* Print all matrices */
        if (glob_cols < 100 && glob_rows < 100) {
            printf("\n");
            printf("Matrix A1:\n");
            printMatrix(A, glob_cols, glob_rows);

            printf("Matrix A2:\n");
            printMatrix(A + (glob_cols * glob_rows), glob_cols, check_rows);

            printf("Matrix B1:\n");
            printMatrix(E, glob_cols, glob_rows);

            printf("Matrix B2:\n");
            printMatrix(E + (glob_cols * glob_rows), glob_cols, check_rows);

            printf("Matrix Q1:\n");
            printMatrix(Q, glob_cols, glob_rows);

            printf("Matrix Q2:\n");
            printMatrix(Q + (glob_cols * glob_rows), glob_cols, check_rows);

            printf("Matrix R:\n");
            printMatrix(R, glob_cols, glob_rows);
        }
    
        /* Print Stats */
        printf("Execution Time: %.5f s\n", t2);
        printf("Checking Time: %.5f s\n", t4);
        printf("Roundoff Error: %f\n", error_norm); 
    }

    if(p_rank == MASTER) printf("\n\n");
    fclose(fp_log);

    MPI_Comm_free(&glob_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Group_free(&row_group);
    MPI_Group_free(&col_group);

    MPI_Finalize();
}