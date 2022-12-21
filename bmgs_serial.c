#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

void printMatrix(double* matrix, size_t n, size_t m) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
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

#define DEBUG 0

// Input:  A[m,n], Q[m,n], R[n,n], n <= m
// Output: Q and R such that A = QR and Q is orthonormal.
// Complexity: O(n^3)
void BMGS(double* A, double* Q, double* R, size_t n, size_t m, size_t b)
{

    size_t i, j, k, l;
    double* Qbar = calloc(b * m, sizeof(double));
    double* Rbar = calloc(b * b, sizeof(double));

    /* Copy A into Q and clear R*/
    for (i = 0; i < n * m; ++i) {
        Q[i] = A[i];
        R[i % (n*n)] = 0;
    }

    /* Iterate down columns */
    for (i = 0; i < n; i += b) {
        //Tall skinny QR decomposition for block
 
        /* Copy Q & R into Qbar & Rbar */
        for (k = 0; k < b && i+k < n; ++k) {
            for (j = 0; j < m; ++j) {
                Qbar[j*b + k] = Q[j*n + (i+k)];
            }
            for (j = 0; j < b; ++j) {
                Rbar[j*b + k] = R[(i+j)*n + (i+k)];
            }        
        }

        if(DEBUG) {
            printf("Q_initial (%d,%d)\n", i/b, 0);
            printMatrix(Qbar, b, m);
        }

        /* Perform standard MGS on Qbar - Rbar sub-matrix */
        size_t x, y, z;
        for (x = 0; x < b; ++x) {
            // R[i,i] = ||Q[:,i]||
            for (y = 0; y < m; ++y) {
                Rbar[x*b + x] += Qbar[y*b + x] * Qbar[y*b + x];
            }
            Rbar[x*b + x] = sqrt(Rbar[x*b + x]);

            //Set zero values to near-zero
            if (Rbar[x*b + x] == 0) Rbar[x*b + x] = 2.3e-308;
            // Normalize Q[:,i]
            for (y = 0; y < m; ++y)
                Qbar[y*b + x] /= Rbar[x*b + x];

            // For upper triangular R[i, i+1:n]
            for (z = x+1; z < b; ++z) {
                //R[i,k] = Q[:,k] * Q[:,i]
                for (y = 0; y < m; ++y) {
                    Rbar[x*b + z] += Qbar[y*b + z] * Qbar[y*b + x];
                }
                //Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
                for (y = 0; y < m; ++y) {
                    Qbar[y*b + z] -= Rbar[x*b + z] * Qbar[y*b + x];
                }
            }
        }

        if(DEBUG) {
            printf("Q_reduced (%d,%d)\n", i/b, 0);
            printMatrix(Qbar, b, m);
        }

        /* Copy Qbar & Rbar back into Q & R */
        for (k = 0; k < b && i+k < n; ++k) {
            for (j = 0; j < m; ++j) {
                Q[j*n + (i+k)] = Qbar[j*b + k];  
            }  
            for (j = 0; j < b; ++j) {
                R[(i+j)*n + (i+k)] = Rbar[j*b + k];
            }
        }

        /*************************************************/

        /* j = from next block to end of matrix */
        for (j = i + b; j < n; ++j) {

            /* l = each row in the current block*/
            for (l = i; l < i + b; ++l) {

                /*  For all rows in block, column j, 
                    R[row l, col j] = Q[col j] dot Q[col l] */
                for (k = 0; k < m; ++k) {
                    R[l*n + j] += Q[k*n + j] * Q[k*n + l];
                }

                /*  Q[col j] reduced by Q[col i] * R[i,j] 
                    i.e. Q[col j] minus itself dot Q[col i]*/
                for (k = 0; k < m; ++k) {
                    Q[k*n + j] -= Q[k*n + l] * R[l*n + j];
                }
            }
        }
        /* NOTE: results of block [i, i+b) affect:
         *  - R: rows [i, i+b) in columns [i+b, n)
         *  - Q: columns [i+b, n) in all rows
         */
    }

    free(Qbar);
    free(Rbar);
}


int main(int argc, char** argv) {
    size_t n, m, b;

    if (argc == 4) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
        b = atoi(argv[3]);
    }
    else {
        printf ("Invalid Input, must have arguements: n m b \n");
        exit(1);
    }
    if (n > m) {
        printf ("Invalid Input, n cannot be greater than m\n");
        exit(1);
    }
    if (b > m) {
        printf ("Invalid Input, b cannot be greater than m\n");
        exit(1);
    }
    

    double* A = calloc(n * m, sizeof(double));
    double* Q = calloc(n * m, sizeof(double));
    double* R = calloc(n * n, sizeof(double));
    
    srand(0);
    randMatrix(A, n, m);

    clock_t start, end;
    printf("Starting...\n");
    start = clock();
    BMGS(A, Q, R, n, m, b);
    end = clock();

    //Compute Execution Time
    printf("Execution Time: %.1f ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Check error = A - QR (should be near 0)
    double* B = calloc(n * m, sizeof(double));
    double sum = 0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            B[i*n + j] = 0;
            for (size_t k = 0; k < n; k++) {
                B[i*n + j] += Q[i*n + k] * R[k*n + j];
            }

            sum += fabs(B[i*n+j] - A[i*n+j]);
        }
    }
    free(B);
    printf("Roundoff Error: %f\n", sum);

    if (n <= 10) {
        printf("\n");
        printf("Matrix A:\n");
        printMatrix(A, n, m);

        printf("Matrix Q:\n");
        printMatrix(Q, n, m);

        printf("Matrix R:\n");
        printMatrix(R, n, n);
    }

    free(A);
    free(Q);
    free(R);

    return 0;
}