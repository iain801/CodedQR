#include <stdlib.h>
#include <math.h>


void ICGS(double* Q, double* R, size_t n, size_t m) {
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        
        // R[i,i] = ||Q[:,i]||
        for (j = 0; j < m; ++j) {
            R[i*n + i] += Q[j*n + i] * Q[j*n + i];
        }
        R[i*n + i] = sqrt(R[i*n + i]);

        //Set zero values to near-zero
        if (R[i*n + i] == 0) R[i*n + i] = 2.3e-308;
        // Normalize Q[:,i]
        for (j = 0; j < m; ++j)
            Q[j*n + i] /= R[i*n + i];

        // For upper triangular R[i, i+1:n]
        for (k = i+1; k < n; ++k) {
            //R[i,k] = Q[:,k] * Q[:,i]
            for (j = 0; j < m; ++j) {
                R[i*n + k] += Q[j*n + k] * Q[j*n + i];
            }
            //Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
            for (j = 0; j < m; ++j) {
                Q[j*n + k] -= R[i*n + k] * Q[j*n + i];
            }
        }
    }
}

// Input:  A[m,n], Q[m,n], R[n,n], n <= m
// Output: Q and R such that A = QR and Q is orthonormal.
// Complexity: O(n^3)
void BMGS(double* A, double* Q, double* R, size_t n, size_t m, size_t b)
{

    size_t i, j, k;
    double* Qbar = calloc(b * m, sizeof(double));
    double* Rbar = calloc(b * m, sizeof(double));

    // Copy A into Q
    for (i = 0; i < n * m; ++i) {
        Q[i] = A[i];
    }

    //Iterate down columns
    for (i = 0; i < n; i += b) {

        //Tall skinny QR decomposition for block
        for (k = 0; k < b; ++k) {
            for (j = 0; j < m; ++j) {
                Qbar[j*b + k] = Q[j*n + (i+k)];
            }
            for (j = 0; j < b; ++j) {
                Rbar[j*b + k] = R[(i+j)*n + (i+k)];
            }        
        }

        ICGS(Qbar, Rbar, b, m);

        for (k = 0; k < b; ++k) {
            for (j = 0; j < m; ++j) {
                Q[j*n + (i+k)] = Qbar[j*b + k];  
            }  
            for (j = 0; j < b; ++j) {
                R[(i+j)*n + (i+k)] = Rbar[j*b + k];
            }
        }

        /*************************************************/
        
        for (j = i + b; j < n; ++j) {
            for (k = 0; k < m; ++k) {
                R[i*n + j] += Q[k*n + j] * Q[k*n + i]; //Column-Wise all reduce
            }
            for (k = 0; k < m; ++k) {
                Q[k*n + j] -= Q[k*n + i] * R[i*n + j];
            }
        }
    }

    free(Qbar);
    free(Rbar);
}

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


int main() {
    const size_t n = 1000;
    const size_t m = 1000;
    const size_t b = 125;

    double* A = calloc(n * m, sizeof(double));
    double* Q = calloc(n * m, sizeof(double));
    double* R = calloc(n * n, sizeof(double));
    
    srand(0);
    for (size_t i = 0; i < n * m; ++i) {
        A[i] = (rand() % 10000 - 5000) / 100.0;
    }

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
    printf("Roundoff Error: %f\n\n", sum);

    if (n <= 10) {
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