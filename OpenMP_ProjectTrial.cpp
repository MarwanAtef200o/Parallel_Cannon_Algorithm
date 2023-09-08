#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

int main() {
    int n, p;
    cout << "Enter the size of the matrices (n): ";
    cin >> n;
    cout << "Enter the number of processes (p): ";
    cin >> p;

    int sqrt_p = sqrt(p);
    int block_size = n / sqrt_p;

    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n));

    srand(time(NULL));

    // Initializing A,B with random numbers
#pragma omp parallel for collapse(2) num_threads(p)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 5;
            B[i][j] = rand() % 5;
            C[i][j] = 0;
        }
    }

    // Printing matrices A and B
#pragma omp master
    {
        cout << "matrix A: " << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << A[i][j] << " ";
            }
            cout << endl;
        }

        cout << "matrix B: " << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << B[i][j] << " ";
            }
            cout << endl;
        }
    }

    double start_time, end_time;

#pragma omp parallel num_threads(p)
    {
        // Record the start time
#pragma omp master
        {
            start_time = omp_get_wtime();
        }

        int thread_id = omp_get_thread_num();
        int row = thread_id / sqrt_p;
        int col = thread_id % sqrt_p;

        for (int p = 0; p < sqrt_p; p++) {
            // Iterate over block size of matrix B instead of n
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    int temp = 0;
                    for (int k = 0; k < block_size; k++) {
                        temp += A[row * block_size + i][k + p * block_size] * B[k + p * block_size][(col * block_size + j)];
                    }
                    // No need for atomic operation as each thread is working on a unique block of matrix C
                    C[row * block_size + i][col * block_size + j] += temp;
                }
            }

            // Rotate blocks of matrices A and B
#pragma omp barrier
#pragma omp master
            {
                // Rotate matrix B
                for (int i = 0; i < block_size; i++) {
                    int temp = B[p * block_size + i][(col + 1) * block_size - 1];
                    for (int j = (col + 1) * block_size - 1; j > col * block_size; j--) {
                        B[p * block_size + i][j] = B[p * block_size + i][j - 1];
                    }
                    B[p * block_size + i][col * block_size] = temp;
                }
                // Rotate matrix A
                for (int i = 0; i < block_size; i++) {
                    int temp = A[(row + 1) * block_size - 1][p * block_size + i];
                    for (int j = (row + 1) * block_size - 1; j > row * block_size; j--) {
                        A[j][p * block_size + i] = A[j - 1][p * block_size + i];
                    }
                    A[row * block_size][p * block_size + i] = temp;
                }
            }
#pragma omp barrier

        }
#pragma omp master
        {
            // Record the end time
            end_time = omp_get_wtime();

            cout << "resultant matrix C: " << endl;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    cout << C[i][j] << " ";
                }
                cout << endl;
            }
        }
    }

    // Calculate and print the time taken
    double time_taken = end_time - start_time;
    cout << "Time taken for parallel Cannon's multiplication algorithm: " << time_taken << " seconds" << endl;

    return 0;
}