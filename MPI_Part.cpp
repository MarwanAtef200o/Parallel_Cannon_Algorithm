#include <iostream>
#include <mpi.h>
#include <math.h>
#include <cstdlib>
#include <ctime>
using namespace std;


void print_complete_matrix(int* matrix, int block_size, int n) {
	for (int row_block = 0; row_block < n; ++row_block) {
		for (int i = 0; i < block_size; ++i) {
			for (int col_block = 0; col_block < n; ++col_block) {
				int base_idx = (row_block * n + col_block) * block_size * block_size + i * block_size;
				for (int j = 0; j < block_size; ++j) {
					cout << matrix[base_idx + j] << " ";
				}
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

int main(int argc, char* argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


		// Check if the number of processes is a perfect square
		int sqrt_size = sqrt(size);
	if (sqrt_size * sqrt_size != size) {
		cerr << "Error: The number of processes must be a perfect square." << endl;
		MPI_Finalize();
		return 1;
	}

	srand(time(NULL));

	int n = sqrt(size);
	int block_size = atoi(argv[1]) / n;
	int* A = new int[block_size * block_size];
	int* B = new int[block_size * block_size];
	int* C = new int[block_size * block_size];
	for (int i = 0; i < block_size * block_size; i++) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
		C[i] = 0;
	}

	// Create Cartesian topology
	int dims[2] = { n, n };
	int periods[2] = { 1, 1 };
	int reorder = 0;
	MPI_Comm comm_cart;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_cart);

	// Get Cartesian coordinates for this process
	int coords[2];
	MPI_Cart_coords(comm_cart, rank, 2, coords);

	double start_time, end_time;

	// Perform initial alignment of blocks
	int rank_source, rank_dest;
	
	MPI_Cart_shift(comm_cart, 1, coords[0], &rank_source, &rank_dest);
	MPI_Sendrecv_replace(A, block_size * block_size, MPI_INT, rank_dest, 0, rank_source, 0, comm_cart, MPI_STATUS_IGNORE);

	MPI_Cart_shift(comm_cart, 0, coords[1], &rank_source, &rank_dest);
	MPI_Sendrecv_replace(B, block_size * block_size, MPI_INT, rank_dest, 0, rank_source, 0, comm_cart, MPI_STATUS_IGNORE);

	start_time = MPI_Wtime();

	// Perform local block multiplication
	for (int i = 0; i < block_size; i++)
		for (int j = 0; j < block_size; j++)
			for (int k = 0; k < block_size; k++)
				C[i * block_size + j] += A[i * block_size + k] * B[k * block_size + j];

	// Perform next block multiplication and add to partial result
	for (int l = 1; l < n; l++)
	{
		// Shift blocks of A and B
		MPI_Cart_shift(comm_cart, 1, 1, &rank_source, &rank_dest);
		MPI_Sendrecv_replace(A, block_size * block_size, MPI_INT, rank_dest, 0, rank_source, 0, comm_cart, MPI_STATUS_IGNORE);

		MPI_Cart_shift(comm_cart, 0, 1, &rank_source, &rank_dest);
		MPI_Sendrecv_replace(B, block_size * block_size, MPI_INT, rank_dest, 0, rank_source, 0, comm_cart, MPI_STATUS_IGNORE);

		for (int i = 0; i < block_size; i++)
			for (int j = 0; j < block_size; j++)
				for (int k = 0; k < block_size; k++)
					C[i * block_size + j] += A[i * block_size + k] * B[k * block_size + j];
	}

	end_time = MPI_Wtime();
	double elapsed_time = end_time - start_time;
	double max_elapsed_time;
	MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	// Synchronize the processes before printing the output
	MPI_Barrier(MPI_COMM_WORLD);

	// Gather all blocks of matrix C in the root process (rank 0)
	int* gathered_A = nullptr;
	int* gathered_B = nullptr;
	int* gathered_C = nullptr;
	if (rank == 0) {
		gathered_A = new int[n * n * block_size * block_size];
		gathered_B = new int[n * n * block_size * block_size];
		gathered_C = new int[n * n * block_size * block_size];
	}
	MPI_Gather(A, block_size * block_size, MPI_INT, gathered_A, block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(B, block_size * block_size, MPI_INT, gathered_B, block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(C, block_size * block_size, MPI_INT, gathered_C, block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);

	// Print the entire matrix C in the root process (rank 0)
	if (rank == 0) {
		cout << "Matrix A:" << endl;
		print_complete_matrix(gathered_A, block_size, n);
		cout << "Matrix B:" << endl;
		print_complete_matrix(gathered_B, block_size, n);
		cout << "Matrix C:" << endl;
		print_complete_matrix(gathered_C, block_size, n);

		cout << "Time taken: " << max_elapsed_time << " seconds" << endl;
	}

	delete[] A;
	delete[] B;
	delete[] C;
	if (rank == 0) {
		delete[] gathered_A;
		delete[] gathered_B;
		delete[] gathered_C;
	}
	MPI_Finalize();

}