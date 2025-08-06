#include <cstdio>
#include <mpi.h>
#include <atlc/mpi.hpp>

int main() {

    int num_procs, proc_num;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    std::string my_hostname;
    int my_node_number;
    int my_node_local_rank;
    int node_count;

    atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
    
    fprintf(stderr, "Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n", proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    MPI_Finalize();

    return 0;
}