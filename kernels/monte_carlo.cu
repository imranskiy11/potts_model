#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA ядро: random-site update, n_sweeps, inner_loops=8 для большой нагрузки.
extern "C" {

__global__ void metropolis_kernel(
    unsigned char* states,     // [total_size]
    const unsigned int* rand_ids, // [n_sweeps * numThreads]
    const float* randoms,      // [n_sweeps * numThreads * 2]
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    unsigned int q,
    float temperature,
    unsigned int total_size,
    unsigned int n_sweeps
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int numThreads = gridDim.x * blockDim.x;
    if(tid >= numThreads) return;

    int neighbors[6][3] = {
        {1,0,0}, {-1,0,0},
        {0,1,0}, {0,-1,0},
        {0,0,1}, {0,0,-1}
    };
    // Дополнительный цикл для нагрузки
    const int inner_loops = 50;;

    for(unsigned int s=0; s< n_sweeps; s++){
        unsigned int gid= rand_ids[s*numThreads + tid];
        if(gid>= total_size) continue;

        unsigned int base= s*(numThreads*2) + tid*2;
        float r_select= randoms[base+0];
        float r_accept= randoms[base+1];

        unsigned char old_st= states[gid];
        unsigned int new_st= (unsigned int)(r_select * q);
        if(new_st== old_st) {
            continue;
        }

        // (x,y,z)
        unsigned int z= gid/(nx*ny);
        unsigned int tmp= z*(nx*ny);
        unsigned int rest= gid- tmp;
        unsigned int y= rest/nx;
        unsigned int x= rest%nx;

        // dE
        int dE=0;
        for(int loop=0; loop<inner_loops; loop++){
            int count_old=0, count_new=0;
            for(int i=0;i<6;i++){
                int xx=(int)x+ neighbors[i][0];
                int yy=(int)y+ neighbors[i][1];
                int zz=(int)z+ neighbors[i][2];
                if(xx>=0 && xx<(int)nx &&
                   yy>=0 && yy<(int)ny &&
                   zz>=0 && zz<(int)nz){
                    unsigned int idx_n= (zz*(ny)+yy)*nx + xx;
                    unsigned char stn= states[idx_n];
                    if(stn== old_st) count_old++;
                    if(stn==(unsigned char)new_st) count_new++;
                }
            }
            dE= count_old - count_new;
        }

        if(dE<=0){
            states[gid]= (unsigned char)new_st;
        } else {
            float p= __expf(-(float)dE / temperature);
            if(r_accept < p){
                states[gid]= (unsigned char)new_st;
            }
        }
    }
}

} // extern "C"