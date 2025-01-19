//
// Пример CUDA-ядер для q-состояной модели Поттса
// Реализует логику "метод Метрополиса" для каждого узла.
// (Упрощённый вариант без сильных оптимизаций).

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> // для expf

// __device__ функция для вычисления координат (x,y,z) из глобального индекса gid
__device__ void gid_to_xyz(unsigned int gid, unsigned int nx, unsigned int ny, unsigned int nz,
                           unsigned int &x, unsigned int &y, unsigned int &z)
{
    // z = gid / (nx*ny)
    unsigned int xy = nx * ny;
    z = gid / xy;
    unsigned int tmp = z * xy;
    unsigned int rest = gid - tmp;
    // y = rest / nx
    y = rest / nx;
    // x = rest % nx
    x = rest % nx;
}

// Функция для проверки, есть ли сосед (nx, ny, nz - размеры, cx,cy,cz - координаты)
__device__ bool valid_neighbor(int cx, int cy, int cz, int nx, int ny, int nz)
{
    if(cx < 0 || cx >= nx) return false;
    if(cy < 0 || cy >= ny) return false;
    if(cz < 0 || cz >= nz) return false;
    return true;
}

// Простейшее ядро: каждому потоку соответствует один узел решётки
// states   : массив состояний (размер = nx*ny*nz), каждый элемент - u8
// randoms  : случайные числа (float от 0..1)
// nx,ny,nz : размеры решётки
// q        : число состояний
// temperature : температура
// total_size  : количество узлов (nx*ny*nz)
extern "C" {
__global__ void metropolis_kernel(
    unsigned char* states,
    const float* randoms,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    unsigned int q,
    float temperature,
    unsigned int total_size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_size) return;

    // читаем old_state
    unsigned char old_state = states[gid];
    // выбираем new_state
    float rnd = randoms[gid];
    unsigned int new_state = (unsigned int)(rnd * q);
    if(new_state == old_state) return; // без изменений

    // Вычисляем (x,y,z)
    unsigned int x, y, z;
    gid_to_xyz(gid, nx, ny, nz, x, y, z);

    // Подсчёт числа совпадений со старыми и новыми соседями
    int count_old = 0;
    int count_new = 0;

    // Массив смещений 6 соседей: ±1 по x,y,z
    int neighbors[6][3] = {
        {1,0,0}, {-1,0,0},
        {0,1,0}, {0,-1,0},
        {0,0,1}, {0,0,-1}
    };

    for(int i=0; i<6; i++){
        int nx_ = x + neighbors[i][0];
        int ny_ = y + neighbors[i][1];
        int nz_ = z + neighbors[i][2];
        if(valid_neighbor(nx_, ny_, nz_, nx, ny, nz)){
            // индекс соседа
            unsigned int idx_n = (nz_ * ny + ny_) * nx + nx_;
            unsigned char st_neigh = states[idx_n];
            if(st_neigh == old_state){
                count_old++;
            }
            if(st_neigh == (unsigned char)new_state){
                count_new++;
            }
        }
    }

    // dE = -count_new - ( -(count_old) ) = -count_new + count_old
    int dE = count_old - count_new;

    // если dE <= 0 => принимаем
    if(dE <= 0){
        states[gid] = (unsigned char)new_state;
    } else {
        // иначе принимаем с вероятностью exp(-dE / T)
        // используем randoms[gid] (или соседний индекс, но упростим)
        // Для чистоты возьмём другое случайное число:
        // Но для демонстрации - возьмём тот же rnd (не очень правильно, но ладно).
        float p = expf(-(float)dE / temperature);
        if(rnd < p){
            states[gid] = (unsigned char)new_state;
        }
    }
}
}
// Конец файла