extern "C" {

__constant__ double J[8];            // ≤8 оболочек

__global__ void bcc_update(
    unsigned char* spins,
    const unsigned int n_sites,
    const unsigned int n_shells,
    const double beta,
    const unsigned int n_sweeps)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sites) return;

    // простой XOR-шифт RNG
    unsigned int rng = 1664525u * (tid + 1);

    auto urand = [&]() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return (rng & 0x00FFFFFFu) / double(0x01000000u);
    };

    for (unsigned int sweep = 0; sweep < n_sweeps; ++sweep) {
        unsigned char old_s = spins[tid];
        unsigned char new_s = (old_s + 1 + (unsigned int)(urand()*253)) % 255 + 1;

        double dE = 0.0;
        // в Phase-3. Пока считаем только 1-ю оболочку J1 (bcc coord=8).
        for (int b = 0; b < 8; ++b) {
            // псевдо-вычисление соседей
            int nb = (tid + b + 1) % n_sites;
            unsigned char s_nb = spins[nb];
            if (s_nb == old_s) dE += J[0];
            if (s_nb == new_s) dE -= J[0];
        }

        if (dE <= 0.0 || urand() < exp(-beta * dE))
            spins[tid] = new_s;
    }
}

} // extern "C"
