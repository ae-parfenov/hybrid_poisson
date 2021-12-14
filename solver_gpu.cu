#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#include<vector>
using std::vector;

typedef double float_type;
typedef int shift_type;
typedef long int index_type;
#define MPI_FLOAT_TYPE MPI_DOUBLE

#include<mpi.h>
MPI_Comm MCW = MPI_COMM_WORLD;
int my_rank = 0;
int num_procs = 1;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define MAX_BSIZE 256
#define MAX_BUFF 1024
int block_size = 256;
int block_size_x = 16;
int block_size_y = 16;


void print_help() {
    printf("Params: M, N, Px, Py [, eps, max_it]\n");
    printf(" M, N   - int > 0 - numbers of elements along the X,Y axes\n");
    printf(" Px, Py - int > 0, Px * Py = number of processes, (M+1)/Px, (N+1)/Py >= 2\n");
    printf(" eps    - float > 0, optional (default = 1e-6) - min |w(k+1)-w(k)|\n");
    printf(" max_it - int > 0, optional (default = 15000) max num of iterations\n");
    printf("Flags: -r -l -g (-R -L -G)\n");
    printf(" -r/-R - print result\n");
    printf(" -l/-L - print CG log\n");
    printf(" -g/-G - use gpu\n");
    printf(" default: -R -L -g\n");
}


enum axis_type { OX = 0, OY = 1 };
enum direction_type { BOTTOM = 0, RIGHT = 1, TOP = 2, LEFT = 3 };
enum condition_type { DIRICHLET, MIXED, INTERFACE };

struct Timer {
    double _time;
    void operator() () { _time = MPI_Wtime(); }
    void operator() (double& t) { _time = MPI_Wtime() - _time; t += _time; _time = MPI_Wtime(); }
} timer;

double time_dot = 0, time_reduce = 0, time_axpby = 0;
double time_app_1 = 0, time_app_2 = 0, time_app_3 = 0;
double time_app_4 = 0, time_app_5 = 0, time_app_6 = 0;

typedef float_type (*function_1d)(float_type);
typedef float_type (*function_2d)(float_type, float_type);

typedef vector<float_type> grid_function_1d;

struct grid_function_2d {
    index_type M, N;
    vector<float_type> data;
    grid_function_2d() { }
    grid_function_2d(index_type M, index_type N) : M(M), N(N), data( (M+1)*(N+1) ) { }
    grid_function_2d(index_type M, index_type N, float_type z) : M(M), N(N), data( (M+1)*(N+1), z ) { }
    float_type* operator[](index_type i) { return &data[ i*(N+1) ]; }
    const float_type* operator[](index_type i) const { return &data[ i*(N+1) ]; }
};

/* --- CUDA --- */
float_type* gpu_alloc(int length) { float_type *d_x; cudaMalloc((void**) &d_x, length * sizeof(float_type)); return d_x; }
int gpu_get_buf_size(int vector_size) { int grid_size = (vector_size - 1) / block_size + 1; return grid_size > MAX_BUFF ? MAX_BUFF : grid_size; }
void gpu_clear(float_type* d_x) { cudaFree(d_x); }
void gpu_extract(int length, float_type* x, float_type* d_x) { cudaMemcpy(x, d_x, length * sizeof(float_type), cudaMemcpyDeviceToHost); }
void gpu_upload(int length, float_type* x, float_type* d_x) { cudaMemcpy(d_x, x, length * sizeof(float_type), cudaMemcpyHostToDevice); }

/* kernels */
__global__ void cuda_set_slice(int i0, int di, int count, float_type* d_M, int k0, float_type* d_x)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;
    int index = i0 + di * gid;

    while (gid < count) {
        d_M[index] = d_x[k0+gid];
        gid += shift;
        index += di * shift;
    }

    return;
}

__global__ void cuda_get_slice(int i0, int di, int count, float_type* d_M, int k0, float_type* d_x)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;
    int index = i0 + di * gid;

    while (gid < count) {
        d_x[k0+gid] = d_M[index];
        gid += shift;
        index += di * shift;
    }

    return;
}

__global__ void cuda_set_0(int count, float_type* d_x)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;

    while (gid < count) {
        d_x[gid] = 0;
        gid += shift;
    }

    return;
}

__global__ void cuda_axpby(int count, float_type a, float_type* d_x, float_type b, float_type* d_y)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;

    while (gid < count) {
        d_x[gid] = a*d_x[gid] + b*d_y[gid];
        gid += shift;
    }

    return;
}

__global__ void cuda_prod(int count, float_type* d_z, float_type* d_x, float_type* d_y)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;

    while (gid < count) {
        d_z[gid] = d_x[gid] * d_y[gid];
        gid += shift;
    }

    return;
}

__global__ void cuda_dot(int count, float_type* d_x, float_type* d_y, float_type* d_buf)
{
    __shared__ float_type rb[MAX_BSIZE];
    int lid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = blockDim.x * gridDim.x;

    rb[lid] = 0;
    while (gid < count) {
        rb[lid] += d_x[gid] * d_y[gid];
        gid += shift;
    }

    __syncthreads();
    shift = blockDim.x / 2;
    while (shift > 0) {
        if (lid < shift) rb[lid] += rb[lid + shift];
        shift /= 2;
        __syncthreads();
    }

    if (lid == 0) d_buf[blockIdx.x] = rb[0];
    return;
}

__global__ void cuda_inner_apply(
    int M, int N, float_type* d_u, float_type* d_w,
    float_type* d_kx, float_type* d_ky, float_type* d_q, float_type hx, float_type hy
) {
    int gidx = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int gidy;
    int shiftx = blockDim.x * gridDim.x;
    int shifty = blockDim.y * gridDim.y;
    int di = N+1;
    int dj = 1;
    /*  //  host version
    for (index_type i = 1; i < M; ++i)
            for (index_type j = 1; j < N; ++j) {
                float_type ddux = (ks[OX][i][j]*(u[i+1][j]-u[i][j]) - ks[OX][i-1][j]*(u[i][j]-u[i-1][j]))/hs[OX]/hs[OX];
                float_type dduy = (ks[OY][i][j]*(u[i][j+1]-u[i][j]) - ks[OY][i][j-1]*(u[i][j]-u[i][j-1]))/hs[OY]/hs[OY];
                w[i][j] = qs[i][j]*u[i][j] - (ddux+dduy);
            }
    */

    while (gidx < M) {
        gidy = 1 + threadIdx.y + blockIdx.y * blockDim.y;
        while (gidy < N) {

            int gid = gidx * di + gidy * dj;
            float_type ddux = (d_kx[gid]*(d_u[gid+di]-d_u[gid]) - d_kx[gid-di]*(d_u[gid]-d_u[gid-di]))/(hx*hx);
            float_type dduy = (d_ky[gid]*(d_u[gid+dj]-d_u[gid]) - d_ky[gid-dj]*(d_u[gid]-d_u[gid-dj]))/(hy*hy);
            d_w[gid] = d_q[gid] * d_u[gid] - (ddux + dduy);

            gidy += shifty;
        }
        gidx += shiftx;
    }

    return;
}

/* host interface */
void gpu_set_slice(int i0, int di, int count, float_type* d_M, int k0, float_type* d_x) {
    int grid_size = (count - 1) / block_size + 1;
    cuda_set_slice<<<grid_size, block_size>>>(i0, di, count, d_M, k0, d_x);
}

void gpu_get_slice(int i0, int di, int count, float_type* d_M, int k0, float_type* d_x) {
    int grid_size = (count - 1) / block_size + 1;
    cuda_get_slice<<<grid_size, block_size>>>(i0, di, count, d_M, k0, d_x);
}

void gpu_axpby(int count, float_type a, float_type* d_x, float_type b, float_type* d_y) {
    int grid_size = (count - 1) / block_size + 1;
    cuda_axpby<<<grid_size, block_size>>>(count, a, d_x, b, d_y);
}

void gpu_set_0(int count, float_type* d_x) {
    int grid_size = (count - 1) / block_size + 1;
    cuda_set_0<<<grid_size, block_size>>>(count, d_x);
}

void gpu_prod(int count, float_type* d_z, float_type* d_x, float_type* d_y) {
    int grid_size = (count - 1) / block_size + 1;
    cuda_prod<<<grid_size, block_size>>>(count, d_z, d_x, d_y);
}

void gpu_dot(int count, float_type* d_x, float_type* d_y, int buf_size, float_type* d_buf) {
    // int grid_size = (count - 1) / block_size + 1;
    cuda_dot<<<buf_size, block_size>>>(count, d_x, d_y, d_buf);
}

void gpu_inner_apply(
    int M, int N, float_type* d_u, float_type* d_w,
    float_type* d_kx, float_type* d_ky, float_type* d_q, float_type hx, float_type hy
) {
    int grid_size_x = (M-1) / block_size_x + 1;
    int grid_size_y = (N-1) / block_size_y + 1;
    cuda_inner_apply<<<dim3(grid_size_x,grid_size_y), dim3(block_size_x,block_size_y)>>>(
        M, N, d_u, d_w, d_kx, d_ky, d_q, hx, hy
    );
}
/* --- END CUDA --- */

float_type reduce(float_type r, MPI_Op op) {
    timer();

    float_type z = r;
    if (num_procs > 1)
        MPI_Allreduce(&r, &z, 1, MPI_FLOAT_TYPE, op, MCW);

    timer(time_reduce);

    return z;
}


float_type dot(int count, float_type* d_x, float_type* d_y, float_type* d_buf, vector<float_type>& buf) {

    timer();

    int size = buf.size();
    gpu_dot(count, d_x, d_y, size, d_buf);
    gpu_extract(size, buf.data(), d_buf);

    float_type z = 0;
    for (int k = 0; k < size; ++k)
        z += buf[k];

    timer(time_dot);

    return reduce(z, MPI_SUM);
}

void axpby(float_type a, grid_function_2d& x, float_type* d_x, float_type b, const grid_function_2d& y, float_type* d_y) {
    timer();

    int M = x.M; int N = x.N; int size = x.data.size();
    gpu_axpby(size, a, d_x, b, d_y);
    for (int k = 0; k <= M; ++k) { x[k][0] = a*x[k][0] + b*y[k][0]; x[k][N] = a*x[k][N] + b*y[k][N]; }
    for (int k = 1; k <  N; ++k) { x[0][k] = a*x[0][k] + b*y[0][k]; x[M][k] = a*x[M][k] + b*y[M][k]; }

    timer(time_axpby);
}


grid_function_1d projection(const function_1d& f, index_type K, float_type z0, float_type hz) {
    grid_function_1d gf = vector<float_type>(K+1);
    for (index_type k = 0; k <= K; ++k)
        gf[k] = f(z0 + k*hz);
    return gf;
}

grid_function_2d projection(
    const function_2d& f, index_type M, index_type N,
    float_type x0, float_type y0, float_type hx, float_type hy
) {
    grid_function_2d gf(M,N);
    for (index_type i = 0; i <= M; ++i)
        for (index_type j = 0; j <= N; ++j)
            gf[i][j] = f(x0 + i*hx, y0 + j*hy);
    return gf;
}


float_type dot(const grid_function_1d& x, const grid_function_1d& y) {
    index_type size = x.size();
    float_type z = 0;

    for (index_type k = 0; k < size; ++k)
            z += x[k] * y[k];

    return reduce(z, MPI_SUM);
}

float_type dot(const grid_function_2d& x, const grid_function_2d& y) {
        return dot(x.data, y.data);
}


void axpby(float_type a, grid_function_1d& x, float_type b, const grid_function_1d& y) {
        index_type size = x.size();

        for (index_type k = 0; k < size; ++k)
            x[k] = a*x[k] + b*y[k];
}

void axpby(float_type a, grid_function_2d& x, float_type b, const grid_function_2d& y) {
    axpby(a, x.data, b, y.data);
}


float_type max_abs(const grid_function_2d& v) {
    float_type z = 0;
    index_type size = v.data.size();
    for (index_type k = 0; k < size; ++k)
        if (fabs(v.data[k]) > z) z = fabs(v.data[k]);
    return reduce(z, MPI_MAX);
}


struct Task {

    struct boundary {
        float_type position;
        condition_type condition;
        function_1d g;
        float_type alpha;

        boundary() : alpha(0) { }
        boundary(float_type position, condition_type condition, function_1d g, float_type alpha = 1)
        : position(position), condition(condition), g(g), alpha(alpha) { }
    };

    function_2d k, q, f;
    vector<boundary> bounds;

    Task() : bounds(4) {    }
};


struct hTask {

    struct boundary {
        condition_type condition;
        grid_function_1d gs;
        float_type alpha;
        index_type i0, j0, count;
        shift_type dni, dnj, dli, dlj;
        axis_type axis;
    };

    struct gateway {
        int nb_rank, exchg_tag;
        vector<float_type> snd_buf, rcv_buf;
        MPI_Request snd_req, rcv_req;
        MPI_Status snd_stat, rcv_stat;
    };

    struct image {
        float_type *d_kx, *d_ky, *d_q, *d_B;
        float_type *d_buf_dt;
        vector<float_type> buf_dt;
        float_type *d_buf_bd;
        vector<float_type> buf_bd;
    };

    vector<direction_type> directions;
    vector<boundary> bounds;
    vector<gateway> gates;

    index_type M, N;
    index_type M_gl, N_gl;
    index_type i0, i1, j0, j1;
    int Px, Py, Pi, Pj;

    grid_function_2d fs, qs, Bs;
    vector<float_type> hs;
    vector< grid_function_2d > ks;

    hTask() : directions(4), bounds(4), gates(4), hs(2), ks(2) { }

    void init(const Task& task, index_type M_gl, index_type N_gl, int Px, int Py) {

        this->M_gl = M_gl;
        this->N_gl = N_gl;

        this->Px = Px;
        this->Py = Py;

        int coords[2];
        MPI_Cart_coords(MCW, my_rank, 2, coords);
        Pi = coords[0];
        Pj = coords[1];

        float_type x0 = task.bounds[LEFT].position;
        float_type y0 = task.bounds[BOTTOM].position;
        float_type x1 = task.bounds[RIGHT].position;
        float_type y1 = task.bounds[TOP].position;

        float_type hx = (x1 - x0) / M_gl;
        float_type hy = (y1 - y0) / N_gl;
        hs[OX] = hx; hs[OY] = hy;

        ++M_gl; ++N_gl;

        i0 = (M_gl / Px) * Pi + (Pi <= int(M_gl % Px) ? Pi : M_gl % Px);
        j0 = (N_gl / Py) * Pj + (Pj <= int(N_gl % Py) ? Pj : N_gl % Py);

        M = (M_gl / Px) + index_type(int(M_gl % Px) > Pi) - 1;
        N = (N_gl / Py) + index_type(int(N_gl % Py) > Pj) - 1;

        i1 = i0 + M;
        j1 = j0 + N;

        x0 = x0 + hs[OX] * i0;
        y0 = y0 + hs[OY] * j0;
        x1 = x0 + hs[OX] * M;
        y1 = y0 + hs[OY] * N;

        { // 2-dimensional

            qs = projection(task.q, M, N, x0, y0, hs[OX], hs[OY]);
            fs = projection(task.f, M, N, x0, y0, hs[OX], hs[OY]);

            ks[OX] = projection(task.k, M, N, x0 + 0.5*hs[OX], y0, hs[OX], hs[OY]);
            ks[OY] = projection(task.k, M, N, x0, y0 + 0.5*hs[OY], hs[OX], hs[OY]);

            Bs = grid_function_2d(M, N);
            for (index_type i = 1; i < M; ++i)
                for (index_type j = 1; j < N; ++j) {
                    float_type ddux = -(ks[OX][i][j] + ks[OX][i-1][j])/(hs[OX]*hs[OX]);
                    float_type dduy = -(ks[OY][i][j] + ks[OY][i][j-1])/(hs[OY]*hs[OY]);
                    Bs[i][j] = 1.0 / (qs[i][j] - (ddux+dduy));
                }
        }

        directions[0] = BOTTOM; directions[1] = RIGHT;
        directions[2] = TOP; directions[3] = LEFT;

        { // boundaries

            vector<axis_type> axes(4);
            vector<index_type> counts(4);

            vector<shift_type> dnis(4), dlis(4), dnjs(4), dljs(4);
            vector<index_type> i0s(4), j0s(4);
            vector<float_type> from(4), step(4);

            axes[0] = OY; counts[0] = M+1;  i0s[0] = 0; j0s[0] = 0;
            axes[1] = OX; counts[1] = N+1;  i0s[1] = M; j0s[1] = 0;
            axes[2] = OY; counts[2] = M+1;  i0s[2] = M; j0s[2] = N;
            axes[3] = OX; counts[3] = N+1;  i0s[3] = 0; j0s[3] = N;

            dnis[0] =  0; dnjs[0] = -1; dlis[0] =  1; dljs[0] =  0;
            dnis[1] =  1; dnjs[1] =  0; dlis[1] =  0; dljs[1] =  1;
            dnis[2] =  0; dnjs[2] =  1; dlis[2] = -1; dljs[2] =  0;
            dnis[3] = -1; dnjs[3] =  0; dlis[3] =  0; dljs[3] = -1;

            from[0] = x0; from[1] = y0; from[2] = x1; from[3] = y1;
            step[0] = hx; step[1] = hy; step[2] =-hx; step[3] =-hy;

            for (int r = 0; r < 4; ++r) {
                direction_type dir = directions[r];
                boundary& bound = bounds[dir];
                bound.condition = task.bounds[dir].condition;
                bound.alpha = task.bounds[dir].alpha;
                bound.axis = axes[r];
                bound.dni = dnis[r]; bound.dnj = dnjs[r];
                bound.dli = dlis[r]; bound.dlj = dljs[r];
                bound.i0 = i0s[r]; bound.j0 = j0s[r];
                bound.count = counts[r];
            }

            if (Pi > 0) bounds[LEFT].condition = INTERFACE;
            if (Pj > 0) bounds[BOTTOM].condition = INTERFACE;
            if (Pi < Px-1) bounds[RIGHT].condition = INTERFACE;
            if (Pj < Py-1) bounds[TOP].condition = INTERFACE;

            for (int r = 0; r < 4; ++r) {
                direction_type dir = directions[r];
                boundary& bound = bounds[dir];

                shift_type dni = bound.dni, dnj = bound.dnj;
                shift_type dli = bound.dli, dlj = bound.dlj;
                float_type hn = hs[bound.axis], hl = hs[!bound.axis];
                const grid_function_2d &kn = ks[bound.axis], &kl = ks[!bound.axis];

                if (bound.condition == DIRICHLET) {
                    bound.gs = projection(task.bounds[dir].g, bound.count, from[r], step[r]);
                    for (index_type k = 0; k < bound.count; ++k) {
                        index_type i = bound.i0 + k*bound.dli;
                        index_type j = bound.j0 + k*bound.dlj;

                        fs[i][j] = bound.gs[k];
                        Bs[i][j] = 1.0;
                    }
                } else if (bound.condition == MIXED) {
                    bound.gs = projection(task.bounds[dir].g, bound.count, from[r], step[r]);
                    for (index_type k = 1; k < bound.count-1; ++k) {
                        index_type i = bound.i0 + k*bound.dli;
                        index_type j = bound.j0 + k*bound.dlj;
                        index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                        index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;

                        fs[i][j] = bound.gs[k]/hn + 0.5*fs[i][j];
                        float_type ddul = -(kl[il+dli][jl+dlj] + kl[il][jl])/(hl*hl);
                        float_type dun = kn[in][jn]/hn;
                        Bs[i][j] = 1.0 / (dun/hn + (bound.alpha/hn + 0.5*qs[i][j]) - 0.5*ddul);
                    }
                } else if (bound.condition == INTERFACE) {
                    gateway& gate = gates[dir];
                    coords[0] = Pi + bound.dni; coords[1] = Pj + bound.dnj;
                    MPI_Cart_rank(MCW, coords, &gate.nb_rank);
                    gate.snd_buf = vector<float_type>(bound.count);
                    gate.rcv_buf = vector<float_type>(bound.count);
                    gate.exchg_tag = 0;
                    bound.gs = grid_function_1d(bound.count);
                    for (index_type k = 0; k < bound.count; ++k) {
                        float_type xk = x0 + hs[OX]*(bound.i0 + 0.5*bound.dni + k*bound.dli);
                        float_type yk = y0 + hs[OY]*(bound.j0 + 0.5*bound.dnj + k*bound.dlj);
                        bound.gs[k] = task.k(xk, yk);
                    }
                    for (index_type k = 1; k < bound.count-1; ++k) {
                        index_type i = bound.i0 + k*bound.dli;
                        index_type j = bound.j0 + k*bound.dlj;
                        index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                        index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;

                        float_type ddul = -(kl[il+dli][jl+dlj] + kl[il][jl])/(hl*hl);
                        float_type ddun = -(bound.gs[k] + kn[in][jn])/(hn*hn);
                        Bs[i][j] = 1.0 / (qs[i][j] - (ddul+ddun));
                    }
                }
            }

        }

        { // corners

            for (int r = 0; r < 4; ++r) {
                boundary& curr = bounds[directions[r]];
                boundary& next = bounds[directions[(r+1)%4]];

                float_type h1 = hs[curr.axis], h2 = hs[next.axis];
                const grid_function_2d &k1 = ks[curr.axis], &k2 = ks[next.axis];
                index_type i = next.i0, j = next.j0;
                shift_type d1i = curr.dni, d1j = curr.dnj;
                shift_type d2i = next.dni, d2j = next.dnj;
                index_type in1 = i - (1+d1i)/2, jn1 = j - (1+d1j)/2;
                index_type in2 = i - (1+d2i)/2, jn2 = j - (1+d2j)/2;

                if (curr.condition == MIXED && next.condition == MIXED) {

                    float_type du1 = k1[in1][jn1]/h1;
                    float_type du2 = k2[in2][jn2]/h2;
                    Bs[i][j] = 1.0 / (0.5*du1/h1 + 0.5*du2/h2 + (0.5*curr.alpha/h1 + 0.5*next.alpha/h2 + 0.25*qs[i][j]));
                    fs[i][j] = 0.25*fs[i][j] + 0.5*(curr.gs[curr.count-1]/h1 + next.gs[0]/h2);

                } else if (curr.condition == MIXED && next.condition == INTERFACE) {

                    float_type du1 = k1[in1][jn1]/h1;
                    float_type ddu2 = -(next.gs[0] + k2[in2][jn2])/h2/h2;
                    Bs[i][j] = 1.0 / (du1/h1 + (curr.alpha/h1 + 0.5*qs[i][j]) - 0.5*ddu2);
                    fs[i][j] = 0.5*fs[i][j] + curr.gs[curr.count-1]/h1;

                } else if (curr.condition == INTERFACE && next.condition == MIXED) {

                    float_type ddu1 = -(curr.gs[curr.count-1] + k1[in1][jn1])/h1/h1;
                    float_type du2 = k2[in2][jn2]/h2;
                    Bs[i][j] = 1.0 / (du2/h2 + (next.alpha/h2 + 0.5*qs[i][j]) - 0.5*ddu1);
                    fs[i][j] = 0.5*fs[i][j] + next.gs[0]/h2;

                } else if (curr.condition == INTERFACE && next.condition == INTERFACE) {

                    float_type ddu1 = -(curr.gs[curr.count-1] + k1[in1][jn1])/h1/h1;
                    float_type ddu2 = -(next.gs[0] + k2[in2][jn2])/(h2*h2);
                    Bs[i][j] = 1.0 / (qs[i][j] - (ddu1+ddu2));

                }
            }

        }
    }

    void apply(const grid_function_2d& u, grid_function_2d& w) {

        // sending
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            gateway& gate = gates[dir];

            if (bound.condition == INTERFACE) {
                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    gate.snd_buf[bound.count-1-k] = u[i][j];
                }
                MPI_Isend(gate.snd_buf.data(), bound.count, MPI_FLOAT_TYPE, gate.nb_rank, gate.exchg_tag, MCW, &gate.snd_req);
                MPI_Irecv(gate.rcv_buf.data(), bound.count, MPI_FLOAT_TYPE, gate.nb_rank, gate.exchg_tag, MCW, &gate.rcv_req);
                ++gate.exchg_tag;
            }
        }

        // inner
        for (index_type i = 1; i < M; ++i)
            for (index_type j = 1; j < N; ++j) {
                float_type ddux = (ks[OX][i][j]*(u[i+1][j]-u[i][j]) - ks[OX][i-1][j]*(u[i][j]-u[i-1][j]))/hs[OX]/hs[OX];
                float_type dduy = (ks[OY][i][j]*(u[i][j+1]-u[i][j]) - ks[OY][i][j-1]*(u[i][j]-u[i][j-1]))/hs[OY]/hs[OY];
                w[i][j] = qs[i][j]*u[i][j] - (ddux+dduy);
            }

        // boundaries
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            gateway& gate = gates[dir];

            shift_type dni = bound.dni, dnj = bound.dnj;
            shift_type dli = bound.dli, dlj = bound.dlj;
            float_type hn = hs[bound.axis], hl = hs[!bound.axis];
            const grid_function_2d &kn = ks[bound.axis], &kl = ks[!bound.axis];

            if (bound.condition == DIRICHLET) {

                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    w[i][j] = u[i][j];
                }

            } else if (bound.condition == MIXED) {

                for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                    index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;
                    float_type ddul = (kl[il+dli][jl+dlj]*(u[i+dli][j+dlj]-u[i][j]) - kl[il][jl]*(u[i][j]-u[i-dli][j-dlj]))/hl/hl;
                    float_type dun = kn[in][jn]*(u[i][j] - u[i-dni][j-dnj])/hn;
                    w[i][j] = dun/hn + (bound.alpha/hn + 0.5*qs[i][j])*u[i][j] - 0.5*ddul;
                }

            } else if (bound.condition == INTERFACE) {

                MPI_Wait(&gate.rcv_req, &gate.rcv_stat);
                for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                    index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;
                    float_type ddul = (kl[il+dli][jl+dlj]*(u[i+dli][j+dlj]-u[i][j]) - kl[il][jl]*(u[i][j]-u[i-dli][j-dlj]))/hl/hl;
                    float_type ddun = (bound.gs[k]*(gate.rcv_buf[k]-u[i][j]) - kn[in][jn]*(u[i][j]-u[i-dni][j-dnj]))/hn/hn;
                    w[i][j] = qs[i][j]*u[i][j] - (ddul+ddun);
                }
            }
        }

        // corners
        for (int r = 0; r < 4; ++r) {
            const boundary& curr = bounds[directions[r]];
            const boundary& next = bounds[directions[(r+1)%4]];
            gateway& gate1 = gates[directions[r]];
            gateway& gate2 = gates[directions[(r+1)%4]];

            float_type h1 = hs[curr.axis], h2 = hs[next.axis];
            const grid_function_2d &k1 = ks[curr.axis], &k2 = ks[next.axis];
            index_type i = next.i0, j = next.j0;
            shift_type d1i = curr.dni, d1j = curr.dnj;
            shift_type d2i = next.dni, d2j = next.dnj;
            index_type in1 = i - (1+d1i)/2, jn1 = j - (1+d1j)/2;
            index_type in2 = i - (1+d2i)/2, jn2 = j - (1+d2j)/2;

            if (curr.condition == MIXED && next.condition == MIXED) {

                float_type du1 = k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j])/h1;
                float_type du2 = k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j])/h2;
                w[i][j] = 0.5*du1/h1 + 0.5*du2/h2 + (0.5*curr.alpha/h1 + 0.5*next.alpha/h2 + 0.25*qs[i][j])*u[i][j];

            } else if (curr.condition == MIXED && next.condition == INTERFACE) {

                float_type du1 = k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j])/h1;
                float_type ddu2 = (next.gs[0]*(gate2.rcv_buf[0]-u[i][j]) - k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j]))/h2/h2;
                w[i][j] = du1/h1 + (curr.alpha/h1 + 0.5*qs[i][j])*u[i][j] - 0.5*ddu2;

            } else if (curr.condition == INTERFACE && next.condition == MIXED) {

                float_type ddu1 = (curr.gs[curr.count-1]*(gate1.rcv_buf[curr.count-1]-u[i][j]) - k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j]))/h1/h1;
                float_type du2 = k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j])/h2;
                w[i][j] = du2/h2 + (next.alpha/h2 + 0.5*qs[i][j])*u[i][j] - 0.5*ddu1;

            } else if (curr.condition == INTERFACE && next.condition == INTERFACE) {

                float_type ddu1 = (curr.gs[curr.count-1]*(gate1.rcv_buf[curr.count-1]-u[i][j]) - k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j]))/h1/h1;
                float_type ddu2 = (next.gs[0]*(gate2.rcv_buf[0]-u[i][j]) - k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j]))/h2/h2;
                w[i][j] = qs[i][j]*u[i][j] - (ddu1+ddu2);

            }
        }

    }

    void apply(grid_function_2d& u, grid_function_2d& w, float_type* d_u, float_type* d_w, image& Im) {

        timer();
        int di = N+1; int dj = 1;

        if (N > 1 && M > 1) { // get border-adjacent
            gpu_get_slice( di + dj, di, M-1, d_u, 1, Im.d_buf_bd ); // b
            gpu_get_slice( di + dj, dj, N-1, d_u, (M+1) + 1, Im.d_buf_bd ); // l
            gpu_get_slice( di + dj*(N-1), di, M-1, d_u, (M+N+2) + 1, Im.d_buf_bd ); // t
            gpu_get_slice( di*(M-1) + dj, dj, N-1, d_u, (2*M+N+3) + 1, Im.d_buf_bd ); // r
        }

        // sending
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            gateway& gate = gates[dir];

            if (bound.condition == INTERFACE) {
                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    gate.snd_buf[bound.count-1-k] = u[i][j];
                }
                MPI_Isend(gate.snd_buf.data(), bound.count, MPI_FLOAT_TYPE, gate.nb_rank, gate.exchg_tag, MCW, &gate.snd_req);
                MPI_Irecv(gate.rcv_buf.data(), bound.count, MPI_FLOAT_TYPE, gate.nb_rank, gate.exchg_tag, MCW, &gate.rcv_req);
                ++gate.exchg_tag;
            }
        }

        timer(time_app_1);

        if (N > 1 && M > 1) { // extract border-adjacent
            gpu_extract(2*(M+1)+2*(N+1), Im.buf_bd.data(), Im.d_buf_bd);
        }

        timer(time_app_2);

        // inner
        if (N > 1 && M > 1)
            gpu_inner_apply(
                M, N, d_u, d_w,
                Im.d_kx, Im.d_ky, Im.d_q, hs[OX], hs[OY]
            );

        if (N > 1 && M > 1) { // set border-adjacent on host
            for (int k = 1; k < M; ++k) { u[k][1] = Im.buf_bd[k]; u[k][N-1] = Im.buf_bd[k + (M+N+2)]; }
            for (int k = 1; k < N; ++k) { u[1][k] = Im.buf_bd[k + (M+1)]; u[M-1][k] = Im.buf_bd[k + (2*M+N+3)]; }
        }
        
        timer(time_app_3);

        // boundaries
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            gateway& gate = gates[dir];

            shift_type dni = bound.dni, dnj = bound.dnj;
            shift_type dli = bound.dli, dlj = bound.dlj;
            float_type hn = hs[bound.axis], hl = hs[!bound.axis];
            const grid_function_2d &kn = ks[bound.axis], &kl = ks[!bound.axis];

            if (bound.condition == DIRICHLET) {

                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    w[i][j] = u[i][j];
                }

            } else if (bound.condition == MIXED) {

                for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                    index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;
                    float_type ddul = (kl[il+dli][jl+dlj]*(u[i+dli][j+dlj]-u[i][j]) - kl[il][jl]*(u[i][j]-u[i-dli][j-dlj]))/hl/hl;
                    float_type dun = kn[in][jn]*(u[i][j] - u[i-dni][j-dnj])/hn;
                    w[i][j] = dun/hn + (bound.alpha/hn + 0.5*qs[i][j])*u[i][j] - 0.5*ddul;
                }

            } else if (bound.condition == INTERFACE) {

                MPI_Wait(&gate.rcv_req, &gate.rcv_stat);
                for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    index_type il = i - (1+dli)/2, jl = j - (1+dlj)/2;
                    index_type in = i - (1+dni)/2, jn = j - (1+dnj)/2;
                    float_type ddul = (kl[il+dli][jl+dlj]*(u[i+dli][j+dlj]-u[i][j]) - kl[il][jl]*(u[i][j]-u[i-dli][j-dlj]))/hl/hl;
                    float_type ddun = (bound.gs[k]*(gate.rcv_buf[k]-u[i][j]) - kn[in][jn]*(u[i][j]-u[i-dni][j-dnj]))/hn/hn;
                    w[i][j] = qs[i][j]*u[i][j] - (ddul+ddun);
                }
            }
        }

        // corners
        for (int r = 0; r < 4; ++r) {
            const boundary& curr = bounds[directions[r]];
            const boundary& next = bounds[directions[(r+1)%4]];
            gateway& gate1 = gates[directions[r]];
            gateway& gate2 = gates[directions[(r+1)%4]];

            float_type h1 = hs[curr.axis], h2 = hs[next.axis];
            const grid_function_2d &k1 = ks[curr.axis], &k2 = ks[next.axis];
            index_type i = next.i0, j = next.j0;
            shift_type d1i = curr.dni, d1j = curr.dnj;
            shift_type d2i = next.dni, d2j = next.dnj;
            index_type in1 = i - (1+d1i)/2, jn1 = j - (1+d1j)/2;
            index_type in2 = i - (1+d2i)/2, jn2 = j - (1+d2j)/2;

            if (curr.condition == MIXED && next.condition == MIXED) {

                float_type du1 = k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j])/h1;
                float_type du2 = k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j])/h2;
                w[i][j] = 0.5*du1/h1 + 0.5*du2/h2 + (0.5*curr.alpha/h1 + 0.5*next.alpha/h2 + 0.25*qs[i][j])*u[i][j];

            } else if (curr.condition == MIXED && next.condition == INTERFACE) {

                float_type du1 = k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j])/h1;
                float_type ddu2 = (next.gs[0]*(gate2.rcv_buf[0]-u[i][j]) - k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j]))/h2/h2;
                w[i][j] = du1/h1 + (curr.alpha/h1 + 0.5*qs[i][j])*u[i][j] - 0.5*ddu2;

            } else if (curr.condition == INTERFACE && next.condition == MIXED) {

                float_type ddu1 = (curr.gs[curr.count-1]*(gate1.rcv_buf[curr.count-1]-u[i][j]) - k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j]))/h1/h1;
                float_type du2 = k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j])/h2;
                w[i][j] = du2/h2 + (next.alpha/h2 + 0.5*qs[i][j])*u[i][j] - 0.5*ddu1;

            } else if (curr.condition == INTERFACE && next.condition == INTERFACE) {

                float_type ddu1 = (curr.gs[curr.count-1]*(gate1.rcv_buf[curr.count-1]-u[i][j]) - k1[in1][jn1]*(u[i][j]-u[i-d1i][j-d1j]))/h1/h1;
                float_type ddu2 = (next.gs[0]*(gate2.rcv_buf[0]-u[i][j]) - k2[in2][jn2]*(u[i][j]-u[i-d2i][j-d2j]))/h2/h2;
                w[i][j] = qs[i][j]*u[i][j] - (ddu1+ddu2);

            }
        }
        
        timer(time_app_4);

        { // load borders

            for (int k = 0; k <= M; ++k) { Im.buf_bd[k] = w[k][0]; Im.buf_bd[k + (M+N+2)] = w[k][N]; }
            for (int k = 0; k <= N; ++k) { Im.buf_bd[k + (M+1)] = w[0][k]; Im.buf_bd[k + (2*M+N+3)] = w[M][k]; }

            timer(time_app_5);
            
            gpu_upload(2*(M+1)+2*(N+1), Im.buf_bd.data(), Im.d_buf_bd);

            timer(time_app_6);
            
            gpu_set_slice( 0, di, M+1, d_w, 0, Im.d_buf_bd ); // b
            gpu_set_slice( 0, dj, N+1, d_w, M+1, Im.d_buf_bd ); // l
            gpu_set_slice( dj*N, di, M+1, d_w, M+N+2, Im.d_buf_bd ); // t
            gpu_set_slice( di*M, dj, N+1, d_w, 2*M+N+3, Im.d_buf_bd ); // r
        }

    }

    void precondition(const grid_function_2d& u, grid_function_2d& w) const {
        index_type size = Bs.data.size();
        for (index_type k = 0; k < size; ++k)
            w.data[k] = Bs.data[k] * u.data[k];
    }
    
    void precondition(const grid_function_2d& u, grid_function_2d& w, float_type* d_u, float_type* d_w, image& Im) const {
        int size = Bs.data.size();
        gpu_prod(size, d_w, d_u, Im.d_B);
        for (int k = 0; k <= M; ++k) { w[k][0] = u[k][0] * Bs[k][0]; w[k][N] = u[k][N] * Bs[k][N]; }
        for (int k = 1; k <  N; ++k) { w[0][k] = u[0][k] * Bs[0][k]; w[M][k] = u[M][k] * Bs[M][k]; }
    }

    float_type approx_error(const grid_function_2d& us_exact) {
        grid_function_2d Au(M,N);
        apply(us_exact, Au);
        axpby(1,Au,-1,fs);

        float_type max_res = 0;
        for (index_type i = 1; i < M; ++i)
            for (index_type j = 1; j < N; ++j)
                if (fabs(Au[i][j]) > max_res) max_res = fabs(Au[i][j]);

        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            float_type z = 0;
            for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    if (fabs(Au[i][j]) > z) z = fabs(Au[i][j]);
            }

            if (bound.condition == MIXED)
                z *= hs[bound.axis];
            if (z > max_res) max_res = z;
        }

        for (int r = 0; r < 4; ++r) {
            const boundary& curr = bounds[directions[r]];
            const boundary& next = bounds[directions[(r+1)%4]];
            index_type i = next.i0, j = next.j0;
            float_type z = fabs(Au[i][j]);

            if (curr.condition == MIXED && next.condition == MIXED)
                z *= 0.5 * (hs[curr.axis] + hs[next.axis]);
            else if (curr.condition == MIXED)
                z *= hs[curr.axis];
            else if (next.condition == MIXED)
                z *= hs[next.axis];
            if (z > max_res) max_res = z;
        }

        return reduce(max_res, MPI_MAX);
    }

    float_type edot(const grid_function_2d& x, const grid_function_2d& y,
                    float_type* d_x, float_type* d_y, float_type* d_buf, vector<float_type>& buf) {

        float_type sum = 0, border_diff = 0, corner_diff = 0;
        float_type h = hs[OX] * hs[OY];

        // inner
        int count = x.data.size();
        gpu_dot(count, d_x, d_y, buf.size(), d_buf);

        // boundaries
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            float_type z = 0;
            if (bound.condition == INTERFACE) continue;
            for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    z += x[i][j] * y[i][j];
            }
            border_diff += 0.5 * z;
        }

        // corners
        for (int r = 0; r < 4; ++r) {
            const boundary& curr = bounds[directions[r]];
            const boundary& next = bounds[directions[(r+1)%4]];
            index_type i = next.i0, j = next.j0;
            float_type z = x[i][j] * y[i][j];

            if (curr.condition == INTERFACE && next.condition == INTERFACE) continue;
            else if (curr.condition == INTERFACE || next.condition == INTERFACE) corner_diff += 0.5 * z;
            else corner_diff += 0.75 * z;
        }
        
        int size = buf.size();
        gpu_extract(size, buf.data(), d_buf);
        for (int k = 0; k < size; ++k)
            sum += buf[k];

        float_type z = reduce(sum - border_diff - corner_diff, MPI_SUM);
        return h*z;
    }
    
    float_type edot(const grid_function_2d& x, const grid_function_2d& y) {
        float_type inner_sum = 0, border_sum = 0, corner_sum = 0;
        float_type h = hs[OX] * hs[OY];

        // inner
        for (index_type i = 1; i < M; ++i)
            for (index_type j = 1; j < N; ++j)
                inner_sum += x[i][j] * y[i][j];

        // boundaries
        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            const boundary& bound = bounds[dir];
            float_type z = 0;
            for (index_type k = 1; k < bound.count-1; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    z += x[i][j] * y[i][j];
            }
            if (bound.condition == INTERFACE) border_sum += z;
            else border_sum += 0.5 * z;
        }

        // corners
        for (int r = 0; r < 4; ++r) {
            const boundary& curr = bounds[directions[r]];
            const boundary& next = bounds[directions[(r+1)%4]];
            index_type i = next.i0, j = next.j0;
            float_type z = x[i][j] * y[i][j];

            if (curr.condition == INTERFACE && next.condition == INTERFACE) corner_sum += z;
            else if (curr.condition == INTERFACE || next.condition == INTERFACE) corner_sum += 0.5 * z;
            else corner_sum += 0.25 * z;
        }

        float_type z = reduce(inner_sum + border_sum + corner_sum, MPI_SUM);
        return h*z;
    }

    grid_function_2d solve_cg(float_type eps = 1e-6, int max_iter = 150000, bool print_log = false, int* it_made = NULL) {

        grid_function_2d xs(M,N,0), ps(M,N,0);
        grid_function_2d rs(M,N), zs(M,N), qs(M,N);
        float_type rho, alpha, beta;
        float_type delta;
        int it = 0;

        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            boundary& bound = bounds[dir];

            if (bound.condition == DIRICHLET)
                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    xs[i][j] = bound.gs[k];
                }
        }

        apply(xs,rs);
        axpby(-1,rs,1,fs);
        rho = 1;

        do {
            precondition(rs,zs);
            beta = rho;
            rho = dot(rs,zs);
            axpby(rho/beta, ps, 1, zs);
            apply(ps,qs);
            alpha = rho / dot(ps, qs);
            axpby(1,xs,alpha,ps);
            axpby(1,rs,-alpha,qs);
            delta = alpha*sqrt(edot(ps,ps));

            ++it;
            if (print_log) {
                float_type l2 = sqrt(dot(rs,rs));
                if (my_rank == 0) { printf("%4d : %f : %f\n", it, delta, l2); fflush(stdout); }
            }

        } while (fabs(delta) > eps && it < max_iter);

        if (it_made != NULL)
            *it_made = it;

        return xs;
    }
    
    grid_function_2d solve_cg_gpu(float_type eps = 1e-6, int max_iter = 150000, bool print_log = false, int* it_made = NULL) {

        int size = (M+1) * (N+1);
        image Im;
        {
            Im.d_kx = gpu_alloc(size); Im.d_ky = gpu_alloc(size);
            Im.d_q = gpu_alloc(size); Im.d_B = gpu_alloc(size);
            gpu_upload(size, ks[OX].data.data(), Im.d_kx);
            gpu_upload(size, ks[OY].data.data(), Im.d_ky);
            gpu_upload(size, qs.data.data(), Im.d_q);
            gpu_upload(size, Bs.data.data(), Im.d_B);
            int buf_size = gpu_get_buf_size(size);
            Im.buf_dt = vector<float_type>(buf_size);
            Im.d_buf_dt = gpu_alloc(buf_size);
            Im.buf_bd = vector<float_type>(2*(M+1) + 2*(N+1));
            Im.d_buf_bd = gpu_alloc(2*(M+1) + 2*(N+1)  );
        }

        grid_function_2d xs(M,N,0), ps(M,N,0);
        grid_function_2d rs(M,N), zs(M,N), qs(M,N);

        float_type* d_x = gpu_alloc(size);
        float_type* d_p = gpu_alloc(size);
        float_type* d_r = gpu_alloc(size);
        float_type* d_z = gpu_alloc(size);
        float_type* d_q = gpu_alloc(size);
        
        float_type rho, alpha, beta;
        float_type delta;
        int it = 0;

        gpu_upload(size, fs.data.data(), d_p);

        for (int r = 0; r < 4; ++r) {
            direction_type dir = directions[r];
            boundary& bound = bounds[dir];

            if (bound.condition == DIRICHLET)
                for (index_type k = 0; k < bound.count; ++k) {
                    index_type i = bound.i0 + k*bound.dli;
                    index_type j = bound.j0 + k*bound.dlj;
                    xs[i][j] = bound.gs[k];
                }
        }

        gpu_upload(size, xs.data.data(), d_x);

        apply(xs,rs, d_x,d_r, Im);
        axpby(-1,rs,d_r,1,fs,d_p);
        rho = 1;
        
        gpu_set_0(size, d_p);

        do {
            precondition(rs,zs, d_r,d_z, Im);
            beta = rho;
            rho = dot(size, d_r, d_z, Im.d_buf_dt, Im.buf_dt);
            axpby(rho/beta, ps, d_p, 1, zs, d_z);
            apply(ps,qs, d_p,d_q, Im);
            alpha = rho / dot(size, d_p, d_q, Im.d_buf_dt, Im.buf_dt);
            axpby(1,xs,d_x, alpha,ps,d_p);
            axpby(1,rs,d_r,-alpha,qs,d_q);
            delta = alpha*sqrt(edot(ps,ps, d_p,d_p, Im.d_buf_dt, Im.buf_dt));

            ++it;
            if (print_log) {
                float_type l2 = sqrt(dot(size, d_r, d_r, Im.d_buf_dt, Im.buf_dt));
                if (my_rank == 0) { printf("%4d : %f : %f\n", it, delta, l2); fflush(stdout); }
            }

        } while (fabs(delta) > eps && it < max_iter);

        gpu_extract(size, xs.data.data(), d_x);

        if (it_made != NULL)
            *it_made = it;

        {
            gpu_clear(Im.d_kx); gpu_clear(Im.d_ky);
            gpu_clear(Im.d_q); gpu_clear(Im.d_B);
            gpu_clear(Im.d_buf_dt);
            gpu_clear(Im.d_buf_bd);
        }

        {
            gpu_clear(d_x);
            gpu_clear(d_p);
            gpu_clear(d_q);
            gpu_clear(d_r);
            gpu_clear(d_z);
        }

        return xs;
    }

    void print_vec(const grid_function_2d& v, index_type shift_i, index_type shift_j) {

        if (my_rank == 0) { printf("\n"); fflush(stdout); }
        MPI_Barrier(MCW);
        bool my_turn = false;

        for (index_type j = 0; j <= N_gl; j += shift_j) {
            for (index_type i = 0; i <= M_gl; i += shift_i) {
                if (i >= i0 && i <= i1 && (N_gl-j) >= j0 && (N_gl-j) <= j1) {
                    printf("%8.4f ", v[i-i0][N_gl-j-j0]);
                    fflush(stdout);
                    my_turn = true;
                } else my_turn = false;
                MPI_Barrier(MCW);
            }
            if (my_turn) { printf("\n"); fflush(stdout); }
            MPI_Barrier(MCW);
        }
    }

};

float_type u(float_type x, float_type y) { return 2/(1 + x*x + y*y); }
float_type k(float_type x, float_type /*y*/) { return 4 + x; }
float_type q(float_type /*x*/, float_type /*y*/) { return 1; }

float_type ux(float_type x, float_type y) { return -4*x/(1 + x*x + y*y)/(1 + x*x + y*y); }
float_type uy(float_type x, float_type y) { return -4*y/(1 + x*x + y*y)/(1 + x*x + y*y); }

float_type uxx(float_type x, float_type y) { return ( -4 + 16*x*x/(1 + x*x + y*y) )/(1 + x*x + y*y)/(1 + x*x + y*y); }
float_type uyy(float_type x, float_type y) { return ( -4 + 16*y*y/(1 + x*x + y*y) )/(1 + x*x + y*y)/(1 + x*x + y*y); }

float_type kx(float_type /*x*/, float_type /*y*/) { return 1; }
float_type ky(float_type /*x*/, float_type /*y*/) { return 0; }

float_type f(float_type x, float_type y) {
    return q(x,y)*u(x,y) - k(x,y)*(uxx(x,y)+uyy(x,y)) - (kx(x,y)*ux(x,y) + ky(x,y)*uy(x,y));
}

float_type edge_bottom(float_type x) {
    float_type y0 = -1;
    return u(x, y0);
}

float_type flow_right(float_type y) {
    float_type alpha = 1, x1 = 3;
    return alpha*u(x1, y) + k(x1, y) * ux(x1, y);
}

float_type flow_top(float_type x) {
    float_type alpha = 1, y1 = 4;
    return alpha*u(x, y1) + k(x, y1) * uy(x, y1);
}

float_type flow_left(float_type y) {
    float_type alpha = 1, x0 = -2;
    return alpha*u(x0, y) - k(x0, y) * ux(x0, y);
}


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MCW, &my_rank);
    MPI_Comm_size(MCW, &num_procs);
    cudaProfilerStart();

    float_type eps = 1e-6;
    int max_iter = 150000;
    bool print_log = false;
    bool print_res = false;
    bool use_gpu = true;

    index_type M = 0;
    index_type N = 0;

    int Px = 0;
    int Py = 0;

    bool valid_params = true;

    if (argc < 5) {
        valid_params = false;
        if (my_rank == 0)
            print_help();
    } else {
        vector<double> params;
        for (int k = 1; k < argc; ++k)
            if (argv[k][0] == '-' && argv[k][1] == 'l') print_log = true;  else
            if (argv[k][0] == '-' && argv[k][1] == 'L') print_log = false; else
            if (argv[k][0] == '-' && argv[k][1] == 'r') print_res = true;  else
            if (argv[k][0] == '-' && argv[k][1] == 'R') print_res = false; else
            if (argv[k][0] == '-' && argv[k][1] == 'g') use_gpu = true;  else
            if (argv[k][0] == '-' && argv[k][1] == 'G') use_gpu = false; else
            params.push_back(atof(argv[k]));
        int count = params.size();
        if (count < 4) valid_params = false;
        else {
            if (params[0] > 0) M = index_type(params[0]);
            if (params[1] > 0) N = index_type(params[1]);
            if (params[2] > 0) Px = index_type(params[2]);
            if (params[3] > 0) Py = index_type(params[3]);
        }
        if (count > 4) eps = params[4];
        if (count > 5) max_iter = int(params[5]);
    }

    if (valid_params) {
        if (my_rank == 0) {
            if (M <= 0 || N <= 0) printf("M, N must be int > 0\n");
            if (Px*Py != num_procs) printf("Px*Py must be = num of processors\n");
            if (Px <= 0 || Py <= 0) printf("Px, Py must be int > 0\n");
            if (Px > 0 && (M+1)/Px < 2) printf("(M+1)/Px must be >= 2\n");
            if (Py > 0 && (N+1)/Py < 2) printf("(N+1)/Py must be >= 2\n");
               if (eps <= 0) printf("eps must be float > 0");
               if (max_iter <= 0) printf("max_it must be int > 0");
        }

        if (M <= 0 || N <= 0 || eps <= 0 || max_iter <= 0 || Px*Py != num_procs
            || Px <= 0 || Py <= 0 || (M+1)/Px < 2 || (N+1)/Py < 2)
            valid_params = false;
    }

    if (!valid_params) {
        cudaProfilerStop();
        MPI_Finalize();
        return 0;
    }

    { // topology

        int dims[2], wrap[2];
        dims[0] = Px; dims[1] = Py;
        wrap[0] = wrap[1] = 0;
        int reorder = 1;
        MPI_Cart_create(MCW, 2, dims, wrap, reorder, &MCW);
        MPI_Comm_rank(MCW, &my_rank);

    }

    if (my_rank == 0) {
        printf("( %ld x %ld ) / ( %d x %d)", M, N, Px, Py);
        if (use_gpu) printf(" , gpu on\n"); else printf(" , gpu off\n");
        printf("max_it = %d , eps = %g\n\n", max_iter, eps);
        fflush(stdout);
    }

    float_type x0 = -2, x1 = 3, y0 = -1, y1 = 4;
    float_type hx = (x1-x0)/M, hy = (y1-y0)/N;

    Task task;
    {
        task.k = k;
        task.q = q;
        task.f = f;
        
        task.bounds[BOTTOM] = Task::boundary(y0, DIRICHLET, edge_bottom);
        task.bounds[RIGHT]  = Task::boundary(x1, MIXED, flow_right);
        task.bounds[TOP]    = Task::boundary(y1, MIXED, flow_top);
        task.bounds[LEFT]   = Task::boundary(x0, MIXED, flow_left);
    }

    hTask htask;
    htask.init(task, M, N, Px, Py);

    {
        clock_t t_start, t_end;
        double mpi_time;
        int it_made = 1;

        if (my_rank == 0) {
            printf("\n-- CG start --\n");
            fflush(stdout);
        }

        mpi_time = MPI_Wtime();
        t_start = clock();
        
        grid_function_2d ws;
        if (use_gpu) ws = htask.solve_cg_gpu(eps, max_iter, print_log, &it_made);
        else ws = htask.solve_cg(eps, max_iter, print_log, &it_made);

        t_end = clock();
        mpi_time = MPI_Wtime() - mpi_time;

        if (my_rank == 0) {
            double t = double(t_end - t_start) / double(CLOCKS_PER_SEC);
            printf("\n");
            printf("Iterations : %d\n", it_made);
            printf("Total time : %f s | %f s\n", t, mpi_time);
            printf("Averg time : %f s | %f s\n", t/it_made, mpi_time/it_made);
            fflush(stdout);
        }

        if (my_rank == 0 && use_gpu) {
            printf("\n");
            printf("dot   : %f s | % f s\n", time_dot, time_dot/(2*it_made));
            printf("reduce: %f s | % f s\n", time_reduce, time_reduce/(2*it_made));
            printf("axpby : %f s | % f s\n", time_axpby, time_axpby/(3*it_made + 1));
            
            printf("app1  : %f s | % f s\n", time_app_1, time_app_1/(1+it_made));
            printf("app2  : %f s | % f s\n", time_app_2, time_app_2/(1+it_made));
            printf("app3  : %f s | % f s\n", time_app_3, time_app_3/(1+it_made));
            printf("app4  : %f s | % f s\n", time_app_4, time_app_4/(1+it_made));
            printf("app5  : %f s | % f s\n", time_app_5, time_app_5/(1+it_made));
            printf("app6  : %f s | % f s\n", time_app_6, time_app_6/(1+it_made));

            fflush(stdout);
        }

        grid_function_2d us = projection(u, htask.M, htask.N, x0 + hx*htask.i0, y0 + hy*htask.j0, hx, hy);
        grid_function_2d rs = us;
        axpby(1,rs,-1,ws);

        float_type c_norm = max_abs(rs);
        float_type e_norm = htask.edot(rs,rs);
        float_type ap_err = htask.approx_error(us);

        if (my_rank == 0) {
            printf("\n");
            printf("Delta C-norm : %f\n", c_norm);
            printf("Delta E-norm : %f\n", e_norm);
            printf("Approx Error : %f\n", ap_err);
            fflush(stdout);
        }

        if (print_res) {
            index_type shift_i = 1;
            index_type shift_j = 1;
            if (M > 5) shift_i = M / 5;
            if (N > 5) shift_j = N / 5;

            if (my_rank == 0) { printf("\nExact solution:\n"); fflush(stdout); }
            htask.print_vec(us, shift_i, shift_j);

            if (my_rank == 0) { printf("\nApprx solution:\n"); fflush(stdout); }
            htask.print_vec(ws, shift_i, shift_j);
        }
    }

    cudaProfilerStop();
    MPI_Finalize();
    return 0;
}
