#include <math.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <stdio.h>
#include <time.h>

#include "agent.cu"

// 1 b 1 t
// 1 b 2 t 1.757163
// 1 b 4 t 0.971593
// 1 b 8 t 0.535290
// 1 b 16 t 0.272678
// 1 b 32 t 0.166751
// 1 b 64 t 0.100114
// 1 b 128 t 0.066683
// 1 b 256 t 0.033397

// 5 b 1 t 0.659117
// 5 b 2 t 0.367519
// 5 b 4 t 0.200212
// 5 b 8 t 0.128884
// 5 b 16 t 0.066888
// 5 b 32 t 0.033538
// 5 b 64 t 0.033623
// 5 b 128 t 0.033387
// 5 b 256 t 0.033380

// 10 b 1 t 0.334435
// 10 b 2 t 0.201202
// 10 b 4 t 0.100757
// 10 b 8 t 0.066805
// 10 b 16 t 0.033427
// 10 b 32 t 0.033694
// 10 b 64 t 0.033424
// 10 b 128 t 0.033341
// 10 b 256 t 0.033337

// 15 b 1 t 0.235213
// 15 b 2 t 0.133889
// 15 b 4 t 0.073308
// 15 b 8 t 0.066677
// 15 b 16 t 0.033366
// 15 b 32 t 0.033352
// 15 b 64 t 0.033348
// 15 b 128 t 0.033342
// 15 b 256 t 0.033342

// 20 b 1 t 0.197165
// 20 b 2 t 0.100239
// 20 b 4 t 0.066765
// 20 b 8 t 0.033386
// 20 b 16 t 0.033370
// 20 b 32 t 0.033365
// 20 b 64 t 0.033360
// 20 b 128 t 0.033355
// 20 b 256 t 0.033339

#define blockSize 256
#define blocks 20

const float p = 0.15;

int nAgentes;

float *trail;
float *ftrail;
int *food;

int leftButton;
int rightButton;
int begin = 0;
double cursorX;
double cursorY;

Agent *agts;

attrAgents *atrs;

float hash(unsigned int x)
{
    x ^= 2747636419;
    x *= 2747636419;
    x ^= x >> 16;
    x *= 2747636419;
    x ^= x >> 16;
    x *= 2747636419;
    return x / 4294967295.0;
}

__device__ int mod(int a, int b)
{
    int ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

__device__ float lerp(float a, float b, float f)
{
    return (a * (1.0 - f)) + (b * f);
}

__device__ float mx(float a, float b)
{
    return a >= b ? a : b;
}

__global__ void update(uchar4 *d_out, Agent *agts, int size, float *t, float *ft, attrAgents *atrs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = index; idx < size; idx += stride)
        agts[idx].update(t, ft, atrs, idx);
    
}

__global__ void blur(uchar4 *d_out, int W, int H, float *t, float *ft, int *fd)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int blurSize = 3;

    for (int idx = index; idx < W * H; idx += stride)
    {
        
        int i = idx % W;
        int j = idx / W;

        int ix = j * W + i;

        float sum = 0.0;
        float sum2 = 0.0;

        for (int k = 0; k < blurSize; k++)
        {
            for (int l = 0; l < blurSize; l++)
            {
                int x = i + (k - blurSize / 2);
                int y = j + (l - blurSize / 2);
                
                if (x >= 0 && x < W && y >= 0 && y < H)
                {
                    sum += t[y * W + x];
                    sum2 += ft[y * W + x];
                }
            }
        }

        float ov = t[ix];
        float br = sum / 9.0;
        float dv = lerp(ov, br, QV * dt);
        float dev = mx(0, dv - ES * dt);

        float ov2 = ft[ix];
        float br2 = sum2 / 9.0;
        float dv2 = lerp(ov2, br2, QVT * dt);
        float dev2 = mx(0, dv2 - EST * dt);

        t[ix] = dev;
        ft[ix] = dev2;
    }
}

__device__ char clip(float a){
    return a > 255 ? a : (a < 0 ? 0 : a);
}

__global__ void draw(uchar4 *d_out, int W, int H, float *t, float *ft, int *fd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = index; idx < W * H; idx += stride)
    {
        d_out[idx].x = clip(t[idx] * 255);
        d_out[idx].y = clip(t[idx] * 255);
        d_out[idx].z = clip(t[idx] * 255);
        d_out[idx].w = 255;

        if (fd[idx] == 1) ft[idx] = 10;

        if (ft[idx] > 0)
        {
            d_out[idx].x = clip(ft[idx] * 255);
            d_out[idx].y = 0;
            d_out[idx].z = clip(ft[idx] * 255);
            d_out[idx].w = 255;
        }
    }
}

__global__ void drawfood(uchar4 *d_out, int W, int H, int *fd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = index; idx < W * H; idx += stride)
    {
        d_out[idx].x = clip(fd[idx] * 255);
        d_out[idx].y = clip(fd[idx] * 255);
        d_out[idx].z = clip(fd[idx] * 255);
        d_out[idx].w = 255;
    }
}

void kernelLauncher(uchar4 *d_out)
{
    if (begin)
    {
        int size = WIDTH * HEIGHT * p;
        update<<<blocks, blockSize>>>(d_out, agts, size, trail, ftrail, atrs);

        blur<<<blocks, blockSize>>>(d_out, WIDTH, HEIGHT, trail, ftrail, food);
        draw<<<blocks, blockSize>>>(d_out, WIDTH, HEIGHT, trail, ftrail, food);
        cudaDeviceSynchronize();
    } else {
        drawfood<<<blocks, blockSize>>>(d_out, WIDTH, HEIGHT, food);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
        exit(-1);
    }
}

void handleButtons()
{
    if (rightButton == GLFW_PRESS) begin = !begin;

    if (leftButton == GLFW_PRESS)
    {
        if (cursorX < WIDTH && cursorX >= 0 && cursorY < HEIGHT && cursorY >= 0)
        {
            int i = (int)(cursorY * WIDTH + cursorX);
            food[i] = 1;
        }
    }
}

void setupModel()
{
    nAgentes = WIDTH * HEIGHT * p;

    cudaMallocManaged(&trail, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&ftrail, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&food, WIDTH * HEIGHT * sizeof(int));
    cudaMallocManaged(&atrs, nAgentes * sizeof(attrAgents));

    std::vector<Agent> agents;

    
    for (int i = 0; i < nAgentes; i++)
    {
        float x = WIDTH / 2 + (hash(random()) * 2 - 1) * WIDTH * 3 / 4;
        float y = HEIGHT / 2 + (hash(random()) * 2 - 1) * HEIGHT * 3 / 4;
        float angle = (int)(hash(random()) * 360);

        Agent a(x, y, angle, WIDTH, HEIGHT);

        atrs[i].x = x;
        atrs[i].y = y;
        atrs[i].angle = angle;
        
        agents.push_back(a);
    }    

    cudaMallocManaged(&agts, nAgentes * sizeof(Agent));
    
    agts = agents.data();
    agents.clear();
}

void cleanModel()
{
    
    cudaFree(trail);
    cudaFree(ftrail);
    cudaFree(food);
    cudaFree(agts);
    cudaFree(atrs);
}