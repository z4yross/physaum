#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const double PI = 3.14159265358979323846;

const int SO = 5;
const int SS = 1;
const int SW = 1;
const float SA = 45.0;

const float RA = 45.0;

const float QV = 0.1;
const float ES = 0.01;

const float QVT = 0.8;
const float EST = 0.2;

const float TW = 9;
const int TD = 1;

const float dt = 1;

const int WIDTH = 1000;
const int HEIGHT = 1000;

struct attrAgents
{
    float x;
    float y;
    float angle;
    float trail;
};

class Agent
{

public:
    // int i;

    float x;
    float y;
    float angle;
    float trail;

    Agent(float x, float y, float angle, int W, int H)
    {
        this->x = x;
        this->y = y;
        this->angle = angle;
        // WIDTH = W;
        // HEIGHT = H;
    }

    __device__ float rnd()
    {
        curandState_t state;
        curand_init(0, 0, 0, &state);
        return (curand(&state) % 100000) / 100000;
    }

    __device__ float hash(unsigned int x)
    {
        x ^= 2747636419;
        x *= 2747636419;
        x ^= x >> 16;
        x *= 2747636419;
        x ^= x >> 16;
        x *= 2747636419;
        return x / 4294967295.0;
    }

    __device__ float mx(float a, float b)
    {
        return a >= b ? a : b;
    }

    __device__ float sense(float angle, float *mv, float *trail, float *ftrail, attrAgents *agts, int id)
    {

        float rad = (agts[id].angle + angle) * PI / 180;

        int sCX = agts[id].x + SO * cos(rad);
        int sCY = agts[id].y + SO * sin(rad);

        float sum = 0;

        for (int i = -SW; i <= SW; i++)
        {
            for (int j = -SW; j <= SW; j++)
            {
                int pX = sCX + i;
                int pY = sCY + j;

                if (pX >= 0 && pX < WIDTH && pY >= 0 && pY < HEIGHT)
                {

                    float tV = trail[(int)((int)pX * HEIGHT + (int)pY)];
                    float ftV = ftrail[(int)((int)pX * HEIGHT + (int)pY)];
                    sum += tV + ftV * TW;

                    if (ftV > 0)
                    {
                        agts[id].trail = TD;
                        *mv = mx(*mv, ftV);
                    }
                }
            }
        }

        // glEnd();

        return sum;
    }

    __device__ void move(float *trail, attrAgents *agts, int id)
    {
        float rad = agts[id].angle * PI / 180.0;

        float vx = SS * cos(rad) * dt;
        float vy = SS * sin(rad) * dt;

        float px = agts[id].x + vx;
        float py = agts[id].y + vy;

        if (pow(px - WIDTH / 2, 2) + pow(py - HEIGHT / 2, 2) > ((WIDTH / 2) * 3 / 4) * ((HEIGHT / 2) * 3 / 4) || px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT)
        {
            // a.x = fmin(WIDTH - 0.01, fmax(0, px));
            // a.y =  fmin(HEIGHT - 0.01, fmax(0, py));
            agts[id].angle = (int)(hash(rnd()) * 360);
            // a.angle = (int) 0;
        }
        else
        {
            agts[id].x = px;
            agts[id].y = py;
            // printf("%d(%f %f %f)\n", i, px, py, a.angle);

            // if (a.trail > 0)
            // ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = 1;

            trail[(int)((int)(px)*HEIGHT + (int)(py))] = 1;
        }

        agts[id].trail = mx(0, agts[id].trail - 1 * dt);
    }

    __device__ void update(float *trail, float *ftrail, attrAgents *agts, int id)
    {
        move(trail, agts, id);

        float mvf = 0;
        float mvfl = 0;
        float mvfr = 0;

        float f = sense(0, &mvf, trail, ftrail, agts, id);
        float fl = sense(SA, &mvfl, trail, ftrail, agts, id);
        float fr = sense(-SA, &mvfr, trail, ftrail, agts, id);

        float rs = hash(rnd());
        float ra = rs * RA * dt;
        // float ra = RA;

        if (f > fl && f > fr)
        {
            agts[id].angle += 0;
            if (mvf > 0 && agts[id].trail > 0)
                ftrail[(int)((int)agts[id].x * HEIGHT + (int)(agts[id].y))] = mvf;
        }
        else if (f < fl && f < fr)
            if (rnd() <= 0.5)
            {
                agts[id].angle = (int)(agts[id].angle - ra);
                if (mvfr > 0 && agts[id].trail > 0)
                    ftrail[(int)((int)agts[id].x * HEIGHT + (int)(agts[id].y))] = mvfr;
            }
            else
            {
                agts[id].angle = (int)(agts[id].angle + ra);
                if (mvfl > 0 && agts[id].trail > 0)
                    ftrail[(int)((int)agts[id].x * HEIGHT + (int)(agts[id].y))] = mvfl;
            }
        else if (fl < fr)
        {
            agts[id].angle = (int)(agts[id].angle - ra);
            if (mvfr > 0 && agts[id].trail > 0)
                ftrail[(int)((int)agts[id].x * HEIGHT + (int)(agts[id].y))] = mvfr;
        }
        else if (fr < fl)
        {
            agts[id].angle = (int)(agts[id].angle + ra);
            if (mvfl > 0 && agts[id].trail > 0)
                ftrail[(int)((int)agts[id].x * HEIGHT + (int)(agts[id].y))] = mvfl;
        }
    }
};