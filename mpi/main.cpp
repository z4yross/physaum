// mpirun -np 6 main
#include <iostream>
#include <algorithm>
#include <mpi.h>
// #include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <math.h>

GLFWwindow *window;

const int WIDTH = 1000;
const int HEIGHT = 1000;

const double PI = 3.14159265358979323846;

const int diffK = 3;
const float p = 0.15;

int nAgentes;

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

// Agents
float *aX;
float *aY;
float *aAngle;
float *aTrail;

// struct{
//     float x;
//     float y;
//     float angle;
//     float trail;
// } Agent;

float *trailC;
float *ftrailC;

float *trail;
float *ftrail;
int *food;

int *rAgents;
int *rEnv;

int partialTrailSize;

int *partialTrail;
int *partialFtrail;

int leftButton;
int rightButton;
int begin = 0;
double cursorX;
double cursorY;

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

float lerp(float a, float b, float f)
{
    return (a * (1.0 - f)) + (b * f);
}

float mx(float a, float b)
{
    return a >= b ? a : b;
}

void draw()
{

    if (begin)
    {
        glPointSize(1.0);
        glBegin(GL_POINTS);

        for (int i = 0; i < WIDTH; i++)
        {
            for (int j = 0; j < HEIGHT; j++)
            {
                float c = trail[i * HEIGHT + j];

                glColor4f(c, c, c, 1);
                // glColor4f(ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], 1);

                // ftrail[i * HEIGHT + j] = fmax(0, ftrail[i * HEIGHT + j] - 1* dt);
                float ft = ftrail[i * HEIGHT + j];

                if (food[i * HEIGHT + j] == 1)
                    ftrail[i * HEIGHT + j] = 10;

                if (ft > 0)
                    glColor4f(ft > 1 ? 1 : ft, 0, ft > 1 ? 1 : ft, 1);

                glVertex2f(i, j);

                // trail[i * HEIGHT + j] = (trail[i * HEIGHT + j]  + ftrail[i * HEIGHT + j]) * 0.1;
            }
        }

        glEnd();
    }
    else
    {
        glPointSize(1.0);
        glBegin(GL_POINTS);

        for (int i = 0; i < WIDTH; i++)
        {
            for (int j = 0; j < HEIGHT; j++)
            {
                float c = food[i * HEIGHT + j];

                glColor4f(c, c, c, 1);
                glVertex2f(i, j);
            }
        }

        glEnd();
    }
}

void blur(int rank)
{
    const int blurSize = diffK;

    for (int idx = 0; idx < partialTrailSize; idx++)
    {

        int i = (idx + (partialTrailSize * rank)) % WIDTH;
        int j = (idx + (partialTrailSize * rank)) / WIDTH;

        int ix = i * WIDTH + j;
        // int ix = i * WIDTH + j;

        float sum = 0.0;
        float sum2 = 0.0;

        for (int k = 0; k < blurSize; k++)
        {
            for (int l = 0; l < blurSize; l++)
            {
                int x = i + (k - blurSize / 2);
                int y = j + (l - blurSize / 2);

                int a = (y * WIDTH + x)  - (partialTrailSize * rank);

                if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
                {
                    sum += partialTrail[a];
                    sum2 += partialFtrail[a];
                }
            }
        }

        float ov = partialTrail[idx];
        float br = sum / 9.0;
        float dv = lerp(ov, br, QV * dt);
        float dev = mx(0, dv - ES * dt);

        float ov2 = partialFtrail[idx];
        float br2 = sum2 / 9.0;
        float dv2 = lerp(ov2, br2, QVT * dt);
        float dev2 = mx(0, dv2 - EST * dt);

        partialTrail[idx] = dev;
        partialFtrail[idx] = dev2;
    }
}

float sense(float angle, float *mv, int idx)
{
    float rad = (aAngle[idx] + angle) * PI / 180;

    int sCX = aX[idx] + SO * cos(rad);
    int sCY = aY[idx] + SO * sin(rad);

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
                    aTrail[idx] = TD;
                    *mv = mx(*mv, ftV);
                }
            }
        }
    }

    return sum;
}

void move(int idx)
{
    float rad = aAngle[idx] * PI / 180.0;

    float vx = SS * cos(rad) * dt;
    float vy = SS * sin(rad) * dt;

    float px = aX[idx] + vx;
    float py = aY[idx] + vy;

    if (pow(px - WIDTH / 2, 2) + pow(py - HEIGHT / 2, 2) > ((WIDTH / 2) * 3 / 4) * ((HEIGHT / 2) * 3 / 4) || px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT)
        aAngle[idx] = (int)(hash(random()) * 360);
    else
    {
        aX[idx] = px;
        aY[idx] = py;

        trail[(int)((int)(px)*HEIGHT + (int)(py))] = 1;
    }

    aTrail[idx] = mx(0, aTrail[idx] - 1 * dt);
}

void update()
{
    for (int idx = 0; idx < nAgentes; idx++)
    {
        move(idx);

        float mvf = 0;
        float mvfl = 0;
        float mvfr = 0;

        float f = sense(0, &mvf, idx);
        float fl = sense(SA, &mvfl, idx);
        float fr = sense(-SA, &mvfr, idx);

        float rs = hash(rand());
        float ra = rs * RA * dt;

        if (f > fl && f > fr)
        {
            aAngle[idx] += 0;
            if (mvf > 0 && aTrail[idx] > 0)
                ftrail[(int)((int)aX[idx] * HEIGHT + (int)(aY[idx]))] = mvf;
        }
        else if (f < fl && f < fr)
            if (rand() <= 0.5)
            {
                aAngle[idx] = (int)(aAngle[idx] - ra);
                if (mvfr > 0 && aTrail[idx] > 0)
                    ftrail[(int)((int)aX[idx] * HEIGHT + (int)(aY[idx]))] = mvfr;
            }
            else
            {
                aAngle[idx] = (int)(aAngle[idx] + ra);
                if (mvfl > 0 && aTrail[idx] > 0)
                    ftrail[(int)((int)aX[idx] * HEIGHT + (int)(aY[idx]))] = mvfl;
            }
        else if (fl < fr)
        {
            aAngle[idx] = (int)(aAngle[idx] - ra);
            if (mvfr > 0 && aTrail[idx] > 0)
                ftrail[(int)((int)aX[idx] * HEIGHT + (int)(aY[idx]))] = mvfr;
        }
        else if (fr < fl)
        {
            aAngle[idx] = (int)(aAngle[idx] + ra);
            if (mvfl > 0 && aTrail[idx] > 0)
                ftrail[(int)((int)aX[idx] * HEIGHT + (int)(aY[idx]))] = mvfl;
        }
    }
}

void handleButtons()
{
    if (rightButton == GLFW_PRESS)
    {
        begin = !begin;
    }

    if (leftButton == GLFW_PRESS)
    {
        if (cursorX < WIDTH && cursorX >= 0 && cursorY < HEIGHT && cursorY >= 0)
        {
            int i = (int)(cursorX * HEIGHT + cursorY);
            food[i] = 1;
        }
    }
}

void setupModel()
{
    nAgentes = WIDTH * HEIGHT * p;

    trail = new float[WIDTH * HEIGHT];
    ftrail = new float[WIDTH * HEIGHT];
    food = new int[WIDTH * HEIGHT];

    aX = new float[nAgentes];
    aY = new float[nAgentes];
    aAngle = new float[nAgentes];
    aTrail = new float[nAgentes];

    for (int i = 0; i < nAgentes; i++)
    {
        float x = WIDTH / 2 + (hash(random()) * 2 - 1) * WIDTH * 3 / 4;
        float y = HEIGHT / 2 + (hash(random()) * 2 - 1) * HEIGHT * 3 / 4;
        float angle = (int)(hash(random()) * 360);

        aX[i] = x;
        aY[i] = y;
        aAngle[i] = angle;
    }
}

void initGLFW()
{
    if (!glfwInit())
        exit(-1);

    window = glfwCreateWindow(WIDTH, HEIGHT, "physarum", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    if (glfwRawMouseMotionSupported() == 0)
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glfwMakeContextCurrent(window);
}

void setupGL()
{
    glClearColor(0, 0, 0, 1);

    glViewport(0, 0, WIDTH, HEIGHT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void clean()
{
    // cleanModel();
    glfwTerminate();
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << 1 << std::endl;

    if (rank == 0)
    {
        initGLFW();
        std::cout << "1a-" << rank << std::endl;
        setupGL();
        std::cout << "1b-" << rank << std::endl;
        setupModel();
        std::cout << "1c-" << rank << std::endl;

        partialTrailSize = (WIDTH * HEIGHT) / size;
    }

    std::cout << "2-" << rank << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // std::cout << 1 << std::endl;

    // BCast ---------------------------

    // MPI_Bcast(&nAgentes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&partialTrailSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&partialEnvSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // std::cout << 2 << std::endl;

    // std::cout << 3 << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0)
    {
        trail = new float[WIDTH * HEIGHT];
        ftrail = new float[WIDTH * HEIGHT];
        food = new int[WIDTH * HEIGHT];
        // rEnv = new int[WIDTH * HEIGHT];

        // aX = new float[nAgentes];
        // aY = new float[nAgentes];
        // aAngle = new float[nAgentes];
        // aTrail = new float[nAgentes];
        // rAgents = new int[nAgentes];
    }

    trailC = new float[WIDTH * HEIGHT];
    ftrailC = new float[WIDTH * HEIGHT];

    partialTrail = new int[partialTrailSize];
    partialFtrail = new int[partialTrailSize];

    // MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Scatter(trail, partialTrailSize, MPI_FLOAT, partialTrail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Scatter(ftrail, partialTrailSize, MPI_FLOAT, partialFtrail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    while (true)
    {

        // MPI_Bcast(aX, nAgentes, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        // MPI_Bcast(aY, nAgentes, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        // MPI_Bcast(aAngle, nAgentes, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        // MPI_Bcast(aTrail, nAgentes, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        // MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            if (glfwWindowShouldClose(window))
                break;

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            leftButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            rightButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

            glClear(GL_COLOR_BUFFER_BIT);
            glfwGetCursorPos(window, &cursorX, &cursorY);

            handleButtons();
        }

        std::cout << "3-" << rank << std::endl;

        MPI_Bcast(&begin, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if (begin)
        {
            MPI_Bcast(trail, WIDTH * HEIGHT, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(ftrail, WIDTH * HEIGHT, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(food, WIDTH * HEIGHT, MPI_INT, 0, MPI_COMM_WORLD);

            // memcpy(trailC, trail, WIDTH * HEIGHT * sizeof(float));
            // memcpy(ftrailC, ftrail, WIDTH * HEIGHT * sizeof(float));

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Scatter(trail, partialTrailSize, MPI_FLOAT, partialTrail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Scatter(ftrail, partialTrailSize, MPI_FLOAT, partialFtrail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);

            blur(rank);

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Gather(partialTrail, partialTrailSize, MPI_FLOAT, trail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(partialFtrail, partialTrailSize, MPI_FLOAT, ftrail, partialTrailSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0)
                update();
        }

        std::cout << "4-" << rank << std::endl;

        if (rank == 0)
        {
            draw();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        clean();
    }

    MPI_Finalize();
    return 0;
}
