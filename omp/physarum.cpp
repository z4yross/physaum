#include <math.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <stdio.h>
#include <time.h>

#include <omp.h>

#include "agent.cpp"

// 1 time per cicle: 0.139648
// 2 time per cicle: 0.082068
// 4 time per cicle: 0.057957
// 8 time per cicle: 0.056052
// 16 time per cicle: 0.054859

const int CORES = 16;
const int GAP = 1;

GLFWwindow *window;

const int WIDTH = 1000;
const int HEIGHT = 1000;

const float QV = 0.1;
const float ES = 0.01;

const float QVT = 0.8;
const float EST = 0.2;

const float dt = 1;

const int diffK = 3;
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

std::vector<Agent> agents;

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

int mod(int a, int b)
{
    int ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

float lerp(float a, float b, float f)
{
    return (a * (1.0 - f)) + (b * f);
}

void *kernel(void *arg)
{
    int id = *((int *)arg);
    int idxI = ((WIDTH * HEIGHT) / CORES) * id;
    int idxF = ((WIDTH * HEIGHT) / CORES) * id + (((WIDTH * HEIGHT) / CORES)) - 1;

    for (int idx = idxI; idx < idxF; idx++)
    {
        int i = idx % WIDTH;
        int j = idx / WIDTH;

        float sum = 0.0;
        float sum2 = 0.0;

        for (int k = -diffK / 2; k <= diffK / 2; k++)
        {
            for (int l = -diffK / 2; l <= diffK / 2; l++)
            {
                int x = i + k;
                int y = j + l;

                int index = x * HEIGHT + y;

                if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
                {
                    sum += trail[index * GAP];
                    sum2 += ftrail[index * GAP];
                }
            }
        }

        // printf("-----------------------\n");
        // out[i * HEIGHT + j] = sum / 9.0;

        float ov = trail[(i * HEIGHT + j) * GAP];
        float br = sum / 9.0;
        float dv = lerp(ov, br, QV * dt);
        float dev = fmax(0, dv - ES * dt);

        float ov2 = ftrail[(i * HEIGHT + j) * GAP];
        float br2 = sum2 / 9.0;
        float dv2 = lerp(ov2, br2, QVT * dt);
        float dev2 = fmax(0, dv2 - EST * dt);

        trail[(i * HEIGHT + j) * GAP] = dev;

        ftrail[(i * HEIGHT + j) * GAP] = dev2;
    }
    return NULL;
}

void blur()
{
#pragma omp parallel num_threads(CORES)
    {
        int ID = omp_get_thread_num();
        kernel(&ID);
    }
}

void show()
{
    glPointSize(1.0);
    glBegin(GL_POINTS);

    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < HEIGHT; j++)
        {
            float c = trail[(i * HEIGHT + j) * GAP];

            glColor4f(c, c, c, 1);
            // glColor4f(ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], 1);

            // ftrail[i * HEIGHT + j] = fmax(0, ftrail[i * HEIGHT + j] - 1* dt);
            float ft = ftrail[(i * HEIGHT + j) * GAP];

            if (food[i * HEIGHT + j] == 1)
                ftrail[(i * HEIGHT + j) * GAP] = 10;

            if (ft > 0)
                glColor4f(ft, 0, ft, 1);

            glVertex2f(i, j);

            // trail[i * HEIGHT + j] = (trail[i * HEIGHT + j]  + ftrail[i * HEIGHT + j]) * 0.1;
        }
    }
    glEnd();
}

void *update(void *args)
{
    int id = *((int *)args);
    int idxI = (agents.size() / CORES) * id;
    int idxF = (agents.size() / CORES) * id + ((agents.size() / CORES)) - 1;

    for (int idx = idxI; idx < idxF; idx++)
    {
        agents.at(idx).update(trail, ftrail);
    }

    return NULL;
}

void update()
{
#pragma omp parallel num_threads(CORES)
    {
        int ID = omp_get_thread_num();
        update(&ID);
    }
    // #pragma omp parallel for
    //     for (long unsigned int i = 0; i < agents.size(); i++)
    //     {
    //         agents.at(i).update(trail, ftrail);
    //     }
}

void setup()
{
    // glfwGetWindowSize(window, &WIDTH, &HEIGHT);
    // glEnable(GL_BLEND);
    glClearColor(0, 0, 0, 1);

    glViewport(0, 0, WIDTH, HEIGHT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //i * height + j
    // agents = (Agent*) calloc(nAgentes, sizeof(Agent));
    trail = (float *)calloc(WIDTH * HEIGHT * GAP, sizeof(float));
    ftrail = (float *)calloc(WIDTH * HEIGHT * GAP, sizeof(float));
    food = (int *)calloc(WIDTH * HEIGHT, sizeof(int));

    printf("(%d, %d)", WIDTH, HEIGHT);

    // float r = WIDTH / 6;

    nAgentes = WIDTH * HEIGHT * p;
    for (int i = 0; i < nAgentes; i++)
    {
        //     float angle =  hash(random()) * 2 * PI;
        //     float x = hash(random()) * r * cos(angle) + WIDTH / 2;
        //     float y = hash(random()) * r * sin(angle) + HEIGHT / 2;

        // a.x = x;
        // a.y = y;
        // a.angle = angle * 180 / PI;
        // a.trail = 0;

        float x = WIDTH / 2 + (hash(random()) * 2 - 1) * WIDTH * 3 / 4;
        float y = HEIGHT / 2 + (hash(random()) * 2 - 1) * HEIGHT * 3 / 4;
        float angle = (int)(hash(random()) * 360);

        Agent a(x, y, angle, WIDTH, HEIGHT, GAP);
        agents.push_back(a);
    }
}

void draw()
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

        // if(cursorX < WIDTH && cursorX >= 0 && cursorY < HEIGHT && cursorY >= 0){
        //     float angle =  hash(random()) * 360;
        //     Agent a;
        //     a.x = (float) cursorX;
        //     a.y = (float) cursorY;
        //     a.angle = angle * 180 / PI;
        //     a.trail = 0;

        //     insertArray(&agents, a);
        // }
    }

    if (begin)
    {
        blur();
        show();
        update();
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

        // for(int i = 0; i < agents.used; i++){
        //     glColor4f(1, 1, 1, 1);
        //     glVertex2f(agents.array[i].x, agents.array[i].y);
        // }
        glEnd();
    }
}

int main(void)
{
    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(WIDTH, HEIGHT, "physarum", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    if (glfwRawMouseMotionSupported() == 0)
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glfwMakeContextCurrent(window);
    setup();

    double previousTime = glfwGetTime();

    double sum = 0;
    int cicles = 0;


    while (!glfwWindowShouldClose(window))
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        leftButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        rightButton = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

        glClear(GL_COLOR_BUFFER_BIT);

        glfwGetCursorPos(window, &cursorX, &cursorY);

        draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        double currentTime = glfwGetTime();

        if (begin)
        {
            double tpc = currentTime - previousTime;

            cicles++;
            sum += tpc;

            // printf("\r               ");
            printf("\rtime per cicle: %f", sum / cicles);
        }

        previousTime = currentTime;
    }

    agents.clear();
    free(trail);
    free(ftrail);
    free(food);
    glfwTerminate();
    return 0;
}

