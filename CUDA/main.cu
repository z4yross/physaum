#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "physarum.cu"

GLFWwindow *window;

GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

void render()
{
    uchar4 *d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);

    kernelLauncher(d_out);
    // printf("%d, %d", WIDTH, HEIGHT);
    // printf("--------------------------------------------");
    
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);

    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, HEIGHT);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(WIDTH, HEIGHT);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(WIDTH, 0);

    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display(){

    // render();
    // drawTexture();

    handleButtons();
    render();
    drawTexture();
}

void initPixelBuffer()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * WIDTH * HEIGHT * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,cudaGraphicsMapFlagsWriteDiscard);
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

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
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

void mainGL()
{
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
        
        display();

        glfwSwapBuffers(window);
        glfwPollEvents();

        double currentTime = glfwGetTime();

        if (begin)
        {
            double tpc = currentTime - previousTime;

            cicles++;
            sum += tpc;

            //printf("\r               ");
            printf("\rtime per cicle: %f", sum / cicles);
        }

        previousTime = currentTime;
    }
}

void clean()
{
    cleanModel();
    glfwTerminate();
}

int main(int argc, char *argv[])
{
    initGLFW();
    setupGL();
    initPixelBuffer();
    setupModel();
    mainGL();
    clean();
    return 0;
}
