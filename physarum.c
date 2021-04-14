#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    int x;
    int y;
    float angle;
} Agent;


GLFWwindow* window;

const double PI = 3.14159265358979323846;
const int diffK = 3;
const int nAgentes = 200;
const int SO = 6;
const int SS = 3;
const float SA = 22.5;
const float RA = 45.0;

int WIDTH = 0;
int HEIGHT = 0;

Agent *agents;
float *trail;

float *kernel(float *out){
    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){

            float sum = 0.0;

            for(int k = -diffK / 2; k <= diffK / 2; k++){
                for(int l = -diffK / 2; l <= diffK / 2; l++){
                    int x = i + k;
                    int y = j + l;
                    
                    int index = x * HEIGHT + y;

                    if(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT){
                        // printf("S f(%d, %d) k(%d, %d) | (%d, %d) => %d\n", i, j, k, l, x, y, index);
                        sum += out[index] / 9;
                    } 
                    // else printf("N f(%d, %d) k(%d, %d) | (%d, %d) => %d\n", i, j, k, l, x, y, index);
                    
                    
                        
                }
            }
            // printf("-----------------------\n");
            out[i * HEIGHT + j] = sum;
        }
    }
    return out;
}

float h(unsigned int x){
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x / 4294967295.0;
}

Agent sense(Agent a){
    float rad = a.angle * PI / 180;
    float sARad = SA * PI / 180;

    int fLX = a.x + SO * (int) cos(rad - SA);
    int fLY = a.y + SO * (int) sin(rad - SA);

    int fX = a.x + SO * (int) cos(rad);
    int fY = a.y + SO * (int) sin(rad);

    int fRX = a.x + SO * (int) cos(rad + SA);
    int fRY = a.y + SO * (int) sin(rad + SA);



    if(fLX >= 0 && fLX < WIDTH && fLY >= 0 && fLY < HEIGHT && 
        fX >= 0 && fX < WIDTH && fY >= 0 && fY < HEIGHT && 
        fRX >= 0 && fRX < WIDTH && fRY >= 0 && fRY < HEIGHT){

        float fl = trail[fLX * HEIGHT + fLY];
        float f = trail[fX * HEIGHT + fY];
        float fr = trail[fRX * HEIGHT + fRY];

        if (f > fl && f > fr) a.angle += 0;
        else if(f <= fl && f <= fr) 
            if(h(fX * HEIGHT + fY) <= 0.5) a.angle = (int) (a.angle - RA) % 360;
            else a.angle = (int) (a.angle + RA) % 360;
        else if(fl < fr) a.angle = (int) (a.angle + RA) % 360;
        else if(fr < fl) a.angle = (int) (a.angle - RA) % 360;
    }

    return a;
}



Agent move(Agent a, int i){
    float rad = a.angle * PI / 180.0;
    int px = round(a.x + SS * cos(rad));
    int py = round(a.y + SS * sin(rad));

    if(px <= 0 || px >= WIDTH - 1) {
        a.angle = (int) (acos(-cos(rad)) * 180 / PI) % 360; 
    } else if (py <= 0 || py >= HEIGHT - 1){
        a.angle = (int) (asin(-sin(rad)) * 180 / PI) % 360; 
    } else if (px >= WIDTH - 1 && py >= HEIGHT - 1){
        a.angle = 225;
    }

    if(px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT){
        a.x = px;
        a.y = py;
        trail[a.x * HEIGHT + a.y] = 1;
    }


    // printf("%d (%d %d %f) (%f %f)\n",i, a.x, a.y, a.angle, cos(rad), sin(rad));
    return a;
}

void setup(){
    glfwGetWindowSize(window, &WIDTH, &HEIGHT);
    glEnable(GL_BLEND);
    glClearColor(0, 0, 0, 1);

    glMatrixMode(GL_PROJECTION);
    glOrtho(0.0, WIDTH, HEIGHT, 0.0, 100, -100);

    //i * height + j
    agents = (Agent*) calloc(nAgentes, sizeof(Agent));
    trail = (float *) calloc(WIDTH * HEIGHT, sizeof(float));

    for(int i = 0; i < nAgentes; i++) {
        int dx = i % (nAgentes / 2) - nAgentes/ 4;
        int dy = i / (nAgentes / 2) - nAgentes/ 4;
        Agent a;
        a.x = WIDTH / 2 + dx;
        a.y = HEIGHT / 2 + dy;
        // a.angle = 45;
        a.angle = (int) (h(i) * 360) % 360;
        agents[i] = a;
    }
}

void draw(){
    

    trail = kernel(trail);

    for(int i = 0; i < nAgentes; i++){
        Agent a = agents[i];    
        agents[i] = move(agents[i], i);
        // agents[i] = sense(agents[i]);
    }

    glPointSize(1.0);
    glBegin(GL_POINTS);

    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            trail[i * HEIGHT + j] -= 0.001;
            float c = trail[i * HEIGHT + j];

            glColor4f(c, c, c, 1.0);
            glVertex2f(i, j);
            
        }
    }

    glEnd();



    // glBegin(GL_POINTS);
    //     glVertex2f(x, y);
    // glEnd();
}

int main(void){
    if (!glfwInit()) return -1;

    window = glfwCreateWindow(1000, 1000, "physarum", NULL, NULL);
    if (!window){
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    setup();

    while (!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT);

        draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}