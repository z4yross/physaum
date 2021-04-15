// make && ./physarum

#include <math.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct{
    float x;
    float y;
    float angle;
} Agent;

const int WIDTH = 1920;
const int HEIGHT = 1080;

GLFWwindow* window;

const double PI = 3.14159265358979323846;

const int diffK = 3;
const float p = 0.15;
int nAgentes = (int) (WIDTH * HEIGHT * p);
// int nAgentes = 5;
const int SO = 9;
const int SS = 1;
const float SA = 22.5;
const float RA = 45.0;
const float QV = 0.1;
const float ES = 0.01;
const int SW = 1;

const float dt = 0.5;



Agent *agents;
float *trail;


float hash(unsigned int x){
    x ^= 2747636419;
    x *= 2747636419;
    x ^= x >> 16;
    x *= 2747636419;
    x ^= x >> 16;
    x *= 2747636419;
    // x = ((x >> 16) ^ x) * 0x45d9f3b;
    // x = ((x >> 16) ^ x) * 0x45d9f3b;
    // x = (x >> 16) ^ x;
    return x / 4294967295.0;
}

int mod (int a, int b){
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}

float lerp(float a, float b, float f) {
    return (a * (1.0 - f)) + (b * f);
}

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
                        sum += out[index];
                    } 
                }
            }

            // printf("-----------------------\n");
            // out[i * HEIGHT + j] = sum / 9.0;

            float ov = out[i * HEIGHT + j];
            float br = sum / 9.0;
            float dv = lerp(ov, br, QV * dt);
            float dev = fmax(0, dv - ES * dt);


            out[i * HEIGHT + j] = dev;
        }
    }
    return out;
}


float sense(Agent a, float angle){
    float rad = (a.angle + angle) * PI / 180;

    float sdX = cos(rad);
    float sdY = sin(rad);

    int sCX = a.x + SO * cos(rad);
    int sCY = a.y + SO * sin(rad);

    float sum = 0;
    
    // glColor4f(0,0,1,1);
    // glPointSize(1.0);

    // glBegin(GL_POINTS);
    

    for(int i = -SW; i <= SW; i++){
        for(int j = -SW; j <= SW ; j++){
            int pX = sCX + i;
            int pY = sCY + j;

            if(pX >= 0 && pX < WIDTH && pY >= 0 && pY < HEIGHT){

                // glVertex2f(pX, pY);
                // glVertex2f(0, 0);
                // glVertex2f(WIDTH / 2, HEIGHT / 2);

                sum += trail[(int) ((int) pX * HEIGHT + (int) pY)];
            }
        }
    }

    
    // glEnd();
    

    return sum;
}



Agent move(Agent a, int i){
    float rad = a.angle * PI / 180.0;

    float vx = SS * cos(rad) * dt;
    float vy = SS * sin(rad) * dt;

    float px = a.x + vx;
    float py = a.y + vy;

    if(px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT){
        // a.x = fmin(WIDTH - 0.01, fmax(0, px));
        // a.y =  fmin(HEIGHT - 0.01, fmax(0, py));
        a.angle = (int) (hash(random()) * 360);     
        // a.angle = (int) 0;     
        
    }else{
        a.x = px;
        a.y = py;
        // printf("%d(%f %f %f)\n", i, px, py, a.angle);
        trail[(int)((int)(px) * HEIGHT + (int)(py))] = 1;
    }    

    
    return a;
}

void setup(){
    // glfwGetWindowSize(window, &WIDTH, &HEIGHT);
    // glEnable(GL_BLEND);
    glClearColor(0, 0, 0, 1);

    glViewport(0,0,WIDTH,HEIGHT);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //i * height + j
    agents = (Agent*) calloc(nAgentes, sizeof(Agent));
    trail = (float *) calloc(WIDTH * HEIGHT, sizeof(float));

    printf("(%d, %d)",WIDTH, HEIGHT);

    float r = WIDTH / 6;
    

    for(int i = 0; i < nAgentes; i++) {
        float angle =  hash(random()) * 2 * PI;
        float x = hash(random()) * r * cos(angle) + WIDTH / 2;
        float y = hash(random()) * r * sin(angle) + HEIGHT / 2;

        
        Agent a;

        a.x = x;
        a.y = y;
        a.angle = angle * 180 / PI;

        // a.x = WIDTH / 2;
        // a.y = HEIGHT / 2;
        // // a.angle = 45;
        // a.angle = (int) (hash(random()) * 360);
        agents[i] = a;
    }
}

void draw(){
    trail = kernel(trail);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPointSize(1.0);
    glBegin(GL_POINTS);
    
    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            float c = trail[i * HEIGHT + j];
            
            glColor4f(c, c, c, 1);
            glVertex2f(i, j);    
            trail[i * HEIGHT + j] = fmax(0, c - ES * dt);        
        }
    }
    

    glEnd();
 

    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);

    // glMatrixMode(GL_MODELVIEW);
    // glLoadIdentity();

    for(int i = 0; i < nAgentes; i++){
        // Agent a = agents[i];    
        agents[i] = move(agents[i], i);

        float f = sense(agents[i], 0);
        float fl = sense(agents[i], SA);
        float fr = sense(agents[i], -SA);

        float rs = hash(rand());
        float ra = rs * RA * dt;

        if (f > fl && f > fr) agents[i].angle += 0;
        else if(f < fl && f < fr) 
            if(rand() <= 0.5) agents[i].angle = (int) (agents[i].angle - ra);
            else agents[i].angle = (int) (agents[i].angle + ra);
        else if(fl < fr) agents[i].angle = (int) (agents[i].angle - ra);
        else if(fr < fl) agents[i].angle = (int) (agents[i].angle + ra);        
    }
    

}

int main(void){
    if (!glfwInit()) return -1;

    window = glfwCreateWindow(WIDTH, HEIGHT, "physarum", NULL, NULL);
    if (!window){
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    // glfwSwapInterval(1);
    setup();

    while (!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT);

        // glViewport(0,0,WIDTH,HEIGHT);
    


        
        draw();

        // glColor4f(1,1,1,1);
        // glPointSize(10.0);
        // glBegin(GL_POINTS);
        // glVertex2f(WIDTH, HEIGHT);
        // glVertex2f(0, 0);
        // glVertex2f(WIDTH / 2, HEIGHT / 2);
        // glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}