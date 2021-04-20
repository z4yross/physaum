// make && ./physarum

#include <math.h>
#include <GLFW/glfw3.h>
#include "Array.c"


GLFWwindow* window;

const int WIDTH = 200;
const int HEIGHT = 200;

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

Array agents;
// Agent *agents;
float *trail;
float *ftrail;
int *food;

int leftButton;
int rightButton;
int begin = 0;
double cursorX;
double cursorY;


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
            float sum2 = 0.0;

            for(int k = -diffK / 2; k <= diffK / 2; k++){
                for(int l = -diffK / 2; l <= diffK / 2; l++){
                    int x = i + k;
                    int y = j + l;
                    
                    int index = x * HEIGHT + y;

                    if(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT){
                        sum += out[index];
                        sum2 += ftrail[index];
                    } 
                }
            }

            // printf("-----------------------\n");
            // out[i * HEIGHT + j] = sum / 9.0;

            float ov = out[i * HEIGHT + j];
            float br = sum / 9.0;
            float dv = lerp(ov, br, QV * dt);
            float dev = fmax(0, dv - ES * dt);

            float ov2 = ftrail[i * HEIGHT + j];
            float br2 = sum2 / 9.0;
            float dv2 = lerp(ov2, br2, QVT * dt);
            float dev2 = fmax(0, dv2 - EST * dt);

            out[i * HEIGHT + j] = dev;
            ftrail[i * HEIGHT + j] = dev2;
        }
    }
    return out;
}


float sense(Agent a, float angle, int id, float *mv){
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
                float tV = trail[(int) ((int) pX * HEIGHT + (int) pY)];
                float ftV = ftrail[(int) ((int) pX * HEIGHT + (int) pY)];
                sum += tV + ftV * TW; 

                if (ftV > 0){
                    agents.array[id].trail = TD;
                    *mv = fmax(*mv, ftV);
                }
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

    if(pow(px - 100.0, 2) + pow(py - 100.0, 2) > 90.0 * 90.0  || px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT){
        // a.x = fmin(WIDTH - 0.01, fmax(0, px));
        // a.y =  fmin(HEIGHT - 0.01, fmax(0, py));
        a.angle = (int) (hash(random()) * 360);     
        // a.angle = (int) 0;     
        
    }else{
        a.x = px;
        a.y = py;
        // printf("%d(%f %f %f)\n", i, px, py, a.angle);

        // if (a.trail > 0)
            // ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = 1; 

        trail[(int)((int)(px) * HEIGHT + (int)(py))] = 1;
    }    

    agents.array[i].trail = fmax(0, agents.array[i].trail - 1 * dt);
    
    return a;
}

void blur(){
    trail = kernel(trail);
}

void show(){    
    glPointSize(1.0);
    glBegin(GL_POINTS);

    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            float c = trail[i * HEIGHT + j];
            
            glColor4f(c, c, c, 1);
            // glColor4f(ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], ftrail[i * HEIGHT + j], 1);

             
            // ftrail[i * HEIGHT + j] = fmax(0, ftrail[i * HEIGHT + j] - 1* dt); 
            float ft = ftrail[i * HEIGHT + j];

            if (food[i * HEIGHT + j]  == 1) ftrail[i * HEIGHT + j] = 10;

            if(ft > 0)
                glColor4f(ft, 0, ft, 1); 
            
            glVertex2f(i, j); 
            
            

            // trail[i * HEIGHT + j] = (trail[i * HEIGHT + j]  + ftrail[i * HEIGHT + j]) * 0.1;
                             
        }
    }
    

    glEnd();
}

void update(){
    for(int i = 0; i < agents.used; i++){
  
        agents.array[i] = move(agents.array[i], i);

        float mvf = 0;
        float mvfl = 0;
        float mvfr = 0;

        float f = sense(agents.array[i], 0, i, &mvf);
        float fl = sense(agents.array[i], SA, i, &mvfl);
        float fr = sense(agents.array[i], -SA, i, &mvfr);

        float rs = hash(rand());
        // float ra = rs * RA * dt;
        float ra = RA;

        Agent a = agents.array[i];

        if (f > fl && f > fr) {
            agents.array[i].angle += 0;
            if(mvf > 0 && a.trail > 0)  ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = mvf;
        }
        else if(f < fl && f < fr) 
            if(rand() <= 0.5) {
                agents.array[i].angle = (int) (agents.array[i].angle - ra);
                if(mvfr > 0 && a.trail > 0)  ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = mvfr;
            }
            else {
                agents.array[i].angle = (int) (agents.array[i].angle + ra);
                if(mvfl > 0 && a.trail > 0)  ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = mvfl;
            }
        else if(fl < fr){
            agents.array[i].angle = (int) (agents.array[i].angle - ra);
            if(mvfr > 0 && a.trail > 0)  ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = mvfr;
        } 
        else if(fr < fl){
            agents.array[i].angle = (int) (agents.array[i].angle + ra);        
            if(mvfl > 0 && a.trail > 0)  ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = mvfl;
        } 
        
    }
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
    // agents = (Agent*) calloc(nAgentes, sizeof(Agent));
    trail = (float *) calloc(WIDTH * HEIGHT, sizeof(float));
    ftrail = (float *) calloc(WIDTH * HEIGHT, sizeof(float));
    food = (int *) calloc(WIDTH * HEIGHT, sizeof(int));
    initArray(&agents, 5);

    printf("(%d, %d)",WIDTH, HEIGHT);

    float r = WIDTH / 6;
    
    nAgentes = WIDTH * HEIGHT * p;
    for(int i = 0; i < nAgentes; i++) {
        // float angle =  hash(random()) * 2 * PI;
        // float x = hash(random()) * r * cos(angle) + WIDTH / 2;
        // float y = hash(random()) * r * sin(angle) + HEIGHT / 2;
   
        Agent a;

        // a.x = x;
        // a.y = y;
        // a.angle = angle * 180 / PI;
        // a.trail = 0;

        a.x = WIDTH / 2  + (hash(random()) * 2 - 1) * WIDTH * 3 / 4;
        a.y = HEIGHT / 2 + (hash(random()) * 2 - 1) * HEIGHT * 3 / 4 ;
        a.angle = (int) (hash(random()) * 360);

        insertArray(&agents, a);
    }
}


void draw(){

    if (rightButton == GLFW_PRESS){
        begin = !begin;
    }

    if(leftButton == GLFW_PRESS){
        if(cursorX < WIDTH && cursorX >= 0 && cursorY < HEIGHT && cursorY >= 0){
            int i = (int) (cursorX * HEIGHT + cursorY);
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

    if (begin){
        blur();
        show();
        update();
    } else {
        glPointSize(1.0);
        glBegin(GL_POINTS);

        for(int i = 0; i < WIDTH; i++){
            for(int j = 0; j < HEIGHT; j++){
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

int main(void){
    if (!glfwInit()) return -1;

    window = glfwCreateWindow(WIDTH, HEIGHT, "physarum", NULL, NULL);
    if (!window){
        glfwTerminate();
        return -1;
    }

    if (glfwRawMouseMotionSupported() == 0){
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glfwMakeContextCurrent(window);
    setup();

    while (!glfwWindowShouldClose(window)){
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
    }

    freeArray(&agents);
    glfwTerminate();
    return 0;
}2