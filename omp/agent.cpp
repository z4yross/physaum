#include <math.h>

const double PI = 3.14159265358979323846;

class Agent
{
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

    int WIDTH;
    int HEIGHT;

    int GAP;

public:
    // int i;

    float x;
    float y;
    float angle;
    float trail;

    Agent(float x, float y, float angle, int WIDTH, int HEIGHT, int GAP)
    {
        // this->i = i;
        this->x = x;
        this->y = y;
        this->angle = angle;
        this->WIDTH = WIDTH;
        this->HEIGHT = HEIGHT;
        this->GAP = GAP;
    }

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

    float sense(float angle, float *mv, float *trail, float *ftrail)
    {
        float rad = (this->angle + angle) * PI / 180;

        int sCX = this->x + SO * cos(rad);
        int sCY = this->y + SO * sin(rad);

        float sum = 0;

        // glColor4f(0,0,1,1);
        // glPointSize(1.0);

        // glBegin(GL_POINTS);

        for (int i = -SW; i <= SW; i++)
        {
            for (int j = -SW; j <= SW; j++)
            {
                int pX = sCX + i;
                int pY = sCY + j;

                if (pX >= 0 && pX < WIDTH && pY >= 0 && pY < HEIGHT)
                {

                    // glVertex2f(pX, pY);
                    // glVertex2f(0, 0);(i * HEIGHT + j) * GAP
                    // glVertex2f(WIDTH / 2, HEIGHT / 2);
                    float tV = trail[(int)((int)pX * HEIGHT + (int)pY) * GAP];
                    float ftV = ftrail[(int)((int)pX * HEIGHT + (int)pY) * GAP];
                    sum += tV + ftV * TW;

                    if (ftV > 0)
                    {
                        this->trail = TD;
                        *mv = fmax(*mv, ftV);
                    }
                }
            }
        }

        // glEnd();

        return sum;
    }

    void move(float *trail)
    {
        float rad = this->angle * PI / 180.0;

        float vx = SS * cos(rad) * dt;
        float vy = SS * sin(rad) * dt;

        float px = this->x + vx;
        float py = this->y + vy;

        if (pow(px - WIDTH / 2, 2) + pow(py - HEIGHT / 2, 2) > ((WIDTH / 2) * 3 / 4)  * ((HEIGHT / 2) * 3 / 4) || px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT)
        {
            // a.x = fmin(WIDTH - 0.01, fmax(0, px));
            // a.y =  fmin(HEIGHT - 0.01, fmax(0, py));
            this->angle = (int)(hash(random()) * 360);
            // a.angle = (int) 0;
        }
        else
        {
            this->x = px;
            this->y = py;
            // printf("%d(%f %f %f)\n", i, px, py, a.angle);

            // if (a.trail > 0)
            // ftrail[(int)((int)a.x * HEIGHT + (int)(a.y))] = 1;

            trail[(int)((int)(px)*HEIGHT + (int)(py)) * GAP] = 1;
        }

        this->trail = fmax(0, this->trail - 1 * dt);
    }

    void update(float *trail, float *ftrail)
    {
        move(trail);

        float mvf = 0;
        float mvfl = 0;
        float mvfr = 0;

        float f = sense(0, &mvf, trail, ftrail);
        float fl = sense(SA, &mvfl, trail, ftrail);
        float fr = sense(-SA, &mvfr, trail, ftrail);

        // float rs = hash(rand());
        // float ra = rs * RA * dt;
        float ra = RA;

        if (f > fl && f > fr)
        {
            this->angle += 0;
            if (mvf > 0 && this->trail > 0)
                ftrail[(int)((int)this->x * HEIGHT + (int)(this->y)) * GAP] = mvf;
        }
        else if (f < fl && f < fr)
            if (rand() <= 0.5)
            {
                this->angle = (int)(this->angle - ra);
                if (mvfr > 0 && this->trail > 0)
                    ftrail[(int)((int)this->x * HEIGHT + (int)(this->y)) * GAP] = mvfr;
            }
            else
            {
                this->angle = (int)(this->angle + ra);
                if (mvfl > 0 && this->trail > 0)
                    ftrail[(int)((int)this->x * HEIGHT + (int)(this->y)) * GAP] = mvfl;
            }
        else if (fl < fr)
        {
            this->angle = (int)(this->angle - ra);
            if (mvfr > 0 && this->trail > 0)
                ftrail[(int)((int)this->x * HEIGHT + (int)(this->y)) * GAP] = mvfr;
        }
        else if (fr < fl)
        {
            this->angle = (int)(this->angle + ra);
            if (mvfl > 0 && this->trail > 0)
                ftrail[(int)((int)this->x * HEIGHT + (int)(this->y)) * GAP] = mvfl;
        }
    }
};