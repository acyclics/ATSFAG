#ifndef PID_H
#define PID_H

typedef struct PID {
	float p, i, d, error, pre_error, integral, current_value;
} PID;

float PID_CALC(float target, PID* unit);
#endif
