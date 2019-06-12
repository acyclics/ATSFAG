/*
*
*	Pid lib
*
*/
#include "pid.h"

/* FUNCTIONS: PID - related */
float PID_CALC(float target, PID* unit) {
	float error = target - unit->current_value;
	unit->integral += error;
	unit->pre_error = error;
	return (unit->p * error) + (unit->i * unit->integral) + (unit->d * (error - unit->pre_error));
}
/* END of FUNCTIONS: PID - related */
