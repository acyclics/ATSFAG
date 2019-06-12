#ifndef TRACK_H
#define TRACK_H

#include "can.h"

void ATS_CREATE_TRACK_Task(void);
void ATS_CREATE_NANOTXRX_Task(void);
void ATS_ENCODER_Read(CAN_RxHeaderTypeDef* rxHeader, uint8_t* canRxMsg, int CAN_msgs_received);

#endif
