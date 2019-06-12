/*
*
*	Can communication
*
*/
#include "freertos.h"
#include "task.h"
#include "cmsis_os.h"
#include "comm_can.h"
#include "can.h"
#include "track.h"

/* VARIABLES: CAN - related */
CAN_RxHeaderTypeDef can1RxHeader;
uint8_t canRxMsg[8];
int CAN_msgs_received = 0;
/* END of VARIABLES: CAN - related */

/* VARIABLES: RTOS - related */
osThreadId ATS_CAN_Handle;
/* END of VARIABLES: RTOS - related */

/* FUNCTIONS: RTOS - related */
void ATS_CAN_Task(void const *argument)
{
	uint32_t can_wake_time = osKernelSysTick();
	for(;;)
  {
		if (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) >= 1)
		{
			HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &can1RxHeader, canRxMsg);
			ATS_ENCODER_Read(&can1RxHeader, canRxMsg, CAN_msgs_received);
			++CAN_msgs_received;
		}
    osDelayUntil(&can_wake_time, 5);
  }
}

void ATS_CREATE_CAN_Task(void)
{
	osThreadDef(ATSCANTask, ATS_CAN_Task, osPriorityRealtime, 0, 256);
	ATS_CAN_Handle = osThreadCreate(osThread(ATSCANTask), NULL);
}
/* END of FUNCTIONS: RTOS - related */
