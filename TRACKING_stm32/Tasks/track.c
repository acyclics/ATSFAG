/*
*
*	Tracking task
*
*/
#include "freertos.h"
#include "task.h"
#include "cmsis_os.h"
#include "track.h"
#include "can.h"
#include "pid.h"
#include "usart.h"
#include "math.h"
#include "motor.h"

/* VARIABLES: TRACK - related */
motor_device yaw_motor, pitch_motor;
uint8_t target_angular_velocity[4];
volatile int DMA_CALLBACK = 0;
float TARGET_yaw_speed = 0, TARGET_pitch_speed = 0;
PID yaw = {
	70.0,
	0,
	40,
	0,
	0,
	0,
	0
};
PID pitch = {
	55.0,
	0.00005,
	30,
	0,
	0,
	0,
	0
};
extern CAN_TxHeaderTypeDef can1TxHeader0;
extern CAN_HandleTypeDef hcan1;
extern UART_HandleTypeDef huart6;
/* END of VARIABLES: TRACK - related */

/* VARIABLES: RTOS - related */
osThreadId ATS_TRACK_Handle, ATS_NANOTXRX_Handle;
/* END of VARIABLES: RTOS - related */

/* FUNCTIONS: TRACK - related */
void ATS_TRANSMIT_Can(int16_t cm1_iq,int16_t cm4_iq)
{
		uint8_t canTxMsg0[8];
    canTxMsg0[0] = (uint8_t)(cm1_iq >> 8);
    canTxMsg0[1] = (uint8_t)cm1_iq;
    canTxMsg0[2] = (uint8_t)(0 >> 8);
    canTxMsg0[3] = (uint8_t)0;
    canTxMsg0[4] = (uint8_t)(0 >> 8);
    canTxMsg0[5] = (uint8_t)0;
    canTxMsg0[6] = (uint8_t)(cm4_iq >> 8);
    canTxMsg0[7] = (uint8_t)cm4_iq;
		HAL_CAN_AddTxMessage(&hcan1,&can1TxHeader0,canTxMsg0,(void*)CAN_TX_MAILBOX0);
}
void ATS_ENCODER_Read(CAN_RxHeaderTypeDef* rxHeader, uint8_t canRxMsg[], int CAN_msgs_received) {
	if (CAN_msgs_received == 0) {
		yaw_motor.can_id = 0x201;
		yaw_motor.init_offset_f = 1;
		pitch_motor.can_id = 0x204;
		pitch_motor.init_offset_f = 1;
		return;
	}
	switch (rxHeader->StdId) {
		case 0x201:
			get_encoder_data(&yaw_motor, canRxMsg);
			break;
		case 0x204:
			get_encoder_data(&pitch_motor, canRxMsg);
			break;
	}
}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef* huart) {
	DMA_CALLBACK = 1;
}
/* END of FUNCTIONS: TRACK - related */

/* FUNCTIONS: RTOS - related */
void ATS_TRACK_Task(void const *argument)
{
	uint32_t TRACK_wake_time = osKernelSysTick();
	for(;;)
  {
		yaw.current_value = yaw_motor.data.speed_rpm * (3.14159265359/30.0) / 19.0;
		pitch.current_value = pitch_motor.data.speed_rpm * (3.14159265359/30.0) / 19.0;
		ATS_TRANSMIT_Can(PID_CALC(TARGET_yaw_speed, &yaw), PID_CALC(TARGET_pitch_speed, &pitch));
    osDelayUntil(&TRACK_wake_time, 5);
  }
}

void ATS_CREATE_TRACK_Task(void)
{
	osThreadDef(ATSTRACKTask, ATS_TRACK_Task, osPriorityRealtime, 0, 256);
	ATS_TRACK_Handle = osThreadCreate(osThread(ATSTRACKTask), NULL);
}
int TRANSMIT_OK = 1, RECEIVE_OK = 0;
void ATS_NANOTXRX_Task(void const *argument)
{
	uint32_t NANORX_wake_time = osKernelSysTick();
	uint8_t angular_velocity[4];
	
	for(;;)
  {
		if (TRANSMIT_OK) {
			int yaw_angular_velocity = (yaw_motor.data.speed_rpm * (3.14159265359/30.0) / 19.0) * 1000;
			int pitch_angular_velocity = (pitch_motor.data.speed_rpm * (3.14159265359/30.0) / 19.0) * 1000;
			angular_velocity[0] = yaw_angular_velocity >> 8;
			angular_velocity[1] = yaw_angular_velocity;
			angular_velocity[2] = pitch_angular_velocity >> 8;
			angular_velocity[3] = pitch_angular_velocity;
			if (HAL_UART_Transmit(&huart6, angular_velocity, 4, 10) != HAL_TIMEOUT) {
				RECEIVE_OK = 1;
				TRANSMIT_OK = 0;
			}
		}
		if (RECEIVE_OK && DMA_CALLBACK) {
			TARGET_yaw_speed = (int16_t)((target_angular_velocity[2] & 0xFF) << 8 | (target_angular_velocity[3] & 0xFF))/ 1000.0;
			TARGET_pitch_speed = (int16_t)((target_angular_velocity[0] & 0xFF) << 8 | (target_angular_velocity[1] & 0xFF))/ 1000.0;
			RECEIVE_OK = 0;
			TRANSMIT_OK = 1;
			DMA_CALLBACK = 0;
		}
    osDelayUntil(&NANORX_wake_time, 5);
  }
}

void ATS_CREATE_NANOTXRX_Task(void)
{
	osThreadDef(ATSNANOTXRXTask, ATS_NANOTXRX_Task, osPriorityRealtime, 0, 256);
	ATS_NANOTXRX_Handle = osThreadCreate(osThread(ATSNANOTXRXTask), NULL);
}
/* END of FUNCTIONS: RTOS - related */
