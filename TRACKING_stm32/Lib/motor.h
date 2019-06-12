/****************************************************************************
 *  Copyright (C) 2019 RoboMaster.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#ifndef __MOTOR_H__
#define __MOTOR_H__

#ifndef ENCODER_ANGLE_RATIO
  #define ENCODER_ANGLE_RATIO (8192.0f / 360.0f)
#endif

#include "main.h"
#include "stm32f4xx_hal.h"
#include "cmsis_os.h"

typedef struct motor_data
{
  uint16_t ecd;
  uint16_t last_ecd;

  int16_t speed_rpm;
  int16_t given_current;

  int32_t round_cnt;
  int32_t total_ecd;
  int32_t total_angle;

  int32_t ecd_raw_rate;

  uint32_t msg_cnt;
  uint16_t offset_ecd;
} motor_data;

typedef struct motor_device
{
  motor_data data;
  uint16_t can_id;
  uint16_t init_offset_f;
  int16_t current;
} motor_device;

void get_encoder_data(motor_device* motor, uint8_t can_rx_data[]);

#endif // __MOTOR_H__
