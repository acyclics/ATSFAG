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

#include "motor.h"

static void get_motor_offset(motor_data* ptr, uint8_t can_rx_data[])
{
  ptr->ecd = (uint16_t)(can_rx_data[0] << 8 | can_rx_data[1]);
  ptr->offset_ecd = ptr->ecd;
}
void get_encoder_data(motor_device* motor, uint8_t can_rx_data[])
{
  motor_data* ptr = &(motor->data);
  ptr->msg_cnt++;

  if (ptr->msg_cnt > 50)
  {
    motor->init_offset_f = 0;
  }

  if (motor->init_offset_f == 1)
  {
    get_motor_offset(ptr, can_rx_data);
    return;
  }

  ptr->last_ecd = ptr->ecd;
  ptr->ecd = (uint16_t)(can_rx_data[0] << 8 | can_rx_data[1]);

  if (ptr->ecd - ptr->last_ecd > 4096)
  {
    ptr->round_cnt--;
    ptr->ecd_raw_rate = ptr->ecd - ptr->last_ecd - 8192;
  }
  else if (ptr->ecd - ptr->last_ecd < -4096)
  {
    ptr->round_cnt++;
    ptr->ecd_raw_rate = ptr->ecd - ptr->last_ecd + 8192;
  }
  else
  {
    ptr->ecd_raw_rate = ptr->ecd - ptr->last_ecd;
  }

  ptr->total_ecd = ptr->round_cnt * 8192 + ptr->ecd - ptr->offset_ecd;
  /* total angle, unit is degree */
  ptr->total_angle = ptr->total_ecd / ENCODER_ANGLE_RATIO;

  ptr->speed_rpm = (int16_t)(can_rx_data[2] << 8 | can_rx_data[3]);
  ptr->given_current = (int16_t)(can_rx_data[4] << 8 | can_rx_data[5]);
}
