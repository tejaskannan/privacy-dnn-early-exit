/*
 * dht11.h
 *
 *  Created on: 27 Apr 2022
 *      Author: tejask
 */
#include <stdint.h>
#include <msp430.h>

#ifndef SENSOR_DHT11_H_
#define SENSOR_DHT11_H_

enum dht11_status {
    CHECKSUM_ERROR = -2,
    TIMEOUT_ERROR = -1,
    OK = 0
};

struct dht11_reading {
    int16_t status;
    int16_t temperature;
    int16_t humidity;
};

void send_start_signal();
int check_response();
int check_checksum(uint8_t data[]);

struct dht11_reading *dht11_read(struct dht11_reading *result);


#endif /* SENSOR_DHT11_H_ */
