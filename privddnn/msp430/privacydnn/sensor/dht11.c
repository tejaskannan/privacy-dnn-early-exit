/*
 * dht11.c
 *
 *  Created on: 27 Apr 2022
 *      Author: tejask
 */
#include "dht11.h"


int16_t wait_while_one(uint16_t microseconds) {
    volatile uint16_t ticks = 0;
    do {
        ticks += 1;
    } while (((P6IN & BIT3) == BIT3) && (ticks <= microseconds));

    if (ticks > microseconds) {
        return TIMEOUT_ERROR;
    }

    return (int16_t) ticks;
}

int16_t wait_while_zero(uint16_t microseconds) {
    volatile uint16_t ticks = 0;
    do {
        ticks += 1;
    } while (((P6IN & BIT3) == 0) && (ticks <= microseconds));

    if (ticks > microseconds) {
        return TIMEOUT_ERROR;
    }

    return (int16_t) ticks;
}


void send_start_signal() {
    // Set P6.3 to output and set the pin low
    P6DIR |= BIT3;
    P6OUT &= ~BIT3;

    // Wait for 18ms
    __delay_cycles(18 * 1000);

    // Set the output to 1 for 40us
    P6OUT |= BIT3;
    __delay_cycles(40);

    // Set the pin to input (in preparation for reading)
    P6OUT &= ~BIT3;
    P6DIR &= ~BIT3;
}

int16_t check_ack() {
    // Wait up to 80 us for the next step
    if (wait_while_zero(80) == TIMEOUT_ERROR) {
        return TIMEOUT_ERROR;
    }

    // Wait up to 80us for the next step
    if (wait_while_one(80) == TIMEOUT_ERROR) {
        return TIMEOUT_ERROR;
    }

    return OK;
}


int16_t check_checksum(uint8_t data[]) {
    if ((data[0] + data[1] + data[2] + data[3]) != data[4]) {
        return CHECKSUM_ERROR;
    }
    return OK;
}

struct dht11_reading *dht11_read(struct dht11_reading *result) {
    // Send the start signal and check the acknowledgement
    send_start_signal();

    if (check_ack() == TIMEOUT_ERROR) {
        result->status = TIMEOUT_ERROR;
        result->temperature = -1;
        result->humidity = -1;
        return result;
    }

    uint8_t data[5] = { 0, 0, 0, 0, 0 };

    // Read the response bit by bit (8 bits / byte and 5 total bytes)
    volatile int16_t delay = 0;
    uint16_t i, j;
    for (i = 0; i < 5; i++) {
        for (j = 8; j > 0; j--) {
            // Wait the initial 50 us
            delay = wait_while_zero(50);
            if (delay == TIMEOUT_ERROR) {
                result->status = TIMEOUT_ERROR;
                result->temperature = -1;
                result->humidity = -1;
                return result;
            }

            // If the number of microseconds of a 1 is > 28, then we have a '1'
            // Otherwise it is a '0'
            delay = wait_while_one(75);
            if (delay > 28) {
                data[i] |= (1 << (j - 1));
            }
        }
    }

    if (check_checksum(data) != CHECKSUM_ERROR) {
        result->status = OK;
        result->temperature = data[2];
        result->humidity = data[0];
        return result;
    }

    result->status = CHECKSUM_ERROR;
    result->temperature = -1;
    result->humidity = -1;
    return result;
}
