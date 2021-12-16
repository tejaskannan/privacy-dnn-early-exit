#include <stdint.h>

#ifndef LFSR_H_
#define LFSR_H_

uint16_t lfsr_step(uint16_t state);
uint8_t *lfsr_array(uint8_t *array, uint8_t n);

#endif
