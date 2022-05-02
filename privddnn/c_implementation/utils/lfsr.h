#include <stdint.h>

#ifndef LFSR_H_
#define LFSR_H_

uint16_t lfsr_step(uint16_t state);
uint16_t rand_int(uint16_t lfsrState, uint16_t bits);

#endif
