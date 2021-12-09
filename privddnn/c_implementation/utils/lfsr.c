#include "lfsr.h"

uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}
