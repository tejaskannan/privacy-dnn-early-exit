#include "lfsr.h"

uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}


uint16_t rand_int(uint16_t lfsrState, uint16_t bits) {
    const uint16_t mask = ~(0xFFFF << bits);
    return lfsrState & mask;

    //uint16_t rand = 0;
    //while (rand < lfsrState) {
    //    rand += stepSize;
    //}

    //rand -= stepSize;
    //return rand + min;
}
