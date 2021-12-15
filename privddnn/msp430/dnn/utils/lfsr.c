#include "lfsr.h"

uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}


uint16_t *lfsr_array(uint16_t *array, uint8_t n) {
    uint8_t i;
    for (i = 0; i < n; i++) {
        array[i] = lfsr_step(array[i]);
    }

    return array;
}
