#include "lfsr.h"

uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}


uint8_t *lfsr_array(uint8_t *array, uint8_t n) {
    uint8_t i, val;
    for (i = 0; i < n; i += 2) {
        val = (((uint16_t) array[i]) << 8) | ((uint16_t) array[i+1]);
        val = lfsr_step(val);

        array[i] = (val >> 8) & 0xFF;
        array[i+1] = val & 0xFF;
    }

    return array;
}
