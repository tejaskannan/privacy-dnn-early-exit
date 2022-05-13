#include "prand.h"

uint16_t pseudo_rand(uint16_t state) {
    uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    return (state >> 1) | (bit << 15);
}


uint16_t rand_int(uint16_t randState, uint16_t bits) {
    const uint16_t mask = ~(0xFFFF << bits);
    return randState & mask;
}
