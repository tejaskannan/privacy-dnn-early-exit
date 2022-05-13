#include "prand.h"

uint16_t pseudo_rand(struct rand_state *state) {
    uint32_t x = state->val;
    x ^= (x << 13);
    x ^= (x >> 17);
    x ^= (x << 5);
    state->val = x;
    return (uint16_t) ((x >> 16) & 0xFFFF);
    //uint16_t bit = ((state) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1u;
    //return (state >> 1) | (bit << 15);
}


uint16_t rand_int(struct rand_state *state, uint16_t bits) {
    uint16_t rand = pseudo_rand(state);
    const uint16_t mask = ~(0xFFFF << bits);
    return (rand & mask);
}
