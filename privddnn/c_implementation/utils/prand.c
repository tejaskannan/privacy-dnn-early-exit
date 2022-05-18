#include "prand.h"


void generate_pseudo_rand(struct rand_state *state) {
    uint16_t i;

    uint32_t seed = 0;
    for (i = 0; i < state->numVals; i++) {
        seed ^= state->vals[i];
    }

    for (i = 0; i < state->numVals; i++) {
        seed ^= (seed << 13);
        seed ^= (seed >> 17);
        seed ^= (seed << 5);
        state->vals[i] = seed;
    }

    state->index = 0;
}


uint16_t pseudo_rand(struct rand_state *state) {
    const uint32_t val = state->vals[state->index];
    state->index += 1;
    return (uint16_t) ((val >> 16) & 0xFFFF);
}


uint16_t rand_int(struct rand_state *state, uint16_t bits) {
    uint16_t rand = pseudo_rand(state);
    const uint16_t mask = ~(0xFFFF << bits);
    return (rand & mask);
}
