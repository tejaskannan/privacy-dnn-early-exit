#include <stdint.h>

#ifndef PRAND_H_
#define PRAND_H_


struct rand_state {
    uint32_t *vals;
    uint16_t index;
    uint16_t numVals;
};

void generate_pseudo_rand(struct rand_state *state);
uint16_t pseudo_rand(struct rand_state *state);
uint16_t rand_int(struct rand_state *state, uint16_t bits);

#endif
