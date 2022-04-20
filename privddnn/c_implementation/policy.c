#include "policy.h"

#define BUFFER_SIZE 16
int16_t PROBS_BUFFER[BUFFER_SIZE];
uint8_t INDEX_BUFFER[BUFFER_SIZE];


uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold) {
    return (uint8_t) (prob >= exitThreshold);
}


uint8_t random_should_exit(uint16_t exitRate, uint16_t *lfsrState) {
    uint8_t shouldExit = (uint8_t) ((*lfsrState & 0x7FFF) <= exitRate);
    *lfsrState = lfsr_step(*lfsrState);
    return shouldExit;
}
