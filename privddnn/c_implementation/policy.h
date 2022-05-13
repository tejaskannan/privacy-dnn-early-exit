#include <stdint.h>
#include "utils/fixed_point.h"
#include "utils/prand.h"

#ifndef POLICY_H_
#define POLICY_H_

struct exit_policy {
    uint16_t *thresholds;
    struct rand_state *randState;
};


struct cgr_state {
    uint16_t windowSize;
    uint16_t step;
    uint16_t *targetExit;
    uint16_t *observedExit;
    int16_t *biases;
    int16_t maxBias;
    int16_t increaseFactor;
    int16_t decreaseFactor;
    uint16_t windowMin;
    uint16_t windowMax;
    uint16_t windowBits;
    uint8_t *prevPreds;
    int16_t *trueExitRate;
};

uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold);
uint8_t random_should_exit(uint16_t exitRate, struct rand_state *randState);

int16_t get_upper_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision);
int16_t get_lower_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision);
uint8_t cgr_should_exit(uint32_t prob, uint8_t pred, uint16_t exitRate, uint16_t exitThreshold, struct rand_state *randState, struct cgr_state *policyState, uint8_t outputIdx, uint8_t numOutputs, uint8_t precision);
void update_cgr(struct cgr_state *policyState, uint8_t outputIdx);

#endif
