#include <stdint.h>
#include "utils/fixed_point.h"
#include "utils/lfsr.h"

#ifndef POLICY_H_
#define POLICY_H_

struct exit_policy {
    uint16_t *thresholds;
    uint16_t *lfsrStates;
};


struct adaptive_random_state {
    uint16_t windowSize;
    uint16_t step;
    uint16_t targetExit;
    uint16_t observedExit;
    int16_t bias;
    int16_t maxBias;
    int16_t increaseFactor;
    int16_t decreaseFactor;
    uint16_t windowMin;
    uint16_t windowMax;
    uint16_t windowBits;
    uint8_t prevPred;
};

uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold);
uint8_t random_should_exit(uint16_t exitRate, uint16_t *lfsrState);

int16_t get_upper_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision);
int16_t get_lower_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision);
uint8_t adaptive_random_should_exit(uint32_t prob, uint8_t pred, uint16_t exitRate, uint16_t exitThreshold, uint16_t *lfsrState, struct adaptive_random_state *policyState, uint8_t precision);

#endif
