#include <stdint.h>
#include "utils/lfsr.h"

#ifndef POLICY_H_
#define POLICY_H_

struct exit_policy {
    uint16_t *thresholds;
    uint16_t *lfsrStates;
};

uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold);
uint8_t random_should_exit(uint16_t exitRate, uint16_t *lfsrState);

#endif
