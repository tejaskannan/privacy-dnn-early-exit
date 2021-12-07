#include <stdint.h>
#include "decision_tree.h"

#ifndef POLICY_H_
#define POLICY_H_

uint8_t max_prob_should_exit(struct inference_result *earlyResult, int16_t exitThreshold);
uint8_t *buffered_max_prob_should_exit(uint8_t *results, struct inference_result *earlyResults, uint16_t exitRate, uint16_t lfsrState, uint16_t exitCount, uint16_t windowSize);
uint8_t random_should_exit(uint16_t exitRate, uint16_t lfsrState);

#endif
