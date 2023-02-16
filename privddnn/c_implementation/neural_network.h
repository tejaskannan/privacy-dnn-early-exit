#include <stdint.h>
#include "matrix.h"
#include "parameters.h"
#include "utils/inference_result.h"
#include "utils/array.h"
#include "policy.h"

#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#define BLOCK_SIZE 32
struct inference_result *branchynet_dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision, struct exit_policy *policy, struct cgr_state *policyState);
struct inference_result *dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision);

#endif
