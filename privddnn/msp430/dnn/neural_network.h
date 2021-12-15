#include <stdint.h>
#include "matrix.h"
#include "parameters.h"
#include "utils/inference_result.h"

#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

struct inference_result *neural_network(struct inference_result *result, struct matrix *hiddenResult, struct matrix *inputs, uint8_t precision);

#endif
