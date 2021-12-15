#include <stdint.h>

#ifndef INFERENCE_RESULT_H_
#define INFERENCE_RESULT_H_

struct inference_result {
    int32_t *logits;
    int32_t *probs;
    uint8_t pred;
};

#endif

