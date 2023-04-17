#include <stdint.h>

#ifndef INFERENCE_RESULT_H_
#define INFERENCE_RESULT_H_

struct inference_result {
    uint8_t pred;
    uint8_t outputIdx;
    int16_t *state;
};

#endif

