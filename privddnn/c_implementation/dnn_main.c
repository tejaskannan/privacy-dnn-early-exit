#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "neural_network.h"
#include "matrix.h"
#include "policy.h"
#include "parameters.h"
#include "data.h"
#include "utils/lfsr.h"
#include "utils/message.h"


#define MESSAGE_BUFFER_SIZE 512
static uint8_t MESSAGE_BUFFER[MESSAGE_BUFFER_SIZE];


int main(void) {
    int16_t inputFeatures[NUM_FEATURES];
    struct matrix inputs = { inputFeatures, NUM_FEATURES, 1 };

    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t totalCount = 0;
    uint16_t lfsrState = 3798;

    int16_t hiddenData[W0.numRows];
    struct matrix hidden = { hiddenData, W0.numRows, 1 };

    int32_t resultLogits[NUM_LABELS];
    int32_t resultProbs[NUM_LABELS];
    struct inference_result result = { resultLogits, resultProbs, 0 };

    uint8_t pred;
    uint8_t shouldExit;
    uint16_t messageSize;
    uint16_t i, j;

    for (i = 0; i < NUM_INPUTS; i++) {

        for (j = 0; j < NUM_FEATURES; j++) {
            inputs.data[j] = DATASET_INPUTS[i * NUM_FEATURES + j];
        }

        label = DATASET_LABELS[i];

        // Run the neural network inference
        neural_network(&result, &hidden, &inputs, PRECISION);
        pred = result.pred;

        #ifdef IS_MAX_PROB
        shouldExit = max_prob_should_exit(&result, THRESHOLD);
        #elif IS_RANDOM
        shouldExit = random_should_exit(EXIT_RATE, lfsrState);
        lfsrState = lfsr_step(lfsrState);
        #endif

        if (shouldExit) {
            messageSize = create_exit_message(MESSAGE_BUFFER, &result);
        } else {
            messageSize = create_elevate_message(MESSAGE_BUFFER, &inputs, &hidden, MESSAGE_BUFFER_SIZE);
        }

	    isCorrect += (pred == label);
	    totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    return 0;
}
