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
    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t totalCount = 0;
    uint16_t lfsrState = 3798;
    uint16_t i, j;

#ifdef IS_BUFFERED_MAX_PROB
    uint16_t windowIdx = 0;

    int16_t inputFeatures[NUM_FEATURES * WINDOW_SIZE];
    struct matrix inputs[WINDOW_SIZE];

    int16_t hiddenData[W0.numRows * WINDOW_SIZE];
    struct matrix hidden[WINDOW_SIZE];

    int32_t resultLogits[NUM_LABELS * WINDOW_SIZE];
    int32_t resultProbs[NUM_LABELS * WINDOW_SIZE];
    struct inference_result result[WINDOW_SIZE];

    uint8_t shouldExit[WINDOW_SIZE];
    
    for (i = 0; i < WINDOW_SIZE; i++) {
        inputs[i].data = inputFeatures + (i * NUM_FEATURES);
        inputs[i].numRows = NUM_FEATURES;
        inputs[i].numCols = 1;

        hidden[i].data = hiddenData + (i * W0.numRows);
        hidden[i].numRows = W0.numRows;
        hidden[i].numCols = 1;

        result[i].logits = resultLogits + (i * NUM_LABELS);
        result[i].probs = resultProbs + (i * NUM_LABELS);
        result[i].pred = 0;
    }
#else
    int16_t inputFeatures[NUM_FEATURES];
    struct matrix inputs = { inputFeatures, NUM_FEATURES, 1 };

    int16_t hiddenData[W0.numRows];
    struct matrix hidden = { hiddenData, W0.numRows, 1 };

    int32_t resultLogits[NUM_LABELS];
    int32_t resultProbs[NUM_LABELS];
    struct inference_result result = { resultLogits, resultProbs, 0 };

    uint8_t shouldExit;
#endif

    uint8_t pred;
    uint16_t messageSize;

    for (i = 0; i < WINDOW_SIZE + 1; i++) {
#ifdef IS_BUFFERED_MAX_PROB
        for (j = 0; j < NUM_FEATURES; j++) {
            inputs[windowIdx].data[j] = DATASET_INPUTS[i * NUM_FEATURES + j];
        }

        label = DATASET_LABELS[i];

        // Run the neural network inference
        neural_network(result + windowIdx, hidden + windowIdx, inputs + windowIdx, PRECISION);
        pred = (result + windowIdx)->pred;
        
        windowIdx += 1;

        if (windowIdx == WINDOW_SIZE) {
            buffered_max_prob_should_exit(shouldExit, result, lfsrState, ELEVATE_COUNT, ELEVATE_REMAINDER, WINDOW_SIZE);
            messageSize = create_buffered_message(MESSAGE_BUFFER, result, inputs, hidden, shouldExit, WINDOW_SIZE, MESSAGE_BUFFER_SIZE);

            printf("Message Size: %d\n", messageSize);

            for (j = 0; j < messageSize; j++) {
                printf("\\x%x", MESSAGE_BUFFER[j]);
            }
            printf("\n");

            lfsrState = lfsr_step(lfsrState);
            windowIdx = 0;
        }
#else
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
#endif

	    isCorrect += (pred == label);
	    totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    return 0;
}
