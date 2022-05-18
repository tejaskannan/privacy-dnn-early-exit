#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "neural_network.h"
#include "matrix.h"
#include "policy.h"
#include "parameters.h"
#include "data.h"
#include "utils/prand.h"


int main(void) {
    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t exitDecision = 0;
    uint32_t totalCount = 0;
    uint16_t i, j;

    #ifdef IS_MAX_PROB
    uint16_t thresholds[NUM_OUTPUTS - 1];
    for (i = 0; i < NUM_OUTPUTS - 1; i++) {
        thresholds[i] = THRESHOLDS[i];
    }
    #elif defined(IS_RANDOM)
    uint16_t thresholds[] = { 0 };
    #elif defined(IS_LABEL_MAX_PROB) || defined(IS_CGR_MAX_PROB)
    uint16_t thresholds[NUM_LABELS * (NUM_OUTPUTS - 1)];

    for (i = 0; i < NUM_LABELS * (NUM_OUTPUTS - 1); i++) {
        thresholds[i] = THRESHOLDS[i];
    }
    #endif

    struct cgr_state policyState;

    #ifdef IS_CGR_MAX_PROB
    uint16_t targetExit[NUM_OUTPUTS];
    uint16_t observedExit[NUM_OUTPUTS];
    int16_t biases[NUM_OUTPUTS];
    uint8_t prevPreds[NUM_OUTPUTS];

    for (i = 0; i < NUM_OUTPUTS; i++) {
        biases[i] = MAX_BIAS;
        prevPreds[i] = NUM_LABELS + 1;
        targetExit[i] = 0;
        observedExit[i] = 0;
    }

    policyState.windowSize = WINDOW_MIN;
    policyState.step = 0;
    policyState.targetExit = targetExit;
    policyState.observedExit = observedExit;
    policyState.biases = biases;
    policyState.maxBias = MAX_BIAS;
    policyState.increaseFactor = INCREASE_FACTOR;
    policyState.decreaseFactor = DECREASE_FACTOR;
    policyState.windowMin = WINDOW_MIN;
    policyState.windowMax = WINDOW_MAX;
    policyState.windowBits = WINDOW_BITS;
    policyState.prevPreds = prevPreds;
    policyState.trueExitRate = (int16_t *) EXIT_RATES;
    #endif

    int16_t inputFeatures[NUM_FEATURES];
    struct matrix inputs = { inputFeatures, NUM_FEATURES, 1 };

    //uint32_t randVals[16] = { 13739, 1784, 12551, 13261, 29481, 18551, 24115, 643, 4136, 24125, 31906, 9173, 27431, 21844, 15768, 12087 };
    uint32_t randVals[8] = { 13739, 1784, 12551, 13261, 29481, 18551, 24115, 643 };
    struct rand_state randState = { randVals, 0, 8 };
    struct exit_policy policy = { thresholds, &randState };

    struct inference_result result;
    uint8_t pred;

    for (i = 0; i < NUM_INPUTS; i++) {
       for (j = 0; j < NUM_FEATURES; j++) {
            inputs.data[j] = DATASET_INPUTS[i * NUM_FEATURES + j];
        }

        label = DATASET_LABELS[i];

        // Run the neural network inference
        branchynet_dnn(&result, &inputs, PRECISION, &policy, &policyState);

        printf("Pred: %d, Label: %d, Decision: %d\n", result.pred, label, result.outputIdx);

        // Generate random values for the next batch
        generate_pseudo_rand(&randState);

	    isCorrect += (result.pred == label);
        exitDecision += result.outputIdx;
	    totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    printf("Avg Output: %d / %d\n", exitDecision, totalCount);
    return 0;
}
