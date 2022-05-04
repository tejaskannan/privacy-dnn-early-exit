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

    uint16_t lfsrStates[] = { 0 };
    #elif defined(IS_RANDOM)
    uint16_t thresholds[] = { 0 };
    uint16_t lfsrStates[] = { 3798 };
    #elif defined(IS_LABEL_MAX_PROB) || defined(IS_ADAPTIVE_RANDOM_MAX_PROB)
    uint16_t thresholds[NUM_LABELS * (NUM_OUTPUTS - 1)];

    for (i = 0; i < NUM_LABELS * (NUM_OUTPUTS - 1); i++) {
        thresholds[i] = THRESHOLDS[i];
    }

    #ifdef IS_LABEL_MAX_PROB
    uint16_t lfsrStates[] = { 0 };
    #else
    uint16_t lfsrStates[] = { 3918, 6742, 10752 };
    #endif
    #endif

    struct adaptive_random_state policyState;

    #ifdef IS_ADAPTIVE_RANDOM_MAX_PROB
    uint16_t targetExit[NUM_OUTPUTS - 1] = { 0 };
    uint16_t observedExit[NUM_OUTPUTS] = { 0 };

    policyState.windowSize = WINDOW_MIN;
    policyState.step = 0;
    policyState.targetExit = targetExit;
    policyState.observedExit = observedExit;
    policyState.bias = MAX_BIAS;
    policyState.maxBias = MAX_BIAS;
    policyState.increaseFactor = INCREASE_FACTOR;
    policyState.decreaseFactor = DECREASE_FACTOR;
    policyState.windowMin = WINDOW_MIN;
    policyState.windowMax = WINDOW_MAX;
    policyState.windowBits = WINDOW_BITS;
    policyState.prevPred = NUM_LABELS + 1;
    policyState.trueExitRate = (int16_t *) EXIT_RATES;
    #endif

    int16_t inputFeatures[NUM_FEATURES];
    struct matrix inputs = { inputFeatures, NUM_FEATURES, 1 };

    struct exit_policy policy = { thresholds, lfsrStates };

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

	    isCorrect += (result.pred == label);
        exitDecision += result.outputIdx;
	    totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    printf("Avg Output: %d / %d\n", exitDecision, totalCount);
    return 0;
}
