#include "policy.h"

#define BUFFER_SIZE 16
int16_t PROBS_BUFFER[BUFFER_SIZE];
uint8_t INDEX_BUFFER[BUFFER_SIZE];


uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold) {
    return (uint8_t) (prob >= exitThreshold);
}


uint8_t random_should_exit(uint16_t exitRate, struct rand_state *randState) {
    uint16_t rand = pseudo_rand(randState);
    uint8_t shouldExit = (uint8_t) ((rand & 0x7FFF) <= exitRate);
    return shouldExit;
}


int16_t get_upper_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision) {
    const int16_t one = 1 << precision;
    const int16_t increase = fp16_mul(fp16_sub(one, continueRate), bias, precision);
    return fp16_add(continueRate, increase);
}


int16_t get_lower_continue_rate(int16_t continueRate, int16_t bias, uint8_t precision) {
    const int16_t one = 1 << precision;
    const int16_t decrease = fp16_sub(one, bias);
    return fp16_mul(continueRate, decrease, precision);
}


uint8_t cgr_should_exit(uint32_t prob, uint8_t pred, uint16_t exitRate, uint16_t exitThreshold, struct rand_state *randState, struct cgr_state *policyState, uint8_t outputIdx, uint8_t numOutputs, uint8_t precision) {
    const uint16_t adjustedExitRate = exitRate >> (15 - precision);
    volatile uint8_t i;
    volatile uint16_t rand;
    
    if ((policyState->step == 0) && (outputIdx == 0)) {
        // Update the window size (maybe do this every time to avoid timing attacks)
        policyState->windowSize = rand_int(randState, policyState->windowBits) + policyState->windowMin;  // Random integer within bounds
        policyState->step = policyState->windowSize;

        // Set the quotas for each output
        const uint16_t windowSizeFp = (policyState->windowSize) << precision;
        const uint16_t mask = ~(0xFFFF << precision);

        volatile uint16_t adjustedTrueExitRate;
        volatile uint16_t remainder;
        volatile uint8_t adjustment;
        volatile int16_t targetExit;

        volatile uint16_t quotaCount = 0;
        for (i = 0; i < numOutputs; i++) {
            rand = pseudo_rand(randState);

            adjustedTrueExitRate = policyState->trueExitRate[i] >> (15 - precision);
            targetExit = fp16_mul(windowSizeFp, adjustedTrueExitRate, precision);
            remainder = (targetExit & mask) << (15 - precision);
            adjustment = (uint8_t) ((rand & 0x7FFF) < remainder);

            policyState->targetExit[i] = (targetExit >> precision) + adjustment;  // (Window Size * exit Rate) + Random part based on float remainder
            policyState->observedExit[i] = 0;
            quotaCount += policyState->targetExit[i];
        }

        i = 0;
        while (quotaCount < policyState->windowSize) {
            policyState->targetExit[i] += 1;
            quotaCount += 1;

            i += 1;
            if (i >= numOutputs) {
                i = 0;
            }
        }
    }

    // Update the probability bias based on the last prediction
    if (pred == policyState->prevPreds[outputIdx]) {
        policyState->biases[outputIdx] = fp16_mul(policyState->biases[outputIdx], policyState->decreaseFactor, precision);
    } else {
        policyState->biases[outputIdx] = fp16_mul(policyState->biases[outputIdx], policyState->increaseFactor, precision);
        policyState->biases[outputIdx] = fp16_min(policyState->biases[outputIdx], policyState->maxBias);
    }

    // Get the number of elements remaining in the window
    const uint16_t remainingToExit = policyState->targetExit[outputIdx] - policyState->observedExit[outputIdx];

    // Get the bias-adjusted exit rates
    const uint16_t one = 1 << ((uint16_t) precision);
    const uint16_t continueRate = fp16_sub(one, adjustedExitRate);
    uint16_t upperRate = get_upper_continue_rate(continueRate, policyState->biases[outputIdx], precision);
    uint16_t lowerRate = get_lower_continue_rate(continueRate, policyState->biases[outputIdx], precision);

    upperRate = fp16_min(upperRate, one);
    lowerRate = fp16_max(lowerRate, 0);

    // Convert to 15 bits of precision
    upperRate = upperRate << (15 - precision);
    lowerRate = lowerRate << (15 - precision);

    // Create the continue probability 
    volatile uint16_t continueProb = 0;

    if (exitRate == (1 << 15)) {
        continueProb = 0;        
    } else if (exitRate == 0) {
        continueProb = (1 << 15);
    } else if (prob < exitThreshold) {
        continueProb = lowerRate;
    } else {
        continueProb = upperRate;
    }

    // Set the exit decision
    rand = pseudo_rand(randState);
    volatile uint8_t shouldExit = (uint8_t) ((rand & 0x7FFF) > continueProb);

    // Set hard boundaries if we have reached to quota (on either side)
    if (remainingToExit == 0) {
        shouldExit = 0;
    } else if (remainingToExit >= policyState->step) {
        shouldExit = 1;
    } else if (!shouldExit) {
        // If all outputs above this one are exhausted, then we should exit here.
        shouldExit = 1;

        for (i = outputIdx + 1; i < numOutputs; i++) {
            if ((policyState->targetExit[i] - policyState->observedExit[i]) > 0) {
                shouldExit = 0;
                break;
            }
        }
    }

    policyState->prevPreds[outputIdx] = pred;
    return shouldExit;
}


void update_cgr(struct cgr_state *policyState, uint8_t outputIdx) {
    // Update the prediction, step and quota
    policyState->observedExit[outputIdx] += 1;
    policyState->step -= 1;
}
