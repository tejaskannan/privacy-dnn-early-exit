#include "policy.h"

#define BUFFER_SIZE 16
int16_t PROBS_BUFFER[BUFFER_SIZE];
uint8_t INDEX_BUFFER[BUFFER_SIZE];


uint8_t max_prob_should_exit(uint32_t prob, uint16_t exitThreshold) {
    return (uint8_t) (prob >= exitThreshold);
}


uint8_t random_should_exit(uint16_t exitRate, uint16_t *lfsrState) {
    uint8_t shouldExit = (uint8_t) ((*lfsrState & 0x7FFF) <= exitRate);
    *lfsrState = lfsr_step(*lfsrState);
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


uint8_t adaptive_random_should_exit(uint32_t prob, uint8_t pred, uint16_t exitRate, uint16_t exitThreshold, uint16_t *lfsrState, struct adaptive_random_state *policyState, uint8_t precision) {
    const uint16_t adjustedExitRate = exitRate >> (15 - precision);

    if (policyState->step == 0) {
        // Update the window size (maybe do this every time to avoid timing attacks)
        policyState->windowSize = rand_int(lfsrState[1], policyState->windowBits) + policyState->windowMin;  // Random integer within bounds
        policyState->observedExit = 0;
        policyState->step = policyState->windowSize;

        const uint16_t windowSizeFp = (policyState->windowSize) << precision;
        const uint16_t targetExit = fp16_mul(windowSizeFp, adjustedExitRate, precision);
        const uint16_t mask = ~(0xFFFF << precision);
        const uint16_t remainder = (targetExit & mask) << (15 - precision);
        const uint16_t adjustment = (uint16_t) (lfsrState[2] < remainder);

        policyState->targetExit = (targetExit >> precision) + adjustment;  // (Window Size * exit Rate) + Random part based on float remainder
        lfsrState[1] = lfsr_step(lfsrState[1]);
        lfsrState[2] = lfsr_step(lfsrState[2]);
    }

    // Get the number of elements remaining in the window
    const uint16_t remainingToExit = policyState->targetExit - policyState->observedExit;

    // Adjust the probability bias based on the run of predictions
    if (pred == policyState->prevPred) {
        policyState->bias = fp16_mul(policyState->bias, policyState->decreaseFactor, precision);
    } else {
        policyState->bias = fp16_mul(policyState->bias, policyState->increaseFactor, precision);
        policyState->bias = fp16_min(policyState->bias, policyState->maxBias);
    }

    // Get the bias-adjusted exit rates
    const uint16_t one = 1 << ((uint16_t) precision);
    const uint16_t continueRate = fp16_sub(one, adjustedExitRate);
    uint16_t upperRate = get_upper_continue_rate(continueRate, policyState->bias, precision);
    uint16_t lowerRate = get_lower_continue_rate(continueRate, policyState->bias, precision);

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
    volatile uint8_t shouldExit = (uint8_t) (lfsrState[0] > continueProb);
    lfsrState[0] = lfsr_step(lfsrState[0]);

    // Set hard boundaries if we have reached to quota (on either side)
    if (remainingToExit == 0) {
        shouldExit = 0;
    } else if (remainingToExit == policyState->step) {
        shouldExit = 1;
    }

    // Update the policy parameters
    policyState->prevPred = pred;
    policyState->step -= 1;

    return shouldExit;
}

