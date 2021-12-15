#include "policy.h"

#define BUFFER_SIZE 16
int16_t PROBS_BUFFER[BUFFER_SIZE];
uint8_t INDEX_BUFFER[BUFFER_SIZE];


uint8_t max_prob_should_exit(struct inference_result *earlyResult, int16_t exitThreshold) {
    const int16_t maxProb = (int16_t) (earlyResult->probs[earlyResult->pred]);
    return (uint8_t) (maxProb >= exitThreshold);
}


uint8_t random_should_exit(uint16_t exitRate, uint16_t lfsrState) {
    return (uint8_t) ((lfsrState & 0x7FFF) <= exitRate);
}


uint8_t *buffered_max_prob_should_exit(uint8_t *results, struct inference_result *earlyResults, uint16_t lfsrState, uint16_t elevCount, uint16_t elevRem, uint16_t windowSize) {
    // Determine the random adjustment to the elevate count
    uint16_t shouldIncrease = ((lfsrState & 0x7FFF) <= elevRem);
    elevCount += shouldIncrease;

    // Place the results into the buffers in a sorted order
    int16_t tempProb;
    uint8_t tempIdx;
    uint8_t pred;

    uint8_t i, j, k;
    for (i = 0; i < windowSize; i++) {
        results[i] = 1;  // By default, exit early

	pred = earlyResults[i].pred;
	PROBS_BUFFER[i] = earlyResults[i].probs[pred];
	INDEX_BUFFER[i] = i;

        for (j = i; j > 0; j--) {
            k = j - 1;

            if (PROBS_BUFFER[k] <= PROBS_BUFFER[j]) {
		break;
	    }

	    tempProb = PROBS_BUFFER[k];
	    PROBS_BUFFER[k] = PROBS_BUFFER[j];
	    PROBS_BUFFER[j] = tempProb;

 	    tempIdx = INDEX_BUFFER[k];
	    INDEX_BUFFER[k] = INDEX_BUFFER[j];
	    INDEX_BUFFER[j] = tempIdx;
	}
    }

    // Set the indices in which to use the full model
    uint8_t elevIdx;
    for (i = 0; i < elevCount; i++) {
        elevIdx = INDEX_BUFFER[i];
        results[elevIdx] = 0;	
    }

    return results;
}

