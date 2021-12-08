#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "decision_tree.h"
#include "policy.h"
#include "parameters.h"
#include "utils/lfsr.h"

#define INPUT_BUFFER_SIZE 1024

int main(void) {

    const char *inputPath = "../data/pen_digits/pen_digits_10_inputs.txt";
    FILE *inputFile = fopen(inputPath, "r"); 
    char inputBuffer[INPUT_BUFFER_SIZE];

    const char *labelPath = "../data/pen_digits/pen_digits_10_labels.txt";
    FILE *labelFile = fopen(labelPath, "r");
    char labelBuffer[NUM_LABELS];

    int16_t inputFeatures[NUM_INPUT_FEATURES];
    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t numExit = 0;
    uint32_t totalCount = 0;
    uint16_t lfsrState = 3798;

#ifdef IS_BUFFERED_MAX_PROB
    int16_t windowInputFeatures[NUM_INPUT_FEATURES * WINDOW_SIZE];
    uint8_t windowLabels[WINDOW_SIZE];
    uint16_t i;
    uint16_t windowIdx = 0;

    uint8_t exitResults[WINDOW_SIZE];

    int32_t earlyLogits[NUM_LABELS * WINDOW_SIZE];
    int32_t earlyProbs[NUM_LABELS * WINDOW_SIZE];
    int32_t fullLogits[NUM_LABELS * WINDOW_SIZE];
    int32_t fullProbs[NUM_LABELS * WINDOW_SIZE];

    struct inference_result earlyResult[WINDOW_SIZE];
    struct inference_result fullResult[WINDOW_SIZE];

    for (i = 0; i < WINDOW_SIZE; i++) {
        earlyResult[i].logits = earlyLogits + (i * NUM_LABELS);
	earlyResult[i].probs = earlyProbs + (i * NUM_LABELS);
	earlyResult[i].pred = 0;

    	fullResult[i].logits = fullLogits + (i * NUM_LABELS);
	fullResult[i].probs = fullProbs + (i * NUM_LABELS);
	fullResult[i].pred = 0;
    }
#else
    int32_t earlyLogits[NUM_LABELS];
    int32_t earlyProbs[NUM_LABELS];
    int32_t fullLogits[NUM_LABELS];
    int32_t fullProbs[NUM_LABELS];

    struct inference_result earlyResult = { earlyLogits, earlyProbs, 0 };
    struct inference_result fullResult = { fullLogits, fullProbs, 0 };
#endif

    uint8_t pred;
    uint8_t shouldExit;

    while (fgets(inputBuffer, INPUT_BUFFER_SIZE, inputFile) != NULL) {

	char *token = strtok(inputBuffer, " ");
        for (featureIdx = 0; featureIdx < NUM_INPUT_FEATURES; featureIdx++) {
	    inputFeatures[featureIdx] = atoi(token);
	    token = strtok(NULL, " ");
	}

	// Fetch the label
	fgets(labelBuffer, NUM_LABELS, labelFile);
	label = atoi(labelBuffer);

#ifdef IS_BUFFERED_MAX_PROB
	adaboost_inference_early(earlyResult + windowIdx, inputFeatures, &ENSEMBLE, PRECISION);

	// Copy inputFeatures into windowInputFeatures[windowIdx]
	for (i = 0; i < NUM_INPUT_FEATURES; i++) {
            windowInputFeatures[windowIdx * NUM_INPUT_FEATURES + i] = inputFeatures[i];
	}

	// Save the label for proper accuracy logging (not needed at runtime in practice)
	windowLabels[windowIdx] = label;

	// Increment the window idx for the next sample	
	windowIdx += 1;

	// On the last element in the window, perform the buffered exiting
	if (windowIdx == WINDOW_SIZE) {
            buffered_max_prob_should_exit(exitResults, earlyResult, lfsrState, ELEVATE_COUNT, ELEVATE_REMAINDER, WINDOW_SIZE);

	    for (i = 0; i < WINDOW_SIZE; i++) {
                if (!exitResults[i]) {
    		    adaboost_inference_full(fullResult + i, windowInputFeatures + i * NUM_INPUT_FEATURES, &ENSEMBLE, earlyResult + i);
                    pred = fullResult[i].pred;
		} else {
                    numExit += 1;
		    pred = earlyResult[i].pred;
		}

		isCorrect += (pred == windowLabels[i]);
	    }

            lfsrState = lfsr_step(lfsrState);
	    windowIdx = 0;
	}

	totalCount += 1;
#else
	adaboost_inference_early(&earlyResult, inputFeatures, &ENSEMBLE, PRECISION);

        #ifdef IS_MAX_PROB
        shouldExit = max_prob_should_exit(&earlyResult, THRESHOLD);
        #elif defined(IS_RANDOM)
        shouldExit = random_should_exit(EXIT_RATE, lfsrState);
        lfsrState = lfsr_step(lfsrState);
        #endif

	if (!shouldExit) {
            adaboost_inference_full(&fullResult, inputFeatures, &ENSEMBLE, &earlyResult);
            pred = fullResult.pred;
	} else {
	    numExit += 1;
            pred = earlyResult.pred;
	}

	isCorrect += (pred == label);
	totalCount += 1;
#endif
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    printf("Exit Rate: %d / %d\n", numExit, totalCount);
    return 0;
}
