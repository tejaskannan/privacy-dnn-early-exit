#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "decision_tree.h"
#include "parameters.h"

#define INPUT_BUFFER_SIZE 1024
#define LABEL_BUFFER_SIZE 16
#define NUM_INPUT_FEATURES 16
#define THRESHOLD 0
#define PRECISION 10

int main(void) {

    const char *inputPath = "../data/pen_digits/pen_digits_10_inputs.txt";
    FILE *inputFile = fopen(inputPath, "r"); 
    char inputBuffer[INPUT_BUFFER_SIZE];

    const char *labelPath = "../data/pen_digits/pen_digits_10_labels.txt";
    FILE *labelFile = fopen(labelPath, "r");
    char labelBuffer[LABEL_BUFFER_SIZE];

    int16_t inputFeatures[NUM_INPUT_FEATURES];
    uint8_t featureIdx = 0;
    uint8_t label = 0;
    uint32_t isCorrect = 0;
    uint32_t totalCount = 0;

    while (fgets(inputBuffer, INPUT_BUFFER_SIZE, inputFile) != NULL) {

	char *token = strtok(inputBuffer, " ");
        for (featureIdx = 0; featureIdx < NUM_INPUT_FEATURES; featureIdx++) {
	    inputFeatures[featureIdx] = atoi(token);
	    token = strtok(NULL, " ");
	}

	// Fetch the label
	fgets(labelBuffer, LABEL_BUFFER_SIZE, labelFile);
	label = atoi(labelBuffer);

	uint8_t pred = adaboost_inference(inputFeatures, &ENSEMBLE, THRESHOLD, PRECISION);
	isCorrect += (pred == label);
	totalCount += 1;
    }

    printf("Accuracy: %d / %d\n", isCorrect, totalCount);
    return 0;
}
