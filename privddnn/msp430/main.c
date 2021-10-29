#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "decision_tree.h"
#include "parameters.h"

#define BUFFER_SIZE 1024
#define NUM_INPUT_FEATURES 16
#define THRESHOLD 1024
#define PRECISION 10

int main(void) {

    const char *inputPath = "../data/pen_digits/pen_digits_10_inputs.txt";
    FILE *inputFile = fopen(inputPath, "r"); 
    char buffer[BUFFER_SIZE];

    int16_t inputFeatures[NUM_INPUT_FEATURES];
    uint8_t featureIdx = 0;

    while (fgets(buffer, BUFFER_SIZE, inputFile) != NULL) {

	char *token = strtok(buffer, " ");
        for (featureIdx = 0; featureIdx < NUM_INPUT_FEATURES; featureIdx++) {
	    inputFeatures[featureIdx] = atoi(token);
	    printf("%d ", inputFeatures[featureIdx]);
	    token = strtok(NULL, " ");
	}

	uint8_t pred = adaboost_inference(inputFeatures, &ENSEMBLE, THRESHOLD, PRECISION);

	break;
    }


    return 0;
}
