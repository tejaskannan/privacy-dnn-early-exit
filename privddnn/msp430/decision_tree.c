#include "decision_tree.h"

#define BUFFER_SIZE 16
static int32_t WEIGHTS_BUFFER[BUFFER_SIZE];
static int32_t PROBS_BUFFER[BUFFER_SIZE];


uint8_t decision_tree_inference(int16_t *inputs, struct decision_tree *tree) {
    // Start inference at the root node
    volatile uint8_t treeIdx = 0;
    volatile int8_t featureIdx;
    volatile int16_t threshold;

    while ((tree->leftChildren[treeIdx] >= 0) && (tree->rightChildren[treeIdx] >= 0)) {
	threshold = tree->thresholds[treeIdx];
	featureIdx = tree->features[treeIdx];

	if (inputs[featureIdx] <= threshold) {
	    treeIdx = tree->leftChildren[treeIdx];
	} else {
	    treeIdx = tree->rightChildren[treeIdx];
	}
    }

    return tree->predictions[treeIdx];
}


uint8_t adaboost_inference(int16_t *inputs, struct adaboost_ensemble *ensemble, int16_t exitThreshold, uint8_t precision) {
    // Zero out the probability buffers
    volatile uint16_t i;
    for (i = 0; i < BUFFER_SIZE; i++) {
         PROBS_BUFFER[i] = 0;
	 WEIGHTS_BUFFER[i] = 0;
    }

    // Execute the decision trees in order
    uint8_t pred;
    int32_t predWeight;
    int32_t currentWeight;

    for (i = 0; i < ensemble->numTrees; i++) {
	// Perform early-exiting if possible
	if (i == ensemble->exitPoint) {
	    array32_fixed_point_normalize(WEIGHTS_BUFFER, PROBS_BUFFER, ensemble->numLabels, precision);
	    pred = array32_argmax(PROBS_BUFFER, ensemble->numLabels);

	    // For now, this only support MaxProb exiting
	    if (PROBS_BUFFER[pred] >= exitThreshold) {
	        return pred;
	    }
	}

	// Execute the decision tree
        pred = decision_tree_inference(inputs, ensemble->trees[i]);

	// Add the prediction to the running weights, scaled by the AdaBoost factor
	currentWeight = WEIGHTS_BUFFER[pred];
        predWeight = fp32_mul(currentWeight, ensemble->boostWeights[i], precision);
	WEIGHTS_BUFFER[pred] = fp32_add(predWeight, currentWeight);
    }

    // Compute the final prediction
    array32_fixed_point_normalize(WEIGHTS_BUFFER, PROBS_BUFFER, ensemble->numLabels, precision);
    return array32_argmax(PROBS_BUFFER, ensemble->numLabels);
}
