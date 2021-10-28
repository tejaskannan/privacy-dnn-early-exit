#include "decision_tree.h"


//static const uint16_t BUFFER_SIZE = 16;
//static int32_t PROBS_BUFFER[BUFFER_SIZE];


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


//uint8_t adaboost_inference(int16_t *inputs, struct adaboost_ensemble *ensemble, int16_t exitThreshold) {
//    // Zero out the probability buffer
//    uint16_t i;
//    for (i = 0; i < BUFFER_SIZE; i++) {
//         PROBS_BUFFER[i] = 0;
//    }
//
//    // Execute the decision trees in order
//    
//
//}
