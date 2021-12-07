#include "decision_tree.h"


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


void adaboost_inference_early(struct inference_result *result, int16_t *inputs, struct adaboost_ensemble *ensemble, uint8_t precision) {
    uint8_t pred;
    uint8_t i;

    // Zero out the logits
    for (i = 0; i < ensemble->numLabels; i++) {
        result->logits[i] = 0;
    }

    for (i = 0; i < ensemble->exitPoint; i++) {
	// Execute the decision tree
        pred = decision_tree_inference(inputs, ensemble->trees[i]);

	// Add the prediction to the running weights, scaled by the AdaBoost factor
        result->logits[pred] = fp32_add(ensemble->boostWeights[i], result->logits[pred]);
    }

    // Compute the output probabilities
    array32_fixed_point_softmax(result->logits, result->probs, ensemble->numLabels, precision); 

    // Compute the final prediction
    result->pred = array32_argmax(result->logits, ensemble->numLabels);
}


void adaboost_inference_full(struct inference_result *result, int16_t *inputs, struct adaboost_ensemble *ensemble, struct inference_result *exitResult) {
    uint8_t pred;
    uint8_t i;

    // Copy the logits from the exit point into the result array
    // IF MSP: Maybe do this via DMA
    for (i = 0; i < ensemble->numLabels; i++) {
        result->logits[i] = exitResult->logits[i];
    }

    for (i = ensemble->exitPoint; i < ensemble->numTrees; i++) {
	// Execute the decision tree
        pred = decision_tree_inference(inputs, ensemble->trees[i]);

	// Add the prediction to the running weights, scaled by the AdaBoost factor
        result->logits[pred] = fp32_add(ensemble->boostWeights[i], result->logits[pred]);
    }

    result->pred = array32_argmax(result->logits, ensemble->numLabels);
}
