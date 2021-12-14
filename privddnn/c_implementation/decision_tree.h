#include <stdint.h>
#include "utils/array.h"
#include "utils/fixed_point.h"
#include "utils/inference_result.h"

#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_

struct decision_tree {
    uint8_t numNodes;
    int16_t *thresholds;
    int8_t *features;
    uint8_t *predictions;
    int8_t *leftChildren;
    int8_t *rightChildren;
};

struct adaboost_ensemble {
    uint8_t numTrees;
    uint8_t exitPoint;
    uint8_t numLabels;
    struct decision_tree **trees;
    int16_t *boostWeights;
};

uint8_t decision_tree_inference(int16_t *inputs, struct decision_tree *tree);
void adaboost_inference_early(struct inference_result *result, int16_t *inputs, struct adaboost_ensemble *ensemble, uint8_t precision);
void adaboost_inference_full(struct inference_result *result, int16_t *inputs, struct adaboost_ensemble *ensemble, struct inference_result *exitResult);

#endif
