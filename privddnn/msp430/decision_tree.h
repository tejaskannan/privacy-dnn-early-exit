#include <stdint.h>
#include "utils/array.h"
#include "utils/fixed_point.h"

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
uint8_t adaboost_inference(int16_t *inputs, struct adaboost_ensemble *ensemble, int16_t exitThreshold, uint8_t precision);

#endif
