#include "neural_network.h"

static int16_t LOGITS[NUM_LABELS];

struct inference_result *neural_network(struct inference_result *result, struct matrix *hiddenResult, struct matrix *inputs, uint8_t precision) {
    // Apply the hidden layer
    matrix_vector_prod(hiddenResult, &W0, inputs, precision);
    vector_add(hiddenResult, hiddenResult, &B0);
    vector_relu(hiddenResult, hiddenResult);

    // Apply the output layer
    struct matrix logits = { LOGITS, NUM_LABELS, 1 };
    matrix_vector_prod(&logits, &W1, hiddenResult, precision);
    vector_add(&logits, &logits, &B1);

    result->pred = vector_argmax(&logits);

    // Compute the output probabilities. We project the logits up to 32 bit quantities for better
    // numerical stability during the softmax computation.
    uint8_t i;
    for (i = 0; i < NUM_LABELS; i++) {
        result->logits[i] = (int32_t) logits.data[VECTOR_INDEX(i)];
    }

    array32_fixed_point_softmax(result->logits, result->probs, NUM_LABELS, precision);

    // Return the resulting log probabilities
    return result;
}
