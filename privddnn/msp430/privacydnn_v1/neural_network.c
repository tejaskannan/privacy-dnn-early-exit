#include "neural_network.h"

static int16_t HIDDEN_BUFFER_0[512];
static int16_t LOGITS[NUM_LABELS * VECTOR_COLS];

#ifndef IS_RANDOM
static int32_t PROBS[NUM_LABELS];
#endif

#define UNUSED(X) (void)(X)


struct matrix *dnn_layer(struct matrix *result, struct matrix *inputs, struct block_matrix *weights, struct matrix *bias, uint8_t precision, uint8_t shouldActivate) {
    /**
     *  Executes a single dense layer with the given inputs and parameters.
     */ 
    block_matrix_vector_prod(result, weights, inputs, precision);
    vector_add(result, result, bias);

    if (shouldActivate) {
        vector_relu(result, result);
    }
    
    return result;
}


struct inference_result *branchynet_dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision, struct exit_policy *policy, struct cgr_state *policyState) {
    #ifndef IS_CGR_MAX_PROB
    UNUSED(policyState);
    #endif

    volatile uint8_t shouldExit = 1;

    // Apply the first hidden layer
    struct matrix hidden0 = { HIDDEN_BUFFER_0, DENSE_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden0, inputs, &DENSE_W, &DENSE_B, precision, 1);

    #ifdef IS_RANDOM
    shouldExit = random_should_exit(SCALED_EXIT_RATES[0], policy->randState);
    #endif

    // Apply the first output layer. For non-random policies, we always execute this
    // layer to assess the probabilites. For random exiting, we already know
    // the decision beforehand and can therefore save on this layer when NOT exiting.
    struct matrix logits = { LOGITS, NUM_LABELS, VECTOR_COLS };

    if (shouldExit) {
        dnn_layer(&logits, &hidden0, &OUTPUT0_W, &OUTPUT0_B, precision, 0);
    }

    // Evaluate the exit policy
    #ifdef IS_MAX_PROB
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[0]);
    #elif defined(IS_LABEL_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);
    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[result->pred]);
    #elif defined(IS_CGR_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = cgr_should_exit(PROBS[result->pred], result->pred, SCALED_EXIT_RATES[0], policy->thresholds[result->pred], policy->randState, policyState, 0, NUM_OUTPUTS, precision);
    #endif

    // Save the hidden state in a 1d array
    dma_load(result->state, hidden0.data, hidden0.numRows * hidden0.numCols);

    //shouldExit = 0;  // Hard code exiting

    // Exit early if specified
    if (shouldExit) {
        #ifdef IS_RANDOM
        result->pred = vector_argmax(&logits);
        #elif defined(IS_CGR_MAX_PROB)
        update_cgr(policyState, 0);
        #endif

        result->outputIdx = 0;
        return result;
    }

    #ifdef IS_CGR_MAX_PROB
    update_cgr(policyState, 1);
    #endif

    // The system does not have the prediction at this point, so we signal to continue execution
    // on the server
    result->pred = 0;
    result->outputIdx = 1;

    return result;
}
