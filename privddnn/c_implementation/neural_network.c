#include "neural_network.h"

static int16_t HIDDEN_BUFFER_0[512];
static int16_t HIDDEN_BUFFER_1[512];
static int16_t CONCAT_BUFFER[128];
static int16_t LOGITS[NUM_LABELS * VECTOR_COLS];
static int32_t PROBS[NUM_LABELS];

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

struct inference_result *dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision) {
    // Apply the first hidden layer
    struct matrix hidden0 = { HIDDEN_BUFFER_0, DENSE_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden0, inputs, &DENSE_W, &DENSE_B, precision, 1);

    // Concatenate the inputs with the initial hidden state
    struct matrix concat = { CONCAT_BUFFER, inputs->numRows + hidden0.numRows, VECTOR_COLS };
    vector_concat(&concat, inputs, &hidden0);

    // Apply the other hidden layers
    struct matrix hidden1 = { HIDDEN_BUFFER_1, DENSE_1_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden1, &concat, &DENSE_1_W, &DENSE_1_B, precision, 1);

    struct matrix hidden2 = { HIDDEN_BUFFER_0, DENSE_2_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden2, &hidden1, &DENSE_2_W, &DENSE_2_B, precision, 1);

    struct matrix hidden3 = { HIDDEN_BUFFER_1, DENSE_3_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden3, &hidden2, &DENSE_3_W, &DENSE_3_B, precision, 1);

    // Compute the logits
    struct matrix logits = { LOGITS, NUM_LABELS, VECTOR_COLS };
    dnn_layer(&logits, &hidden3, &OUTPUT2_W, &OUTPUT2_B, precision, 0);

    result->pred = vector_argmax(&logits);
    result->outputIdx = 2;
    return result;
}


struct inference_result *branchynet_dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision, struct exit_policy *policy, struct cgr_state *policyState) {
    #ifndef IS_CGR_MAX_PROB
    UNUSED(policyState);
    #endif

    // Apply the first hidden layer
    struct matrix hidden0 = { HIDDEN_BUFFER_0, DENSE_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden0, inputs, &DENSE_W, &DENSE_B, precision, 1);

    // Apply the first output layer
    struct matrix logits = { LOGITS, NUM_LABELS, VECTOR_COLS };
    dnn_layer(&logits, &hidden0, &OUTPUT0_W, &OUTPUT0_B, precision, 0);

    volatile uint8_t shouldExit;

    // Evaluate the exit policy
    #ifdef IS_MAX_PROB
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[0]);
    #elif defined(IS_LABEL_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);
    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[result->pred]);
    #elif defined(IS_RANDOM)
    shouldExit = random_should_exit(SCALED_EXIT_RATES[0], policy->randState);
    #elif defined(IS_CGR_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = cgr_should_exit(PROBS[result->pred], result->pred, SCALED_EXIT_RATES[0], policy->thresholds[result->pred], policy->randState, policyState, 0, NUM_OUTPUTS, precision);
    #endif

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

    // Concatenate the inputs with the initial hidden state
    struct matrix concat = { CONCAT_BUFFER, inputs->numRows + hidden0.numRows, VECTOR_COLS };
    vector_concat(&concat, inputs, &hidden0);

    struct matrix hidden1 = { HIDDEN_BUFFER_1, DENSE_1_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden1, &concat, &DENSE_1_W, &DENSE_1_B, precision, 1);

    #if NUM_OUTPUTS == 3
    // Apply the second output layer
    dnn_layer(&logits, &hidden1, &OUTPUT1_W, &OUTPUT1_B, precision, 0);

    // Execute the exit policy
    #ifdef IS_MAX_PROB
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[1]);
    #elif defined(IS_LABEL_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);
    shouldExit = max_prob_should_exit(PROBS[result->pred], policy->thresholds[NUM_LABELS + result->pred]);
    #elif defined(IS_RANDOM)
    shouldExit = random_should_exit(SCALED_EXIT_RATES[1], policy->randState);
    #elif defined(IS_CGR_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    result->pred = vector_argmax(&logits);

    shouldExit = cgr_should_exit(PROBS[result->pred], result->pred, SCALED_EXIT_RATES[1], policy->thresholds[NUM_LABELS + result->pred], policy->randState, policyState, 1, NUM_OUTPUTS, precision);
    #endif

    // Exit early if specified
    if (shouldExit) {
        #ifdef IS_RANDOM
        result->pred = vector_argmax(&logits);
        #elif defined(IS_CGR_MAX_PROB)
        update_cgr(policyState, 1);
        #endif

        result->outputIdx = 1;
        return result;
    }
    #endif 

    // Execute the remaining parts of the model
    struct matrix hidden2 = { HIDDEN_BUFFER_0, DENSE_2_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden2, &hidden1, &DENSE_2_W, &DENSE_2_B, precision, 1);

    struct matrix hidden3 = { HIDDEN_BUFFER_1, DENSE_3_W.numRows, VECTOR_COLS };
    dnn_layer(&hidden3, &hidden2, &DENSE_3_W, &DENSE_3_B, precision, 1);

    #if NUM_OUTPUTS == 2
    dnn_layer(&logits, &hidden3, &OUTPUT1_W, &OUTPUT1_B, precision, 0);
    #else
    dnn_layer(&logits, &hidden3, &OUTPUT2_W, &OUTPUT2_B, precision, 0);
    #endif

    #if defined(IS_CGR_MAX_PROB)
    update_cgr(policyState, 2);
    #endif

    result->pred = vector_argmax(&logits);
    result->outputIdx = NUM_OUTPUTS - 1;

    // Return the inference result
    return result;
}
