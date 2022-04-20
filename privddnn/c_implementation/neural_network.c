#include "neural_network.h"

static int16_t HIDDEN_BUFFER_0[512];
static int16_t HIDDEN_BUFFER_1[512];
static int16_t CONCAT_BUFFER[128];
static int16_t LOGITS[NUM_LABELS * VECTOR_COLS];
static int32_t PROBS[NUM_LABELS];


struct matrix *dnn_layer(struct matrix *result, struct matrix *inputs, struct matrix *weights, struct matrix *bias, uint8_t precision, uint8_t shouldActivate) {
    /**
     *  Executes a single dense layer with the given inputs and parameters.
     */ 
    matrix_vector_prod(result, weights, inputs, precision);
    vector_add(result, result, bias);

    if (shouldActivate) {
        vector_relu(result, result);
    }
    
    return result;
}



struct inference_result *branchynet_dnn(struct inference_result *result, struct matrix *inputs, uint8_t precision, struct exit_policy *policy) {
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
    shouldExit = random_should_exit(EXIT_RATE, policy->lfsrStates);
    #endif

    // Exit early if specified
    if (shouldExit) {
        #if defined(IS_MAX_PROB) || defined(IS_RANDOM)
        result->pred = vector_argmax(&logits);
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
    shouldExit = max_prob_should_exit(PROBS, policy->thresholds[0]);
    #elif defined(IS_LABEL_MAX_PROB)
    vector_softmax(PROBS, &logits, precision);
    pred = vector_argmax(&logits);
    shouldExit = label_max_prob_should_exit(PROBS, policy->thresholds[pred]);
    #elif defined(IS_RANDOM)
    shouldExit = random_should_exit(policy->exitRate, policy->lfsrStates);
    #endif

    // Exit early if specified
    if (shouldExit) {
        #if defined(IS_MAX_PROB) || defined(IS_RANDOM)
        pred = vector_argmax(&logits);
        #endif

        result->pred = pred;
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

    result->pred = vector_argmax(&logits);
    result->outputIdx = NUM_OUTPUTS - 1;

    // Return the inference result
    return result;
}
