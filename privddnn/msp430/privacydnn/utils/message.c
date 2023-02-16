#include "message.h"


uint16_t encode_vector(uint8_t *messageBuffer, struct matrix *vec, uint16_t messageIdx, const uint16_t bufferSize) {
    volatile int16_t dataValue;
    uint8_t i;

    for (i = 0; i < vec->numRows; i++) {
        if (messageIdx >= (bufferSize - 1)) {
            return messageIdx;
        }

        dataValue = vec->data[VECTOR_INDEX(i)];
        messageBuffer[messageIdx] = (dataValue >> 8) & 0xFF;
        messageIdx += 1;

        messageBuffer[messageIdx] = dataValue & 0xFF;
        messageIdx += 1;
    }

    return messageIdx;
}


uint16_t create_exit_message(uint8_t *messageBuffer, struct inference_result *inferenceResult) {
    messageBuffer[0] = EXIT_BYTE;
    messageBuffer[1] = inferenceResult->pred;
    return 2;
}


uint16_t create_elevate_message(uint8_t *messageBuffer, struct matrix *inputs, struct matrix *hidden, const uint16_t bufferSize) {
    // Signal that this is an 'elevation' message
    messageBuffer[0] = ELEVATE_BYTE;

    // Set the number of bytes required for (1) the input features, (2) the hidden result
    messageBuffer[1] = inputs->numRows * sizeof(int16_t);
    messageBuffer[2] = hidden->numRows * sizeof(int16_t);

    uint16_t messageIdx = 3;

    // Encode the input values
    messageIdx = encode_vector(messageBuffer, inputs, messageIdx, bufferSize);

    // Encode the hidden state values
    messageIdx = encode_vector(messageBuffer, hidden, messageIdx, bufferSize);

    return messageIdx;
}


uint16_t create_buffered_message(uint8_t *messageBuffer, struct inference_result *inferenceResult, struct matrix *inputs, struct matrix *hidden, uint8_t *shouldExit, const uint8_t windowSize, const uint16_t bufferSize) {
    // Signal that this is a buffered message
    messageBuffer[0] = BUFFERED_BYTE;

    // Set the number of bytes required for the (1) input features, (2) the hidden result, and (3) the window size
    messageBuffer[1] = inputs->numRows * sizeof(uint16_t);
    messageBuffer[2] = hidden->numRows * sizeof(uint16_t);
    messageBuffer[3] = windowSize;

    // Write a bitmask signalling 'which' samples got elevated
    uint16_t messageIdx = 4;
    uint8_t currentByte = 0;
    uint8_t offset = 0;
    uint8_t bitValue = 0;

    uint8_t i;
    for (i = 0; i < windowSize; i++) {
        if (offset == 8) {
            messageBuffer[messageIdx] = currentByte;

            currentByte = 0;
            messageIdx += 1;
            offset = 0;
        }

        currentByte |= (shouldExit[i] & 1) << offset;
        offset += 1;
    }

    if ((windowSize & 0x7) != 0) {
        messageBuffer[messageIdx] = currentByte;
        messageIdx += 1;
    }

    // Write the prediction from each sample
    for (i = 0; i < windowSize; i++) {
        messageBuffer[messageIdx] = (inferenceResult + i)->pred;
        messageIdx += 1;
    }

    // Write the input and hidden values for each sample (if not exiting)
    for (i = 0; i < windowSize; i++) {
        if (!shouldExit[i]) {
            messageIdx = encode_vector(messageBuffer, inputs + i, messageIdx, bufferSize);
            messageIdx = encode_vector(messageBuffer, hidden + i, messageIdx, bufferSize);
        }
    }

    return messageIdx;
}



