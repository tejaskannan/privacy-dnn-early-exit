#include "message.h"


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

    uint16_t bufferIdx = 3;
    int16_t dataValue;

    // Encode the input values
    uint8_t i;
    for (i = 0; i < inputs->numRows; i++) {
        if (bufferIdx >= (bufferSize - 1)) {
            return bufferIdx;
        }

        dataValue = inputs->data[VECTOR_INDEX(i)];
        messageBuffer[bufferIdx] = (dataValue >> 8) & 0xFF;
        bufferIdx += 1;

        messageBuffer[bufferIdx] = dataValue & 0xFF;
        bufferIdx += 1;
    }

    // Encode the hidden state values
    for (i = 0; i < hidden->numRows; i++) {
        if (bufferIdx >= (bufferSize - 1)) {
            return bufferIdx;
        }

        dataValue = hidden->data[VECTOR_INDEX(i)];
        messageBuffer[bufferIdx] = (dataValue >> 8) & 0xFF;
        bufferIdx += 1;

        messageBuffer[bufferIdx] = dataValue & 0xFF;
        bufferIdx += 1;
    }

    return bufferIdx;
}

