#include "message.h"


uint16_t encode_vector(uint8_t *messageBuffer, int16_t *vec, uint16_t messageIdx, const uint16_t numElements) {
    volatile int16_t dataValue;
    uint16_t i;

    for (i = 0; i < numElements; i++) {
        dataValue = vec[VECTOR_INDEX(i)];
        messageBuffer[messageIdx] = (dataValue >> 8) & 0xFF;
        messageIdx += 1;

        messageBuffer[messageIdx] = dataValue & 0xFF;
        messageIdx += 1;
    }

    return messageIdx;
}

