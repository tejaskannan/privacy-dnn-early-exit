#include <stdint.h>
#include "inference_result.h"
#include "../matrix.h"

#ifndef MESSAGE_H_
#define MESSAGE_H_

uint16_t encode_vector(uint8_t *messageBuffer, int16_t *vec, uint16_t messageIdx, const uint16_t bufferSize);

#endif

