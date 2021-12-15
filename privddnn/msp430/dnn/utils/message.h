#include <stdint.h>
#include "inference_result.h"
#include "../matrix.h"

#ifndef MESSAGE_H_
#define MESSAGE_H_

#define EXIT_BYTE 0x12u
#define ELEVATE_BYTE 0x34u

uint16_t create_exit_message(uint8_t *messageBuffer, struct inference_result *inferenceResult);
uint16_t create_elevate_message(uint8_t *messageBuffer, struct matrix *hidden, struct matrix *inputs, const uint16_t bufferSize);

#endif

