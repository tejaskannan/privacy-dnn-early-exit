#include <stdint.h>
#include "inference_result.h"
#include "../matrix.h"

#ifndef MESSAGE_H_
#define MESSAGE_H_

#define EXIT_BYTE 0x12u
#define ELEVATE_BYTE 0x34u
#define BUFFERED_BYTE 0x56u

uint16_t create_exit_message(uint8_t *messageBuffer, struct inference_result *inferenceResult);
uint16_t create_elevate_message(uint8_t *messageBuffer, struct matrix *inputs, struct matrix *hidden, const uint16_t bufferSize);
uint16_t create_buffered_message(uint8_t *messageBuffer, struct inference_result *inferenceResult, struct matrix *inputs, struct matrix *hidden, uint8_t *shouldExit, const uint8_t windowSize, const uint16_t bufferSize);

#endif

