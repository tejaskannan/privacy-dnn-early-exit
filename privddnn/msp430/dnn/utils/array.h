#include <stdint.h>
#include "fixed_point.h"

#ifndef ARRAY_H_
#define ARRAY_H_

uint8_t array32_argmax(int32_t *array, uint8_t n);
int32_t array32_max(int32_t *array, uint8_t n);
int32_t array32_fixed_point_exp_sum(int32_t *array, int32_t max, uint8_t n, uint8_t precision);
void array32_fixed_point_softmax(int32_t *src, int32_t *dst, uint8_t n, uint8_t precision);

#endif
