#include <stdint.h>
#include "fixed_point.h"

#ifndef ARRAY_H_
#define ARRAY_H_

uint8_t array32_argmax(int32_t *array, uint8_t n);
uint32_t array32_max(int32_t *array, uint8_t n);
int32_t array32_fixed_point_sum(int32_t *array, uint8_t n);
void array32_fixed_point_normalize(int32_t *src, int32_t *dst, uint8_t n, uint8_t precision);

#endif
