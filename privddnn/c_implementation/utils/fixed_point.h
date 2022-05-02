#include <stdint.h>

#ifndef FIXED_POINT_H_
#define FIXED_POINT_H_

int16_t fp16_add(int16_t x, int16_t y);
int16_t fp16_mul(int16_t x, int16_t y, uint8_t precision);
int16_t fp16_div(int16_t x, int16_t y, uint8_t precision);
int16_t fp16_exp(int16_t x, uint8_t precision);  // Piecewise Linear approximation to exp(x)
int16_t fp16_max(int16_t x, int16_t y);
int16_t fp16_min(int16_t x, int16_t y);
int16_t fp16_sub(int16_t x, int16_t y);

int32_t fp32_add(int32_t x, int32_t y);
int32_t fp32_sub(int32_t x, int32_t y);
int32_t fp32_mul(int32_t x, int32_t y, uint8_t precision);
int32_t fp32_div(int32_t x, int32_t y, uint8_t precision);
int32_t fp32_exp(int32_t x, uint8_t precision);

#endif
