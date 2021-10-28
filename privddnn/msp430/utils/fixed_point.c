#include "fixed_point.h"


int16_t fp16_add(int16_t x, int16_t y) {
    return x + y;
}


int16_t fp16_mul(int16_t x, int16_t y, uint8_t precision) {
    return (x * y) >> precision;
}


int16_t fp16_div(int16_t x, int16_t y, uint8_t precision) {
    return ((x << precision) / y);
}


int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}


int32_t fp32_mul(int32_t x, int32_t y, uint8_t precision) {
    return (x * y) >> precision;
}


int32_t fp32_div(int32_t x, int32_t y, uint8_t precision) {
    return ((x << precision) / y);
}
