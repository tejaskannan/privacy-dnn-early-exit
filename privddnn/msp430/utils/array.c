#include "array.h"


uint8_t array32_argmax(int32_t *array, uint8_t n) {
    volatile int32_t arrayValue;
    volatile int32_t maxValue = INT32_MIN;
    volatile uint8_t maxIdx = 0;

    uint8_t i;
    for (i = 0; i < n; i++) {
	arrayValue = array[i];
	if (arrayValue > maxValue) {
            maxValue = arrayValue;
	    maxIdx = i;
	}
    }

    return maxIdx;
}


int32_t array32_max(int32_t *array, uint8_t n) {
    return array[array32_argmax(array, n)];
}


int32_t array32_fixed_point_sum(int32_t *array, uint8_t n) {
    volatile int32_t sum = 0;

    uint8_t i;
    for (i = 0; i < n; i++) {
        sum = fp32_add(sum, array[i]);
    }

    return sum;
}


void array32_fixed_point_normalize(int32_t *src, int32_t *dst, uint8_t n, uint8_t precision) {
    int32_t sum = array32_fixed_point_sum(src, n);

    uint8_t i;
    for (i = 0; i < n; i++) {
        dst[i] = fp32_div(src[i], sum, precision);
    }
}
