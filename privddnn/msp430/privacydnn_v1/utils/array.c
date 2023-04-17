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
    volatile int32_t maxValue = INT32_MIN;
    volatile int32_t arrayValue;

    uint8_t i;
    for (i = 0; i < n; i++) {
        arrayValue = array[VECTOR_INDEX(i)];
	    if (arrayValue > maxValue) {
	        maxValue = arrayValue;
        }
    }

    return maxValue;
}


int32_t array32_fixed_point_exp_sum(int32_t *array, int32_t max, uint8_t n, uint8_t precision) {
    volatile int32_t sum = 0;
    volatile int32_t element;

    uint8_t i;
    for (i = 0; i < n; i++) {
	    element = fp32_sub(array[VECTOR_INDEX(i)], max);
        sum = fp32_add(sum, fp32_exp(element, precision));
    }

    return sum;
}


void array32_fixed_point_softmax(int32_t *src, int32_t *dst, uint8_t n, uint8_t precision) {
    const int32_t max = array32_max(src, n);
    const int32_t sum = array32_fixed_point_exp_sum(src, max, n, precision);
    volatile int32_t element;
    volatile int32_t cumSum = 0;

    uint8_t i;
    for (i = 0; i < (n - 1); i++) {
	    element = fp32_sub(src[VECTOR_INDEX(i)], max);
        dst[i] = fp32_div(fp32_exp(element, precision), sum, precision);
	    cumSum = fp32_add(dst[i], cumSum);
    }

    dst[n-1] = fp32_sub((1 << precision), cumSum);
}
