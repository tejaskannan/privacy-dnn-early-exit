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


int16_t fp16_max(int16_t x, int16_t y) {
    int16_t comp = (x >= y);
    return comp * x + (1 - comp) * y;
}


int16_t fp16_min(int16_t x, int16_t y) {
    int16_t comp = (x <= y);
    return comp * x + (1 - comp) * y;
}


int16_t fp16_sub(int16_t x, int16_t y) {
    return x - y;
}


int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}


int32_t fp32_sub(int32_t x, int32_t y) {
    return x - y;
}


int32_t fp32_mul(int32_t x, int32_t y, uint8_t precision) {
    return (x * y) >> precision;
}


int32_t fp32_div(int32_t x, int32_t y, uint8_t precision) {
    return ((x << precision) / y);
}


int32_t fp32_exp(int32_t x, uint8_t precision) {
    const int32_t threeEighths = 3 << (precision - 3);
    const int32_t negOne = -1 * (1 << precision);
    const int32_t negOneThreeQuarters = negOne - (3 << (precision - 2));
    const int32_t negTwoThreeQuarters = negOneThreeQuarters + negOne;
    const int32_t negThreeHalf = -1 * (3 << precision) - (1 << (precision - 1));
    const int32_t negFive = -1 * (5 << precision);

    if (x >= threeEighths) {
	    const int32_t two = 2 << precision;
	    const int32_t elevenSixteenths = 11 << (precision - 4);
        return two + fp32_mul(two, x - elevenSixteenths, precision);
    } else if (x >= -1 * threeEighths) {
	    const int32_t one = 1 << precision;
	    return one + x;
    } else if (x >= negOne) {
        const int32_t oneHalf = 1 << (precision - 1);
	    const int32_t elevenSixteenths = 11 << (precision - 4);
	    return oneHalf + fp32_mul(oneHalf, x + elevenSixteenths, precision);
    } else if (x >= negOneThreeQuarters) {
	    const int32_t oneFourth = 1 << (precision - 2);
	    const int32_t threeEighths = 3 << (precision - 3);
	    const int32_t one = 1 << precision;
	    return oneFourth + fp32_mul(oneFourth, x + one + threeEighths, precision);
    } else if (x >= negTwoThreeQuarters) {
        const int32_t oneEighth = 1 << (precision - 3);
	    const int32_t oneSixteenth = 1 << (precision - 4);
	    const int32_t two = 2 << precision;
	    return oneEighth + fp32_mul(oneEighth, x + two + oneSixteenth, precision);
    }  else if (x >= negThreeHalf) {
        const int32_t threeSixtyFourths = 3 << (precision - 6);
        const int32_t three = 3 << precision;
        return threeSixtyFourths + fp32_mul(threeSixtyFourths, x + three, precision);
    } else if (x >= negFive) {
        const int32_t four = 4 << precision;
	    const int32_t oneSixtyFourth = 1 << (precision - 6);
	    return oneSixtyFourth + fp32_mul(oneSixtyFourth, x + four, precision);
    } else {
	    return 0;
    }
}
