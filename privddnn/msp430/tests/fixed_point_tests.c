#include "fixed_point_tests.h"


int main(void) {
    printf("Testing 16-bit Add.\n");
    test_add_16();
    printf("\tPassed.\n");

    printf("Testing 16-bit Multiply.\n");
    test_mul_16();
    printf("\tPassed.\n");

    printf("Testing 16-bit Division.\n");
    test_div_16();
    printf("\tPassed.\n");

    printf("Testing 32-bit exp.\n");
    test_exp_32();
    printf("\tPassed.\n");
}


void test_add_16(void) {
    assert(fp16_add(10, 6) == 16);
    assert(fp16_add(-5, 1) == -4);
    assert(fp16_add(1024, -1024) == 0);
}


void test_mul_16(void) {
    assert(fp16_mul(1024, 512, 10) == 512);
    assert(fp16_mul(256, -256, 7) == -512);
    assert(fp16_mul(-16, -64, 5) == 32);
}


void test_div_16(void) {
    assert(fp16_div(1024, 512, 10) == 2048);
    assert(fp16_div(256, -256, 7) == -128);
    assert(fp16_div(-16, -64, 5) == 8);
}

void test_exp_32(void) {
    assert(fp32_exp(0, 10) == 1024);
    assert(fp32_exp(358, 9) == 1036);
    assert(fp32_exp(-512, 10) == 608);
    assert(fp32_exp(-768, 9) == 112);
    assert(fp32_exp(-1024, 9) == 68);
    assert(fp32_exp(-2048, 9) == 0);
}

