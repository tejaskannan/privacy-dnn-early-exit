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

