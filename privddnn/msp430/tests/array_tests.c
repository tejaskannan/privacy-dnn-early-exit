#include "array_tests.h"

int main(void) {

    printf("Testing Array Max.\n");
    test_array_max();
    printf("\tPassed.\n");

    printf("Testing Array Exp Sum.\n");
    test_array_exp_sum();
    printf("\tPassed.\n");

    printf("Testing Array Softmax.\n");
    test_array_softmax();
    printf("\tPassed.\n");
}


void test_array_max(void) {
    int32_t array1[] = { 4, -1, 2, 5 };
    assert(array32_max(array1, 4) == 5);

    int32_t array2[] = { -9, 12, -459, 11, 5 };
    assert(array32_max(array2, 5) == 12);

    int32_t array3[] = { -29, -19, -1, -98, -100, -12 };
    assert(array32_max(array3, 6) == -1);
}


void test_array_exp_sum(void) {
    int32_t array1[] = { 4096, -1024, 2048, 5120 };
    assert(array32_fixed_point_exp_sum(array1, 5120, 4, 10) == 1384);
}


void test_array_softmax(void) {
    int32_t array1[] = { 3072, 4096, 1024 };
    int32_t result1[3];
    array32_fixed_point_softmax(array1, result1, 3, 10);

    assert(result1[0] == 260);
    assert(result1[1] == 757);
    assert(result1[2] == 7);

    int32_t array2[] = { 3072, 4096, 10240, -2048 };
    int32_t result2[4];
    array32_fixed_point_softmax(array2, result2, 4, 10);

    assert(result2[0] == 0);
    assert(result2[1] == 0);
    assert(result2[2] == 1024);
    assert(result2[3] == 0);
}
