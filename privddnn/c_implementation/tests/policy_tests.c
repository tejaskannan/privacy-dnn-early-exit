#include "policy_tests.h"

int main(void) {
    printf("Testing Upper Bounds.\n");
    test_upper_bound();
    printf("\tPassed.\n");

    printf("Testing Lower Bounds.\n");
    test_lower_bound();
    printf("\tPassed.\n");

    return 0;
}


void test_upper_bound(void) {
    assert(get_upper_continue_rate(512, 512, 10) == 768);
    assert(get_upper_continue_rate(768, 512, 10) == 896);
    assert(get_upper_continue_rate(768, 256, 10) == 832);
    assert(get_upper_continue_rate(128, 128, 9) == 224);
}


void test_lower_bound(void) {
    assert(get_lower_continue_rate(512, 512, 10) == 256);
    assert(get_lower_continue_rate(768, 512, 10) == 384);
    assert(get_lower_continue_rate(768, 256, 10) == 576);
    assert(get_lower_continue_rate(128, 128, 9) == 96);
}

