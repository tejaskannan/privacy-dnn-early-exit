#include "prand_tests.h"

int main(void) {

    printf("Testing Cycle\n");
    test_cycle();
    printf("\tPassed.\n");

    return 0;
}


void test_cycle(void) {
    uint16_t start = 0xABCDu;
    uint16_t state = start;
    uint16_t period = 0;

    do {
        state = pseudo_rand(state);
        period += 1;
    } while (state != start);

    assert(period == 65535);
}
