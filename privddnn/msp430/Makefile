CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: earlyexit

earlyexit: utils/fixed_point.h utils/array.h utils/lfsr.h policy.h decision_tree.h parameters.h data.h utils/fixed_point.c utils/array.c utils/lfsr.c policy.c decision_tree.c main.c
	$(CC) -o earlyexit utils/fixed_point.c utils/array.c utils/lfsr.c policy.c decision_tree.c main.c

fixedpoint: utils/fixed_point.h tests/fixed_point_tests.h utils/fixed_point.c tests/fixed_point_tests.c
	$(CC) -o fixedpoint utils/fixed_point.c tests/fixed_point_tests.c

array: utils/fixed_point.h utils/array.h tests/array_tests.h utils/fixed_point.c utils/array.c tests/array_tests.c
	$(CC) -o array utils/fixed_point.c utils/array.c tests/array_tests.c

lfsr: utils/lfsr.h tests/lfsr_tests.h utils/lfsr.c tests/lfsr_tests.c
	$(CC) -o lfsr utils/lfsr.c tests/lfsr_tests.c

clean: earlyexit
	rm -f *.o earlyexit fixedpoint array lfsr *.dSYM
