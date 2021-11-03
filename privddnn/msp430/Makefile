CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: earlyexit

earlyexit: utils/fixed_point.h utils/array.h decision_tree.h parameters.h utils/fixed_point.c utils/array.c decision_tree.c main.c
	$(CC) -o earlyexit utils/fixed_point.c utils/array.c decision_tree.c main.c

fixedpoint: utils/fixed_point.h tests/fixed_point_tests.h utils/fixed_point.c tests/fixed_point_tests.c
	$(CC) -o fixedpoint utils/fixed_point.c tests/fixed_point_tests.c

clean: earlyexit
	rm -f *.o earlyexit fixedpoint *.dSYM
