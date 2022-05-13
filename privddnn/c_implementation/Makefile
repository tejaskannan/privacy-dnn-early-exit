CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: dnn

dnn: utils/fixed_point.h utils/array.h utils/prand.h utils/inference_result.h utils/message.h policy.h matrix.h neural_network.h parameters.h data.h utils/fixed_point.c utils/array.c utils/prand.c utils/message.c policy.c matrix.c neural_network.c dnn_main.c
	$(CC) -o dnn utils/fixed_point.c utils/array.c utils/prand.c utils/message.c policy.c matrix.c neural_network.c dnn_main.c

fixedpoint: utils/fixed_point.h tests/fixed_point_tests.h utils/fixed_point.c tests/fixed_point_tests.c
	$(CC) -o fixedpoint utils/fixed_point.c tests/fixed_point_tests.c

array: utils/fixed_point.h utils/array.h tests/array_tests.h utils/fixed_point.c utils/array.c tests/array_tests.c
	$(CC) -o array utils/fixed_point.c utils/array.c tests/array_tests.c

matrix: utils/fixed_point.h matrix.h tests/matrix_tests.h utils/fixed_point.c matrix.c tests/matrix_tests.c
	$(CC) -o matrix utils/fixed_point.c matrix.c tests/matrix_tests.c

prand: utils/prand.h tests/prand_tests.h utils/prand.c tests/prand_tests.c
	$(CC) -o prand utils/prand.c tests/prand_tests.c

policy: utils/fixed_point.h utils/array.h utils/prand.h policy.h matrix.h tests/policy_tests.h parameters.h data.h utils/fixed_point.c utils/array.c utils/prand.c policy.c matrix.c tests/policy_tests.c
	$(CC) -o policy utils/fixed_point.c utils/array.c utils/prand.c utils/message.c policy.c matrix.c tests/policy_tests.c

clean:
	rm -f *.o ensemble dnn fixedpoint array prand matrix policy *.dSYM
