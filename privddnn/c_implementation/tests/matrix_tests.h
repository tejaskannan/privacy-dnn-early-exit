#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "../matrix.h"

#ifndef MATRIX_TESTS_H_
#define MATRIX_TESTS_H_

void test_prod_3(void);
void test_prod_3_4(void);
void test_relu_3(void);
void test_relu_4(void);
void test_add_3(void);
void test_add_4(void);

uint8_t are_mats_equal(struct matrix *mat1, struct matrix *mat2);

#endif
