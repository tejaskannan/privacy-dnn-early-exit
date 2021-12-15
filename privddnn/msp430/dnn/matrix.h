#include <stdint.h>
#include <msp430.h>
#include "utils/fixed_point.h"
#include "DSPLib.h"

#ifndef MATRIX_H_
#define MATRIX_H_

#define VECTOR_COLS 2
#define MATRIX_INDEX(R, C, NUM_COLS) ((R * NUM_COLS) + C)
#define VECTOR_INDEX(R) (R * VECTOR_COLS)

struct matrix {
    int16_t *data;
    uint8_t numRows;
    uint8_t numCols;
};

struct matrix *matrix_vector_prod(struct matrix *result, struct matrix *mat, struct matrix *vec, uint8_t precision);
struct matrix *vector_relu(struct matrix *result, struct matrix *vec);
struct matrix *vector_add(struct matrix *result, struct matrix *vec1, struct matrix *vec2);
uint8_t vector_argmax(struct matrix *vec);

#endif
