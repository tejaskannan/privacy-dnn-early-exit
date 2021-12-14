#include "matrix.h"


struct matrix *matrix_vector_prod(struct matrix *result, struct matrix *mat, struct matrix *vec, uint8_t precision) {
    // Check the dimensions
    if ((result->numRows != mat->numRows) || (mat->numCols != vec->numRows) || (vec->numCols != VECTOR_COLS) || (vec->numCols != result->numCols)) {
        return result;
    }

    uint16_t r, c;
    int16_t sum, prod;

    uint16_t n = mat->numRows;
    uint16_t m = mat->numCols;

    for (r = 0; r < n; r++) {
        sum = 0;

        for (c = 0; c < m; c++) {
            prod = fp16_mul(mat->data[MATRIX_INDEX(r, c, m)], vec->data[VECTOR_INDEX(c)], precision);
            sum = fp16_add(sum, prod);
        }

        result->data[VECTOR_INDEX(r)] = sum;
    }

    return result;
}


struct matrix *vector_relu(struct matrix *result, struct matrix *vec) {
    if ((result->numRows != vec->numRows) || (result->numCols != vec->numCols) || (vec->numCols != VECTOR_COLS)) {
        return result;
    }

    uint16_t n = vec->numRows;
    uint16_t r;

    for (r = 0; r < n; r++) {
        result->data[VECTOR_INDEX(r)] = fp16_max(vec->data[VECTOR_INDEX(r)], 0);
    }

    return result;
}


struct matrix *vector_add(struct matrix *result, struct matrix *vec1, struct matrix *vec2) {
    if ((vec1->numCols != VECTOR_COLS) || (vec2->numCols != VECTOR_COLS) || (result->numCols != VECTOR_COLS) || (vec1->numRows != vec2->numRows) || (vec1->numRows != result->numRows)) {
        return result;
    }

    uint16_t n = result->numRows;
    uint16_t r;

    for (r = 0; r < n; r++) {
        result->data[VECTOR_INDEX(r)] = fp16_add(vec1->data[VECTOR_INDEX(r)], vec2->data[VECTOR_INDEX(r)]);
    }

    return result;
}


uint8_t vector_argmax(struct matrix *vec) {
    if (vec->numCols != VECTOR_COLS) {
        return 0;
    }

    int16_t maxValue = vec->data[0];
    uint8_t maxIdx = 0;

    int16_t v;
    uint8_t i;
    for (i = 1; i < vec->numRows; i++) {
        v = vec->data[VECTOR_INDEX(i)];

        if (v > maxValue) {
            maxValue = v;
            maxIdx = i;
        }
    }

    return maxIdx;
}
