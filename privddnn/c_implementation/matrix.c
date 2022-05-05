#include "matrix.h"


static int16_t RESULT_BUFFER[400];


struct matrix *matrix_vector_prod(struct matrix *result, struct matrix *mat, struct matrix *vec, const uint8_t precision) {
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


uint16_t min(uint16_t x, uint16_t y) {
    return (x > y) ? y : x;
}


struct matrix *block_matrix_vector_prod(struct matrix *result, struct block_matrix *mat, struct matrix *vec, const uint8_t precision) {
    // Check the dimensions
    if ((result->numRows != mat->numRows) || (mat->numCols != vec->numRows) || (vec->numCols != VECTOR_COLS) || (vec->numCols != result->numCols)) {
        return result;
    }

    uint16_t r, rowBlock, colBlock, blockIdx;
    int16_t sum, prod;

    const uint16_t n = mat->numRows;
    const uint16_t m = mat->numCols;
    
    uint16_t destIdx, matIdx, vecIdx;
    int16_t matElement, vecElement;

    // Zero out the result vector
    for (r = 0; r < n; r++) {
        result->data[VECTOR_INDEX(r)] = 0;
    }

    struct matrix tempVec;
    tempVec.numCols = vec->numCols;

    struct matrix tempResult;
    tempResult.data = RESULT_BUFFER;
    tempResult.numCols = vec->numCols;

    for (blockIdx = 0; blockIdx < mat->numBlocks; blockIdx++) {
        rowBlock = mat->rows[blockIdx];
        colBlock = mat->cols[blockIdx];

        tempResult.numRows = mat->blocks[blockIdx]->numRows;

        tempVec.numRows = mat->blocks[blockIdx]->numCols;
        tempVec.data = vec->data + VECTOR_INDEX(colBlock);

        matrix_vector_prod(&tempResult, mat->blocks[blockIdx], &tempVec, precision);

        for (r = rowBlock; r < rowBlock + mat->blocks[blockIdx]->numRows; r++) {
            result->data[VECTOR_INDEX(r)] = fp16_add(result->data[VECTOR_INDEX(r)], tempResult.data[VECTOR_INDEX(r - rowBlock)]);
        }
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


struct matrix *vector_concat(struct matrix *result, struct matrix *vec1, struct matrix *vec2) {
    if ((result->numCols != VECTOR_COLS) || (vec1->numCols != VECTOR_COLS) || (vec2->numCols != VECTOR_COLS) || (result->numRows != (vec1->numRows + vec2->numRows))) {
        return result;
    }

    uint16_t i = 0;
    for (; i < vec1->numRows; i++) {
        result->data[VECTOR_INDEX(i)] = vec1->data[VECTOR_INDEX(i)];
    }

    uint16_t offset = i;
    for (i = 0; i < vec2->numRows; i++) {
        result->data[VECTOR_INDEX(i + offset)] = vec2->data[VECTOR_INDEX(i)];
    }

    return result;
}


int32_t vector_exp_sum(struct matrix *vec, const int16_t max, const uint8_t precision) {
    if (vec->numCols != VECTOR_COLS) {
        return 0;
    }

    volatile int32_t sum = 0;
    volatile int16_t vecElement;
    volatile int32_t element;

    uint8_t i;
    for (i = 0; i < vec->numRows; i++) {
        vecElement = vec->data[VECTOR_INDEX(i)];

        if (vecElement >= (INT16_MIN + max)) {
	        element = (int32_t) fp16_sub(vecElement, max);
            sum = fp32_add(sum, fp32_exp(element, precision));
        }
    }

    return sum;
}


int32_t *vector_softmax(int32_t *result, struct matrix *vec, const uint8_t precision) {
    const uint8_t maxIdx = vector_argmax(vec);
    const int16_t max = vec->data[VECTOR_INDEX(maxIdx)];
    const int32_t sum = vector_exp_sum(vec, max, precision);

    volatile int32_t element;
    volatile int16_t vecElement;
    volatile int32_t cumSum = 0;

    const uint16_t n = vec->numRows - 1;
    uint8_t i;
    for (i = 0; i < n; i++) {
        vecElement = vec->data[VECTOR_INDEX(i)];

        if (vecElement >= (INT16_MIN + max)) {
            element = (int32_t) fp16_sub(vec->data[VECTOR_INDEX(i)], max);
            result[i] = fp32_div(fp32_exp(element, precision), sum, precision);
	        cumSum = fp32_add(result[i], cumSum);
        } else {
            result[i] = 0;
        }
    }

    result[n] = fp32_sub((1 << precision), cumSum);

    return result;
}
