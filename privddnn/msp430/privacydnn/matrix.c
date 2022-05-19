#include "matrix.h"

DSPLIB_DATA(MULTIPLY_BUFFER, 4);
static int16_t MULTIPLY_BUFFER[1400];

DSPLIB_DATA(RESULT_BUFFER, 4);
static int16_t RESULT_BUFFER[400];


int16_t *dma_load(int16_t *result, int16_t *data, const uint16_t n) {
    /**
     * Loads the first n elements of the data array into the result array using
     * DMA.
     */
    // Configure DMA channel 0
    __data20_write_long((uintptr_t) &DMA0SA, (uintptr_t) data);   // Source block address
    __data20_write_long((uintptr_t) &DMA0DA, (uintptr_t) result); // Destination single address
    DMA0SZ = n;                                      // Block size
    DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3; // Rpt, inc
    DMA0CTL |= DMAEN;                                // Enable DMA0
    DMA0CTL |= DMAREQ;

    return result;
}


struct matrix *matrix_vector_prod(struct matrix *result, struct matrix *mat, struct matrix *vec, uint8_t precision) {
    // Check the dimensions
    if ((result->numRows != mat->numRows) || (mat->numCols != vec->numRows) || (vec->numCols != VECTOR_COLS) || (vec->numCols != result->numCols)) {
        return result;
    }

    const uint16_t numMatElems = ((uint16_t) mat->numRows) * ((uint16_t) mat->numCols);
    const uint16_t numVecElems = ((uint16_t) vec->numRows) * ((uint16_t) vec->numCols);
    const uint16_t numResultElems = ((uint16_t) result->numRows) * ((uint16_t) result->numCols);

    // First transfer the input matrices to the LEA RAM segment using DMA
    volatile uint16_t offset = 0;
    int16_t *vecData = dma_load(MULTIPLY_BUFFER, vec->data, numVecElems);
    offset += numVecElems;  // Ensure we have room for the vector columns

    int16_t *matData = dma_load(MULTIPLY_BUFFER + offset, mat->data, numMatElems);
    offset += numMatElems;

    int16_t *resultData = MULTIPLY_BUFFER + offset;  // Temporary buffer (in LEA RAM) for the result

    // When using the MSP430, we use the LEA for Matrix multiplications. Based on profiling,
    // the LEA can take up to 5x fewer compute cycles than a standard implementation.
    msp_status status;
    msp_matrix_mpy_q15_params mulParams;

    // Initialze LEA metadata
    mulParams.srcARows = mat->numRows;
    mulParams.srcACols = mat->numCols;
    mulParams.srcBRows = vec->numRows;
    mulParams.srcBCols = vec->numCols;

    // Perform Matrix multiplication using the LEA
    status = msp_matrix_mpy_q15(&mulParams, matData, vecData, resultData);
    msp_checkStatus(status);

    // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = result->numRows;
    shiftParams.cols = result->numCols;
    shiftParams.shift = 15 - precision;

    // Perform element-wise shift using the LEA
    if (shiftParams.shift > 0) {
        status = msp_matrix_shift_q15(&shiftParams, resultData, resultData);
        msp_checkStatus(status);
    }

    // Load result back into the given result vector
    dma_load(result->data, resultData, numResultElems);

    return result;
}


struct matrix *block_matrix_vector_prod(struct matrix *result, struct block_matrix *mat, struct matrix *vec, const uint8_t precision) {
    // Check the dimensions
    if ((result->numRows != mat->numRows) || (mat->numCols != vec->numRows) || (vec->numCols != VECTOR_COLS) || (vec->numCols != result->numCols)) {
        return result;
    }

    volatile uint16_t r;
    volatile uint16_t rowBlock;
    volatile uint16_t colBlock;
    volatile uint16_t blockIdx;

    const uint16_t n = mat->numRows;
    const uint16_t m = mat->numCols;

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

    const uint16_t vec1Elems = vec1->numRows * vec1->numCols;
    const uint16_t vec2Elems = vec2->numRows * vec2->numCols;

    dma_load(result->data, vec1->data, vec1Elems);
    dma_load(result->data + vec1Elems, vec2->data, vec2Elems);

    return result;
}


int32_t vector_exp_sum(struct matrix *vec, const int16_t max, const uint8_t precision) {
    if (vec->numCols != VECTOR_COLS) {
        return 0;
    }

    volatile int32_t sum = 0;
    volatile int32_t element;

    uint8_t i;
    for (i = 0; i < vec->numRows; i++) {
	    element = (int32_t) fp16_sub(vec->data[VECTOR_INDEX(i)], max);
        sum = fp32_add(sum, fp32_exp(element, precision));
    }

    return sum;
}


int32_t *vector_softmax(int32_t *result, struct matrix *vec, const uint8_t precision) {
    const uint8_t maxIdx = vector_argmax(vec);
    const int16_t max = vec->data[VECTOR_INDEX(maxIdx)];
    const int32_t sum = vector_exp_sum(vec, max, precision);

    volatile int32_t element;
    volatile int32_t cumSum = 0;

    const uint16_t n = vec->numRows - 1;
    uint8_t i;
    for (i = 0; i < n; i++) {
	    element = (int32_t) fp16_sub(vec->data[VECTOR_INDEX(i)], max);
        result[i] = fp32_div(fp32_exp(element, precision), sum, precision);
	    cumSum = fp32_add(result[i], cumSum);
    }

    result[n] = fp32_sub((1 << precision), cumSum);

    return result;
}
