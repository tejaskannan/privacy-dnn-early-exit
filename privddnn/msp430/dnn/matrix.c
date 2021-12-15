#include "matrix.h"

DSPLIB_DATA(MULTIPLY_BUFFER, 4);
static FixedPoint MULTIPLY_BUFFER[1800];

FixedPoint *dma_load(FixedPoint *result, FixedPoint *data, uint16_t n) {
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

    uint16_t numRows = mat->numRows;
    uint16_t numCols = mat->numCols;

    // First transfer the input matrices to the LEA RAM segment using DMA
    uint16_t offset = 0;
    FixedPoint *vecData = dma_load(MULTIPLY_BUFFER, vec->data, numRows);
    offset += numRows * VECTOR_COLS;  // Ensure we have room for the vector columns

    FixedPoint *matData = dma_load(MULTIPLY_BUFFER + offset, mat->data, numRows * numCols);
    offset += numRows * numCols;

    FixedPoint *resultData = MULTIPLY_BUFFER + offset;  // Temporary buffer (in LEA RAM) for the result

    // When using the MSP430, we use the LEA for Matrix multiplications. Based on profiling,
    // the LEA can take up to 5x fewer compute cycles than a standard implementation.
    msp_status status;
    msp_matrix_mpy_q15_params mulParams;

    // Initialze LEA metadata
    mulParams.srcARows = numRows;
    mulParams.srcACols = numCols;
    mulParams.srcBRows = numCols;
    mulParams.srcBCols = VECTOR_COLS;

    // Perform Matrix multiplication using the LEA
    status = msp_matrix_mpy_q15(&mulParams, matData, vecData, resultData);
    msp_checkStatus(status);

    // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = numCols;
    shiftParams.cols = VECTOR_COLS;
    shiftParams.shift = 15 - precision;

    // Perform element-wise shift using the LEA
    if (shiftParams.shift > 0) {
        status = msp_matrix_shift_q15(&shiftParams, resultData, resultData);
        msp_checkStatus(status);
    }

    // Load result back into the given result vector
    dma_load(result->data, resultData, numCols * VECTOR_COLS);

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
