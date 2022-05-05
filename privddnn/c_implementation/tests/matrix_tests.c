#include "matrix_tests.h"


int main(void) {

    printf("Testing Matrix Vector Products.\n");
    test_prod_3();
    test_prod_3_4();
    printf("\tPassed.\n");

    printf("Testing Block Matrix Vector Products.\n");
    test_block_prod_4_6();
    test_block_prod_4_5();
    printf("\tPassed.\n");

    printf("Testing Vector Relu.\n");
    test_relu_3();
    test_relu_4();
    printf("\tPassed.\n");

    printf("Testing Vector Addition.\n");
    test_add_3();
    test_add_4();
    printf("\tPassed.\n");

    printf("Testing Vector Concat.\n");
    test_vector_concat_3();
    test_vector_concat_4_5();
    printf("\tPassed.\n");
}


void test_prod_3(void) {
    uint8_t precision = 10;
    int16_t one = 1 << precision;
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t eighth = 1 << (precision - 3);

    int16_t matData[] = { one, fourth, half, -1 * fourth, eighth, -1 * half, -1 * one, 0, one };
    struct matrix mat = { matData, 3, 3 };

    int16_t vecData[] = { fourth, half, -1 * half };
    struct matrix vec = { vecData, 3, 1 };

    int16_t resultData[3];
    struct matrix result = { resultData, 3, 1 };

    int16_t expectedData[] = { eighth, fourth, -1 * half - fourth };
    struct matrix expected = { expectedData, 3, 1 };

    matrix_vector_prod(&result, &mat, &vec, precision);
    assert(are_mats_equal(&result, &expected));
}


void test_block_prod_4_6(void) {
    uint8_t precision = 10;
    int16_t two = 1 << (precision + 1);
    int16_t one = 1 << precision;
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t eighth = 1 << (precision - 3);

    // Make the 6 matrix blocks
    int16_t blk1Data[] = { one, eighth, 0, -half };
    int16_t blk2Data[] = { -1 * eighth, half, -one, one + fourth };
    int16_t blk3Data[] = { fourth, one, -fourth, -one };
    int16_t blk4Data[] = { -one, half + fourth, 0, half };
    int16_t blk5Data[] = { half, -(half + fourth), 0, one };
    int16_t blk6Data[] = { one, fourth, -one, 0 };

    struct matrix blk1 = { blk1Data, 2, 2 };
    struct matrix blk2 = { blk2Data, 2, 2 };
    struct matrix blk3 = { blk3Data, 2, 2 };
    struct matrix blk4 = { blk4Data, 2, 2 };
    struct matrix blk5 = { blk5Data, 2, 2 };
    struct matrix blk6 = { blk6Data, 2, 2 };

    struct matrix *blocks[] = { &blk1, &blk2, &blk3, &blk4, &blk5, &blk6 };
    uint8_t rows[] = { 0, 0, 0, 2, 2, 2 };
    uint8_t cols[] = { 0, 2, 4, 0, 2, 4 };
    struct block_matrix mat = { blocks, 6, 4, 6, rows, cols };

    int16_t vecData[] = { one, two, fourth, -half, half + fourth, fourth };
    struct matrix vec = { vecData, 6, 1 };

    int16_t resultData[4];
    struct matrix result = { resultData, 4, 1 };

    int16_t expectedData[] = { 1440, -2368, 1856, -fourth };
    struct matrix expected = { expectedData, 4, 1 };

    block_matrix_vector_prod(&result, &mat, &vec, precision);
    assert(are_mats_equal(&result, &expected));
}


void test_block_prod_4_5(void) {
    uint8_t precision = 10;
    int16_t two = 1 << (precision + 1);
    int16_t one = 1 << precision;
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t eighth = 1 << (precision - 3);

    // Make the 6 matrix blocks
    int16_t blk1Data[] = { one, eighth, 0, -half };
    int16_t blk2Data[] = { -1 * eighth, half, -one, one + fourth };
    int16_t blk3Data[] = { fourth, one };
    int16_t blk4Data[] = { -one, half + fourth, 0, half };
    int16_t blk5Data[] = { half, -(half + fourth), 0, one };
    int16_t blk6Data[] = { one, fourth };

    struct matrix blk1 = { blk1Data, 2, 2 };
    struct matrix blk2 = { blk2Data, 2, 2 };
    struct matrix blk3 = { blk3Data, 2, 1 };
    struct matrix blk4 = { blk4Data, 2, 2 };
    struct matrix blk5 = { blk5Data, 2, 2 };
    struct matrix blk6 = { blk6Data, 2, 1 };

    struct matrix *blocks[] = { &blk1, &blk2, &blk3, &blk4, &blk5, &blk6 };
    uint8_t rows[] = { 0, 0, 0, 2, 2, 2 };
    uint8_t cols[] = { 0, 2, 4, 0, 2, 4 };
    struct block_matrix mat = { blocks, 6, 4, 5, rows, cols };

    int16_t vecData[] = { one, two, fourth, -half, half + fourth };
    struct matrix vec = { vecData, 5, 1 };

    int16_t resultData[4];
    struct matrix result = { resultData, 4, 1 };

    int16_t expectedData[] = { 1184, -1152, 1792, 704 };
    struct matrix expected = { expectedData, 4, 1 };

    block_matrix_vector_prod(&result, &mat, &vec, precision);
    assert(are_mats_equal(&result, &expected));
}



void test_prod_3_4(void) {
    uint8_t precision = 10;
    int16_t two = 1 << (precision + 1);
    int16_t one = 1 << precision;
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t eighth = 1 << (precision - 3);

    int16_t matData[] = { one, eighth, -1 * eighth, half, -1 * fourth, one, one + fourth, one + half, -1 * one - half, -1 * eighth, half + fourth, two };
    struct matrix mat = { matData, 3, 4 };

    int16_t vecData[] = { one, one, -1 * one, half };
    struct matrix vec = { vecData, 4, 1 };

    int16_t resultData[3];
    struct matrix result = { resultData, 3, 1 };

    int16_t expectedData[] = { one + half, fourth, -1 * one - fourth - eighth };
    struct matrix expected = { expectedData, 3, 1 };

    matrix_vector_prod(&result, &mat, &vec, precision);
    assert(are_mats_equal(&result, &expected));
}



void test_relu_3(void) {
    int16_t data[] = { 1024, -1024, -1 };
    struct matrix vec = { data, 3, 1 };

    int16_t expectedData[] = { 1024, 0, 0 };
    struct matrix expected = { expectedData, 3, 1 };

    vector_relu(&vec, &vec);
    assert(are_mats_equal(&vec, &expected));
}


void test_relu_4(void) {
    int16_t data[] = { 0, 1024, -2, 1 };
    struct matrix vec = { data, 4, 1 };

    int16_t expectedData[] = { 0, 1024, 0, 1 };
    struct matrix expected = { expectedData, 4, 1 };

    vector_relu(&vec, &vec);
    assert(are_mats_equal(&vec, &expected));
}


void test_add_3(void) {
    int16_t data1[] = { 1024, -1024, -1 };
    struct matrix vec1 = { data1, 3, 1 };

    int16_t data2[] = { -1024, 512, 2 };
    struct matrix vec2 = { data2, 3, 1 };

    int16_t expectedData[] = { 0, -512, 1 };
    struct matrix expected = { expectedData, 3, 1 };

    vector_add(&vec1, &vec1, &vec2);
    assert(are_mats_equal(&vec1, &expected));
}


void test_add_4(void) {
    int16_t data1[] = { 5, 1024, -1024, -1 };
    struct matrix vec1 = { data1, 4, 1 };

    int16_t data2[] = { -1024, 10, 0, 100 };
    struct matrix vec2 = { data2, 4, 1 };

    int16_t expectedData[] = { -1019, 1034, -1024, 99};
    struct matrix expected = { expectedData, 4, 1 };

    vector_add(&vec1, &vec1, &vec2);
    assert(are_mats_equal(&vec1, &expected));
}


void test_vector_concat_3(void) {
    int16_t data1[] = { 1024, -1024, -1 };
    struct matrix vec1 = { data1, 3, 1 };

    int16_t data2[] = { -1024, 512, 2 };
    struct matrix vec2 = { data2, 3, 1 };

    int16_t expectedData[] = { 1024, -1024, -1, -1024, 512, 2 };
    struct matrix expected = { expectedData, 6, 1 };

    int16_t resultData[6];
    struct matrix result = { resultData, 6, 1 };

    vector_concat(&result, &vec1, &vec2);
    assert(are_mats_equal(&result, &expected));
}


void test_vector_concat_4_5(void) {
    int16_t data1[] = { 7, 1024, -1024, -1 };
    struct matrix vec1 = { data1, 4, 1 };

    int16_t data2[] = { -1024, 12, 512, -9, 2 };
    struct matrix vec2 = { data2, 5, 1 };

    int16_t expectedData[] = { 7, 1024, -1024, -1, -1024, 12, 512, -9, 2 };
    struct matrix expected = { expectedData, 9, 1 };

    int16_t resultData[9];
    struct matrix result = { resultData, 9, 1 };

    vector_concat(&result, &vec1, &vec2);
    assert(are_mats_equal(&result, &expected));
}


uint8_t are_mats_equal(struct matrix *mat1, struct matrix *mat2) {
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols)) {
        return 0;
    }

    uint16_t i, j;
    for (i = 0; i < mat1->numRows; i++) {
        for (j = 0; j < mat1->numCols; j++) {
            if (mat1->data[MATRIX_INDEX(i, j, mat1->numCols)] != mat2->data[MATRIX_INDEX(i, j, mat1->numCols)]) {
                return 0;
            }
        }
    }

    return 1;
}


