#include "prand.h"

static uint8_t RAND_BUFFER[AES_BLOCK_SIZE];
static uint8_t IV[AES_BLOCK_SIZE] = { 0x9B, 0xCD, 0xAD, 0xE2, 0xDE, 0x47, 0x4A, 0xE2, 0x71, 0xF4, 0x20, 0x1C, 0x66, 0x39, 0x9a, 0xEF };


void generate_pseudo_rand(struct rand_state *state) {
    // Encrypt the current values
    AES256_encryptData(AES256_BASE, state->values, IV, RAND_BUFFER);

    // Copy the values back into the state buffer
    uint16_t i;
    for (i = 0; i < AES_BLOCK_SIZE; i++) {
        state->values[i] = RAND_BUFFER[i];
    }

    state->index = 0;
}


uint16_t pseudo_rand(struct rand_state *state) {
    const uint16_t i = state->index;
    uint16_t result = (((uint16_t) state->values[i+1]) << 8) | ((uint16_t) state->values[i]);
    state->index += 2;
    return result;
}


uint16_t rand_int(struct rand_state *state, uint16_t bits) {
    uint16_t rand = pseudo_rand(state);
    const uint16_t mask = ~(0xFFFF << bits);
    return (rand & mask);
}

void update_iv(uint8_t *iv) {
    uint16_t i;
    uint32_t value;

    for (i = 0; i < AES_BLOCK_SIZE; i++) {
        value = (((uint32_t) iv[i]) << 24) | (((uint32_t) iv[i+1]) << 16) | (((uint32_t) iv[i+2]) << 8) | ((uint32_t) iv[i+3]);
        value ^= (value << 13);
        value ^= (value >> 17);
        value ^= (value << 5);

        iv[i] = (uint8_t) (value & 0xFF);
        iv[i+1] = (uint8_t) ((value >> 8) & 0xFF);
        iv[i+2] = (uint8_t) ((value >> 16) & 0xFF);
        iv[i+3] = (uint8_t) ((value >> 24) & 0xFF);
    }
}
