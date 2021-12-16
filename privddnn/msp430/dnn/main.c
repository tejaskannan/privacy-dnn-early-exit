#include <msp430.h> 
#include <stdint.h>

#include "init.h"
#include "neural_network.h"
#include "parameters.h"
#include "data.h"
#include "policy.h"
#include "utils/aes256.h"
#include "utils/encryption.h"
#include "utils/lfsr.h"
#include "utils/message.h"
#include "utils/inference_result.h"
#include "utils/bt_functions.h"

#define TIMER_LIMIT 5
#define HIDDEN_SIZE 24

#define START_BYTE 0xAA
#define RESET_BYTE 0xBB
#define SEND_BYTE 0xCC

#define START_RESPONSE 0xAB
#define RESET_RESPONSE 0xCD

#define MESSAGE_BUFFER_SIZE 512
#define AES_BLOCK_SIZE 16
#define MESSAGE_OFFSET 32
#define LENGTH_SIZE 2

// Encryption Parameters
static const uint8_t AES_KEY[AES_BLOCK_SIZE] = { 52,159,220,0,180,77,26,170,202,163,162,103,15,212,66,68 };
static uint8_t aesIV[AES_BLOCK_SIZE] = { 0x66, 0xa1, 0xfc, 0xdc, 0x34, 0x79, 0x66, 0xee, 0xe4, 0x26, 0xc1, 0x5a, 0x17, 0x9e, 0x78, 0x31 };

volatile uint16_t timerIdx = 0;
volatile uint16_t sampleIdx = 0;
volatile uint8_t pred = 0;

enum OpMode { IDLE = 0, START = 1, SAMPLE = 2, SEND = 3, RESET = 4 };
volatile enum OpMode opMode = IDLE;

#pragma PERSISTENT(messageBuffer)
uint8_t messageBuffer[512] = { 0 };

#ifndef IS_MAX_PROB
uint16_t lfsrState = 2489;
#endif

#if defined(IS_MAX_PROB) || defined(IS_RANDOM)
int16_t inputFeatures[NUM_FEATURES * VECTOR_COLS] = { 0 };
int16_t hiddenData[HIDDEN_SIZE * VECTOR_COLS] = { 0 };

int32_t logits[NUM_LABELS];
int32_t probs[NUM_LABELS];
struct inference_result inferenceResult = { logits, probs, 0 };
volatile uint8_t shouldExit = 0;

#elif defined(IS_BUFFERED_MAX_PROB)
#pragma PERSISTENT(inputFeatures)
int16_t inputFeatures[NUM_INPUT_FEATURES * WINDOW_SIZE * VECTOR_COLS] = { 0 };

#pragma PERSISTENT(hiddenFeatures)
int16_t hiddenFeatures[HIDDEN_SIZE * WINDOW_SIZE * VECTOR_COLS] = { 0 };

#pragma PERSISTENT(logits);
int32_t logits[NUM_LABELS * WINDOW_SIZE] = { 0 };

#pragma PERSISTENT(probs);
int32_t probs[NUM_LABELS * WINDOW_SIZE] = { 0 };

struct matrix inputs[WINDOW_SIZE];
struct matrix hidden[WINDOW_SIZE];
struct inference_result inferenceResults[WINDOW_SIZE];
uint8_t shouldExit[WINDOW_SIZE];

volatile uint16_t windowIdx = 0;
#endif


/**
 * main.c
 */
int main(void)
{
	WDTCTL = WDTPW | WDTHOLD;	// stop watchdog timer
	
	init_gpio();
	init_uart_pins();
	init_uart_system();
	init_timer();

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    // Set the Encryption Key
    uint8_t status = AES256_setCipherKey(AES256_BASE, AES_KEY, AES256_KEYLENGTH_128BIT);
    if (status == STATUS_FAIL) {
        P1OUT |= BIT0;
        return 1;
    }

    sampleIdx = 0;
    timerIdx = 0;
    uint16_t i;

    #ifdef IS_BUFFERED_MAX_PROB
    for (i = 0; i < WINDOW_SIZE; i++) {
        inputs[i].data = inputFeatures + (i * NUM_FEATURES * VECTOR_COLS);
        inputs[i].numRows = NUM_FEATURES;
        inputs[i].numCols = VECTOR_COLS;

        hidden[i].data = hiddenFeatures + (i * HIDDEN_SIZE * VECTOR_COLS);
        hidden[i].numRows = HIDDEN_SIZE;
        hidden[i].numCols = VECTOR_COLS;

        inferenceResults[i].logits = logits + (i * NUM_LABELS);
        inferenceResults[i].probs = probs + (i * NUM_LABELS);
        inferenceResults[i].pred = 0;
    }
    #else
    struct matrix inputs = { inputFeatures, NUM_FEATURES, VECTOR_COLS };
    struct matrix hidden = { hiddenData, HIDDEN_SIZE, VECTOR_COLS };
    #endif

    volatile uint16_t messageSize;

    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);

    while (1) {

        if (opMode == START) {
            // Change the mode to sample
            opMode = SAMPLE;
            sampleIdx = 0;

            #ifdef IS_BUFFERED_MAX_PROB
            windowIdx = 0;
            #endif

            // Send the start response
            send_byte(START_RESPONSE);
        } else if (opMode == SAMPLE) {
#ifdef IS_BUFFERED_MAX_PROB
            // Load the current input sample
            for (i = 0; i < NUM_FEATURES; i++) {
                (inputs + windowIdx)->data[VECTOR_INDEX(i)] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
            }

            // Run the neural network inference on this sample
            neural_network(inferenceResults + windowIdx, hidden + windowIdx, inputs + windowIdx, PRECISION);

            windowIdx += 1;
            sampleIdx += 1;

            if ((windowIdx == WINDOW_SIZE) || (sampleIdx == NUM_INPUTS)) {
                // Determine the exiting decisions
                buffered_max_prob_should_exit(shouldExit, inferenceResults, lfsrState, ELEVATE_COUNT, ELEVATE_REMAINDER, windowIdx);

                // Encode the buffered message
                messageSize = create_buffered_message(messageBuffer + MESSAGE_OFFSET + LENGTH_SIZE, inferenceResults, inputs, hidden, shouldExit, windowIdx, MESSAGE_BUFFER_SIZE - MESSAGE_OFFSET - LENGTH_SIZE);

                // Include the original message length
                messageBuffer[MESSAGE_OFFSET] = (messageSize >> 8) & 0xFF;
                messageBuffer[MESSAGE_OFFSET + 1] = messageSize & 0xFF;

                // Encrypt the result
                messageSize = round_to_aes_block(messageSize);
                encrypt_aes128(messageBuffer + MESSAGE_OFFSET, aesIV, messageBuffer + AES_BLOCK_SIZE, messageSize);

                // Write the IV into the first 16 bytes of the message
                for (i = 0; i < AES_BLOCK_SIZE; i++) {
                    messageBuffer[i] = aesIV[i];
                }

                // Account for the initialization vector
                messageSize += AES_BLOCK_SIZE;

                // Update the IV
                lfsr_array(aesIV, AES_BLOCK_SIZE);

                // Update the policy's random state
                lfsrState = lfsr_step(lfsrState);

                // Reset the window and wait until the sending phase
                windowIdx = 0;
                opMode = IDLE;
            } else {
                opMode = SAMPLE;  // Collect the next sample (do not send the results until the end of the window)
            }
#else
            // Load the current input sample
            for (i = 0; i < NUM_FEATURES; i++) {
                inputFeatures[VECTOR_INDEX(i)] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
            }

            // Run the neural network inference
            neural_network(&inferenceResult, &hidden, &inputs, PRECISION);

            // Run the exit policy
            #ifdef IS_MAX_PROB
            shouldExit = max_prob_should_exit(&inferenceResult, THRESHOLD);
            #elif IS_RANDOM
            shouldExit = random_should_exit(EXIT_RATE, lfsrState);
            lfsrState = lfsr_step(lfsrState);
            #endif

            if (shouldExit) {
                messageSize = create_exit_message(messageBuffer + MESSAGE_OFFSET + LENGTH_SIZE, &inferenceResult);
            } else {
                messageSize = create_elevate_message(messageBuffer + MESSAGE_OFFSET + LENGTH_SIZE, &hidden, &inputs, MESSAGE_BUFFER_SIZE - MESSAGE_OFFSET - LENGTH_SIZE);
            }
            // Include the original message length
            messageBuffer[MESSAGE_OFFSET] = (messageSize >> 8) & 0xFF;
            messageBuffer[MESSAGE_OFFSET + 1] = messageSize & 0xFF;

            // Encrypt the result
            messageSize = round_to_aes_block(messageSize);
            encrypt_aes128(messageBuffer + MESSAGE_OFFSET, aesIV, messageBuffer + AES_BLOCK_SIZE, messageSize);

            // Write the IV into the first 16 bytes of the message
            for (i = 0; i < AES_BLOCK_SIZE; i++) {
                messageBuffer[i] = aesIV[i];
            }

            // Account for the initialization vector
            messageSize += AES_BLOCK_SIZE;

            // Update the IV
            lfsr_array(aesIV, AES_BLOCK_SIZE);
            
            // Update the phase to idle. The server will pull the result.
            sampleIdx += 1;
            opMode = IDLE;
#endif
        } else if (opMode == SEND) {
            // Send the result to the server machine
            send_message(messageBuffer, messageSize);

            if (sampleIdx == NUM_INPUTS) {
                sampleIdx = 0;
                opMode = IDLE;
            } else {
                opMode = SAMPLE;
            }
        } else if (opMode == RESET) {
            // Reset the experiment paramters
            sampleIdx = 0;
            timerIdx = 0;
            opMode = IDLE;

            #ifdef IS_BUFFERED_MAX_PROB
            windowIdx = 0;
            #endif

            // Send the response message
            send_byte(RESET_RESPONSE);
        }

        // Place the device back into LPM
        __bis_SR_register(LPM3_bits | GIE);
    }

	return 0;
}


/**
 * ISR for Timer A overflow
 */
#pragma vector = TIMER0_A1_VECTOR
__interrupt void Timer0_A1_ISR (void) {
    /**
     * Timer Interrupts to perform operations.
     */
    switch(__even_in_range(TA0IV, TAIV__TAIFG))
    {
        case TAIV__NONE:   break;           // No interrupt
        case TAIV__TACCR1: break;           // CCR1 not used
        case TAIV__TACCR2: break;           // CCR2 not used
        case TAIV__TACCR3: break;           // reserved
        case TAIV__TACCR4: break;           // reserved
        case TAIV__TACCR5: break;           // reserved
        case TAIV__TACCR6: break;           // reserved
        case TAIV__TAIFG:                   // overflow
            timerIdx += 1;

            if (timerIdx >= TIMER_LIMIT) {
                timerIdx = 0;

                if (opMode == SAMPLE) {
                    __bic_SR_register_on_exit(LPM3_bits |GIE);
                }
            }

            break;
        default: break;
    }
}


/**
 * UART communication
 */
#pragma vector=EUSCI_A3_VECTOR
__interrupt void USCI_A3_ISR(void) {
    char c;

    switch(__even_in_range(UCA3IV, USCI_UART_UCTXCPTIFG)) {
        case USCI_NONE: break;
        case USCI_UART_UCRXIFG:
            // Wait until TX Buffer is not busy
            while(!(UCA3IFG & UCTXIFG));

            c = (char) UCA3RXBUF;

            if (c == START_BYTE) {
                opMode = START;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            } else if (c == RESET_BYTE) {
                opMode = RESET;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            } else if (c == SEND_BYTE) {
                opMode = SEND;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            }

            break;
        case USCI_UART_UCTXIFG: break;
        case USCI_UART_UCSTTIFG: break;
        case USCI_UART_UCTXCPTIFG: break;
        default: break;
    }
}
