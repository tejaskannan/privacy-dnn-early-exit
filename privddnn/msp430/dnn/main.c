#include <msp430.h> 
#include <stdint.h>

#include "init.h"
#include "decision_tree.h"
#include "parameters.h"
#include "data.h"
#include "policy.h"
#include "utils/aes256.h"
#include "utils/encryption.h"
#include "utils/lfsr.h"
#include "utils/inference_result.h"
#include "utils/message.h"
#include "utils/bt_functions.h"

#define TIMER_LIMIT 5

#define START_BYTE 0xAA
#define RESET_BYTE 0xBB
#define SEND_BYTE 0xCC

#define START_RESPONSE 0xAB
#define RESET_RESPONSE 0xCD

// Encryption Parameters
#define AES_BLOCK_SIZE 16
static const uint8_t AES_KEY[16] = { 52,159,220,0,180,77,26,170,202,163,162,103,15,212,66,68 };
static uint8_t aesIV[16] = { 0x66, 0xa1, 0xfc, 0xdc, 0x34, 0x79, 0x66, 0xee, 0xe4, 0x26, 0xc1, 0x5a, 0x17, 0x9e, 0x78, 0x31 };

volatile uint16_t timerIdx = 0;
volatile uint16_t sampleIdx = 0;
volatile uint8_t pred = 0;
volatile uint8_t shouldExit = 0;
int16_t inputFeatures[NUM_FEATURES * VECTOR_COLS];
int16_t hiddenData[W0.numRows * VECTOR_COLS];

enum OpMode { IDLE = 0, START = 1, SAMPLE = 2, SEND = 3, RESET = 4 };
volatile OpMode opMode = NOT_SARTED;

#pragma PERSISTENT(MESSAGE_BUFFER)
int8_t MESSAGE_BUFFER[512];

#ifndef IS_MAX_PROB
uint16_t lfsrState = 2489;
#endif

#if defined(IS_MAX_PROB) || defined(IS_RANDOM)
int32_t logits[NUM_LABELS];
int32_t probs[NUM_LABELS];
struct inference_result inferenceResult = { logits, probs, 0 };
#elif defined(IS_BUFFERED_MAX_PROB)
#pragma PERSISTENT(windowInputFeatures)
int16_t windowInputFeatures[NUM_INPUT_FEATURES * WINDOW_SIZE * VECTOR_COLS] = { 0 };
uint16_t windowIdx = 0;

uint8_t exitResults[WINDOW_SIZE];
#pragma PERSISTENT(earlyLogits);
int32_t logits[NUM_LABELS * WINDOW_SIZE] = { 0 };
#pragma PERSISTENT(earlyProbs);
int32_t probs[NUM_LABELS * WINDOW_SIZE] = { 0 };

struct inference_result inferenceResults[WINDOW_SIZE];
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
        inferenceResult[i].logits = logits + (i * NUM_LABELS);
        inferenceResult[i].probs = probs + (i * NUM_LABELS);
    }
    #endif

    struct matrix inputs = { inputFeatures, NUM_FEATURES, VECTOR_COLS };
    struct matrix hiddenResult = { hiddenData, W0.numRows, VECTOR_COLS };
    volatile uint16_t messageSize;

    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);

    while (1) {

        if (opMode == START) {
            // Change the mode to sample
            opMode = SAMPLE;
            sampleIdx = 0;

            // Send the start response
            send_byte(START_RESPONSE);
        } else if (opMode == SAMPLE) {
            // Load the current input sample
            for (i = 0; i < NUM_FEATURES; i++) {
                inputFeatures[VECTOR_INDEX(i)] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
            }

            // Run the neural network inference
            neural_network(&inferenceResult, &hiddenResult, &inputs, PRECISION);

            // Run the exit policy
            #ifdef IS_MAX_PROB
            shouldExit = max_prob_should_exit(&inferenceResult, THRESHOLD);
            #elif IS_RANDOM
            shouldExit = random_should_exit(EXIT_RATE, lfsrState);
            lfsrState = lfsr_step(lfsrState);
            #endif

            if (shouldExit) {
                messageSize = create_exit_message(MESSAGE_BUFFER + AES_BLOCK_SIZE + 2, &inferenceResult);
            } else {
                messageSize = create_elevate_message(MESSAGE_BUFFER + AES_BLOCK_SIZE + 2, &inferenceResult, &inputs);
            }

            // Include the original message length
            MESSAGE_BUFFER[AES_BLOCK_SIZE] = (messageSize >> 8) & 0xFF;
            MESSAGE_BUFFER[AES_BLOCK_SIZE + 1] = messageSize & 0xFF;

            // Encrypt the result
            messageSize = round_to_aes_block(messageSize);
            encrypt_aes_128(MESSAGE_BUFFER + AES_BLOCK_SIZE, aesIV, MESSAGE_BUFFER, messageSize);

            // Update the IV
            lfsr_array(aesIV, AES_BLOCK_SIZE);

            // Update the sample index
            sampleIdx += 1;
            
            // Update the phase to idle. The server will pull the result.
            opMode = IDLE;
        } else if (opMode == SEND) {
            // Send the result to the server machine
            send_message(MESSAGE_BUFFER, messageSize);

            // Update the phase to return to sampling or
            // end the experiment
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

            if (timerIdx == TIMER_LIMIT) {
                timerIdx = 0;

                if (mode != NOT_STARTED) {
                    __bic_SR_register_on_exit(LPM3_bits | GIE);
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
