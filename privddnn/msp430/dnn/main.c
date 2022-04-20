#include <msp430.h> 
#include <stdint.h>

#include "init.h"
#include "neural_network.h"
#include "parameters.h"
#include "data.h"
#include "policy.h"
#include "matrix.h"
#include "utils/aes256.h"
#include "utils/encryption.h"
#include "utils/lfsr.h"
#include "utils/message.h"
#include "utils/inference_result.h"
#include "utils/bt_functions.h"

#define TIMER_LIMIT 1

#define START_BYTE 0xAA
#define RESET_BYTE 0xBB
#define SEND_BYTE 0xCC
#define ACK_BYTE 0xDD

#define START_RESPONSE 0xAB
#define RESET_RESPONSE 0xCD

#define AES_BLOCK_SIZE 16

// Encryption Parameters
static const uint8_t AES_KEY[AES_BLOCK_SIZE] = { 52,159,220,0,180,77,26,170,202,163,162,103,15,212,66,68 };
static uint8_t aesIV[AES_BLOCK_SIZE] = { 0x66, 0xa1, 0xfc, 0xdc, 0x34, 0x79, 0x66, 0xee, 0xe4, 0x26, 0xc1, 0x5a, 0x17, 0x9e, 0x78, 0x31 };

volatile uint16_t timerIdx = 0;
volatile uint16_t sampleIdx = 0;

enum OpMode { IDLE = 0, START = 1, SAMPLE = 2, SEND = 3, ACK = 4, RESET = 5 };
volatile enum OpMode opMode = IDLE;

uint8_t messageBuffer[64] = { 0 };

uint16_t thresholds[NUM_LABELS];
uint16_t lfsrStates[NUM_OUTPUTS];

int16_t inputFeatures[NUM_FEATURES * VECTOR_COLS] = { 0 };
volatile uint8_t shouldExit = 0;


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

    // Start with the bluetooth module on. This helps us coordinate the starting point
    P6OUT = SET_BIT(P6OUT, BIT3)

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

    struct matrix inputs = { inputFeatures, NUM_FEATURES, VECTOR_COLS };
    struct inference_result inferenceResult;

    // Initialize the policy
    #ifdef IS_MAX_PROB
    thresholds[0] = THRESHOLD;
    #elif defined(IS_RANDOM)
    lfsrStates[0] = 26894;
    #elif defined(IS_LABEL_MAX_PROB)
    for (i = 0; i < NUM_LABELS; i++) {
        thresholds[i] = THRESHOLDS[i];
    }
    #endif

    struct exit_policy policy = { thresholds, lfsrStates };

    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);

    while (1) {

        if (opMode == START) {
            // Change the mode to acknowledge
            opMode = ACK;
            sampleIdx = 0;

            #ifdef IS_BUFFERED_MAX_PROB
            windowIdx = 0;
            #endif

            // Send the start response
            send_byte(START_RESPONSE);
        } else if (opMode == SAMPLE) {
            // Load the current input sample
            for (i = 0; i < NUM_FEATURES; i++) {
                inputFeatures[VECTOR_INDEX(i)] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
            }

            // Run the neural network inference
            branchynet_dnn(&inferenceResult, &inputs, PRECISION, &policy);

            // Create the message and encrypt the result
            messageBuffer[MESSAGE_OFFSET] = inferenceResult.pred & 0xFF;
            messageBuffer[MESSAGE_OFFSET + 1] = inferenceResult.outputIdx & 0xFF;

            // Encrypt the result
            encrypt_aes128(messageBuffer + MESSAGE_OFFSET, aesIV, messageBuffer + AES_BLOCK_SIZE, AES_BLOCK_SIZE);

            // Write the IV into the first 16 bytes of the message
            dma_load(messageBuffer, aesIV, AES_BLOCK_SIZE);

            // Update the IV
            lfsr_array(aesIV, AES_BLOCK_SIZE);
            
            // Update the sample index
            sampleIdx += 1;

            // Set mode to send (the server will pull the result) and turn on the bluetooth module
            opMode = SEND;
            P6OUT = SET_BIT(P6OUT, BIT3);
        } else if (opMode == SEND) {
            // Send the result to the server machine
            send_message(messageBuffer, messageSize);

            if (sampleIdx == NUM_INPUTS) {
                sampleIdx = 0;
            }

            opMode = ACK;
        } else if (opMode == RESET) {
            // Reset the experiment paramters
            sampleIdx = 0;
            timerIdx = 0;
            opMode = ACK;

            // Send the response message
            send_byte(RESET_RESPONSE);
        } else if (opMode == ACK) {
            // Turn off the Bluetooth Module
            P6OUT = CLEAR_BIT(P6OUT, BIT3);

            // Set to sample mode
            timerIdx = 0;
            opMode = SAMPLE;
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
            } else if (c == ACK_BYTE) {
                opMode = ACK;
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            }

            break;
        case USCI_UART_UCTXIFG: break;
        case USCI_UART_UCSTTIFG: break;
        case USCI_UART_UCTXCPTIFG: break;
        default: break;
    }
}
