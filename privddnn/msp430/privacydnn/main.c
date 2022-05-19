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
#include "utils/prand.h"
#include "utils/message.h"
#include "utils/inference_result.h"
#include "utils/bt_functions.h"
#include "sensor/dht11.h"

#define TIMER_LIMIT 15

#define START_BYTE 0xAA
#define RESET_BYTE 0xBB
#define SEND_BYTE 0xCC
#define ACK_BYTE 0xDD

#define START_RESPONSE 0xAB
#define RESET_RESPONSE 0xCD

#define MESSAGE_OFFSET 32
#define MESSAGE_SIZE 32

// Encryption Parameters
static const uint8_t AES_KEY[AES_BLOCK_SIZE] = { 52,159,220,0,180,77,26,170,202,163,162,103,15,212,66,68 };
static uint8_t aesIV[AES_BLOCK_SIZE] = { 0x66, 0xa1, 0xfc, 0xdc, 0x34, 0x79, 0x66, 0xee, 0xe4, 0x26, 0xc1, 0x5a, 0x17, 0x9e, 0x78, 0x31 };
uint8_t randSeed[AES_BLOCK_SIZE] = { 0x31,0x29,0x7e,0x21,0xaa,0x34,0xc2,0xa8,0xd5,0xf5,0x3a,0xa3,0x55,0x73,0xe5,0xae };

volatile uint16_t timerIdx = 0;
volatile uint16_t sampleIdx = 0;

enum OpMode { IDLE = 0, START = 1, SAMPLE = 2, SEND = 3, ACK = 4, RESET = 5 };
volatile enum OpMode opMode = IDLE;

uint8_t messageBuffer[MESSAGE_SIZE + MESSAGE_OFFSET];

#ifdef IS_CGR_MAX_PROB
uint16_t targetExit[NUM_OUTPUTS];
uint16_t observedExit[NUM_OUTPUTS];
int16_t biases[NUM_OUTPUTS];
uint8_t prevPreds[NUM_OUTPUTS];
#endif

int16_t inputFeatures[NUM_FEATURES * VECTOR_COLS] = { 0 };
volatile uint8_t shouldExit = 0;
struct dht11_reading sensorResult;


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

	P1DIR |= BIT3;  // Set P1.3 to output for the switch on the BT module
	P6DIR &= ~BIT3;  // Set P6.3 to input for the temperature sensor

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    // Start with the bluetooth module on. This helps us coordinate the starting point
    P1OUT |= BIT3;

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
    struct rand_state randState = { randSeed, 0, AES_BLOCK_SIZE };
    struct exit_policy policy = { THRESHOLDS, &randState };

    struct cgr_state policyState;

    #ifdef IS_CGR_MAX_PROB
    for (i = 0; i < NUM_OUTPUTS; i++) {
        biases[i] = MAX_BIAS;
        prevPreds[i] = NUM_LABELS + 1;
        targetExit[i] = 0;
        observedExit[i] = 0;
    }

    policyState.windowSize = WINDOW_MIN;
    policyState.step = 0;
    policyState.targetExit = targetExit;
    policyState.observedExit = observedExit;
    policyState.biases = biases;
    policyState.maxBias = MAX_BIAS;
    policyState.increaseFactor = INCREASE_FACTOR;
    policyState.decreaseFactor = DECREASE_FACTOR;
    policyState.windowMin = WINDOW_MIN;
    policyState.windowMax = WINDOW_MAX;
    policyState.windowBits = WINDOW_BITS;
    policyState.prevPreds = prevPreds;
    policyState.trueExitRate = EXIT_RATES;
    #endif


    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);

    while (1) {

        if (opMode == START) {
            // Change mode to idle. The server will send an ACK message which will
            // put the device into sample mode
            opMode = IDLE;
            sampleIdx = 0;

            // Send the start response
            send_byte(START_RESPONSE);
        } else if (opMode == SAMPLE) {
            // Engage the sensor (dummy reading for power purposes only)
            dht11_read(&sensorResult);

            // Load the current input sample
            for (i = 0; i < NUM_FEATURES; i++) {
                inputFeatures[VECTOR_INDEX(i)] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
            }

            // Run the neural network inference
            branchynet_dnn(&inferenceResult, &inputs, PRECISION, &policy, &policyState);

            // Create the message and encrypt the result
            messageBuffer[MESSAGE_OFFSET] = inferenceResult.pred & 0xFF;
            messageBuffer[MESSAGE_OFFSET + 1] = inferenceResult.outputIdx & 0xFF;

            // Encrypt the result
            encrypt_aes128(messageBuffer + MESSAGE_OFFSET, aesIV, messageBuffer + AES_BLOCK_SIZE, AES_BLOCK_SIZE);

            // Write the IV into the first 16 bytes of the message. TODO: Do this via DMA
            for (i = 0; i < AES_BLOCK_SIZE; i++) {
                messageBuffer[i] = aesIV[i];
            }

            // Update the IV
            update_iv(aesIV);

            // Update the sample index
            sampleIdx += 1;

            // Set mode to idle (the server will pull the result) and turn on the bluetooth module
            opMode = IDLE;
            P1OUT ^= BIT3;
        } else if (opMode == SEND) {
            // Send the result to the server machine
            send_message(messageBuffer, MESSAGE_SIZE);

            if (sampleIdx == NUM_INPUTS) {
                sampleIdx = 0;
            }

            // Generate pseudo-random numbers for the next batch (if needed)
            #if defined(IS_RANDOM) || defined(IS_CGR_MAX_PROB)
            generate_pseudo_rand(&randState);
            #endif

            // Set the device to idle mode
            opMode = IDLE;
        } else if (opMode == RESET) {
            // Reset the experiment paramters
            sampleIdx = 0;
            timerIdx = 0;
            opMode = IDLE;

            // Send the response message
            send_byte(RESET_RESPONSE);
        } else if (opMode == ACK) {
            // Turn off the Bluetooth Module
            P1OUT ^= BIT3;

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
