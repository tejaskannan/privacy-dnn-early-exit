#include <msp430.h> 
#include <stdint.h>

#include "init.h"
#include "decision_tree.h"
#include "parameters.h"
#include "data.h"
#include "policy.h"
#include "utils/lfsr.h"

#define TIMER_LIMIT 5

volatile uint16_t timerIdx = 0;
volatile uint16_t sampleIdx = 0;
volatile uint8_t pred = 0;
volatile uint8_t shouldExit = 0;
int16_t inputFeatures[NUM_FEATURES];

#ifndef IS_MAX_PROB
uint16_t lfsrState = 2489;
#endif

#if defined(IS_MAX_PROB) || defined(IS_RANDOM)
int32_t earlyLogits[NUM_LABELS];
int32_t earlyProbs[NUM_LABELS];
int32_t fullLogits[NUM_LABELS];
struct inference_result earlyResult = { earlyLogits, earlyProbs, 0 };
struct inference_result fullResult = { fullLogits, fullLogits, 0 };
#elif defined(IS_BUFFERED_MAX_PROB)
#pragma PERSISTENT(windowInputFeatures)
int16_t windowInputFeatures[NUM_INPUT_FEATURES * WINDOW_SIZE] = { 0 };
uint16_t windowIdx = 0;

uint8_t exitResults[WINDOW_SIZE];
#pragma PERSISTENT(earlyLogits);
int32_t earlyLogits[NUM_LABELS * WINDOW_SIZE] = { 0 };
#pragma PERSISTENT(earlyProbs);
int32_t earlyProbs[NUM_LABELS * WINDOW_SIZE] = { 0 };
int32_t fullLogits[NUM_LABELS];

struct inference_result earlyResult[WINDOW_SIZE];
struct inference_result fullResult = { fullLogits, fullLogits, 0 };
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

    sampleIdx = 0;
    timerIdx = 0;
    uint16_t i;

    #ifdef IS_BUFFERED_MAX_PROB
    for (i = 0; i < WINDOW_SIZE; i++) {
        earlyResult[i].logits = earlyLogits + (i * NUM_LABELS);
        earlyResult[i].probs = earlyProbs + (i * NUM_LABELS);
    }
    #endif

    // Put into Low Power Mode
    __bis_SR_register(LPM3_bits | GIE);


    while (1) {

        // Get the current input sample
        for (i = 0; i < NUM_FEATURES; i++) {
            inputFeatures[i] = DATASET_INPUTS[sampleIdx * NUM_FEATURES + i];
        }

#ifdef IS_BUFFERED_MAX_PROB
        adaboost_inference_early(earlyResult + windowIdx, inputFeatures, &ENSEMBLE, PRECISION);

        // Copy the input features for possible later elevation
        for (i = 0; i < NUM_FEATURES; i++) {
            windowInputFeatures[windowIdx * NUM_INPUT_FEATURES + i] = inputFeatures[i];
        }

        // Increment the window index for the next sample
        windowIdx += 1;

        // Ony the last element in the window, perform buffered exiting
        if (windowIdx == WINDOW_SIZE) {
            buffered_max_prob_should_exit(exitResults, earlyResult, lfsrState, ELEVATE_COUNT, ELEVATE_REMAINDER, WINDOW_SIZE);

            for (i = 0; i < WINDOW_SIZE; i++) {
                if (!exitResults[i]) {
                    adaboost_inference_full(&fullResult, windowInputFeatures + i * NUM_FEATURES, &ENSEMBLE, earlyResult + i);
                    pred = fullResult.pred;
                } else {
                    pred = (earlyResult + i)->pred;
                }
            }

            lfsrState = lfsr_step(lfsrState);
            windowIdx = 0;
        }
#else

        // Perform the early inference
        adaboost_inference_early(&earlyResult, inputFeatures, &ENSEMBLE, PRECISION);

        #ifdef IS_MAX_PROB
        shouldExit = max_prob_should_exit(&earlyResult, THRESHOLD);
        #elif defined(IS_RANDOM)
        shouldExit = random_should_exit(EXIT_RATE, lfsrState);
        lfsrState = lfsr_step(lfsrState);
        #endif

        if (!shouldExit) {
            P1OUT |= BIT0;
            adaboost_inference_full(&fullResult, inputFeatures, &ENSEMBLE, &earlyResult);
            pred = fullResult.pred;
            P1OUT &= ~BIT0;
        } else {
            pred = earlyResult.pred;
        }
#endif

        // Update the state indices
        timerIdx = 0;
        sampleIdx += 1;

        if (sampleIdx == NUM_INPUTS) {
            sampleIdx = 0;
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
     * Timer Interrupts to make data pull requests.
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
                __bic_SR_register_on_exit(LPM3_bits | GIE);
            }

            break;
        default: break;
    }
}
