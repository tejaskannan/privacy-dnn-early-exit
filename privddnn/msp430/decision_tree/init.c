/*
 * This file contains functions which initialize the device. This initialization
 * includes MSP pins, UART configuration, and the clock system.
 */
#include "init.h"

void init_gpio(void) {
    /**
     * Initializes all pins to output and sets pins to LOW. This
     * prevents unnecessary current consumption by floating pins.
     */
    P1OUT = 0x0;
    P1DIR = 0xFF;

    P2OUT = 0x0;
    P2DIR = 0xFF;

    P3OUT = 0x0;
    P3DIR = 0xFF;

    P4OUT = 0x0;
    P4DIR = 0xFF;

    P5OUT = 0x0;
    P5DIR = 0xFF;

    P6OUT = 0x0;
    P6DIR = 0xFF;

    P7OUT = 0x0;
    P7DIR = 0xFF;

    P8OUT = 0x0;
    P8DIR = 0xFF;

    P9OUT = 0x0;
    P9DIR = 0xFF;

    PAOUT = 0x0;
    PADIR = 0xFF;

    PBOUT = 0x0;
    PBDIR = 0xFF;

    PCOUT = 0x0;
    PCDIR = 0xFF;

    PDOUT = 0x0;
    PDDIR = 0xFF;

    PEOUT = 0x0;
    PEDIR = 0x0;

    PJOUT = 0x0;
    PJDIR = 0xFFFF;
}


void init_uart_pins(void) {
    /*
     * Configures USCI_A3 Pins
     */
    P6SEL1 &=  ~(BIT0 | BIT1);
    P6SEL0 |= (BIT0 | BIT1);
}


void init_uart_system(void) {
    /**
     * Initializes the UART system by setting the correct baudrate.
     */
    // Set clock system with DCO of ~1MHz
    CSCTL0_H = CSKEY_H;  // Unlock clock system control registers
    CSCTL1 = DCOFSEL_0;  // Set DCO to 1MHz
    CSCTL2 = SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 =  DIVA__1 | DIVS__1 | DIVM__1;  // Set dividers
    CSCTL0_H = 0;   // Lock the clock system control registers

    // Configure USCI_A3 for UART
    UCA3CTLW0 = UCSWRST;  // Put eUSCI in reset
    UCA3CTLW0 |= UCSSEL__SMCLK;  // CLK = SMCLK
    UCA3BRW = 6;  // integer part of (10^6) / (16 * 9600)
    UCA3MCTLW |= UCOS16 | UCBRF_8 | 0xAA;  // UCBRSx = 0xAA (User Guide Table 30-4)
    UCA3CTLW0 &= ~UCSWRST;  // Initialize eUSCI
    UCA3IE |= UCRXIE;  // Enable RX Interrupt
}


void init_timer(void) {
    // Set Timer A0 to SMCLK, clear TAR, enable overflow interrupt,
    // set to up mode. With these settings, we have 5 timer interrupts / second.
    TA0CCR0 = 50000;
    TA0CTL = TASSEL__SMCLK | MC__UP | TACLR | TAIE | ID__4;
}
