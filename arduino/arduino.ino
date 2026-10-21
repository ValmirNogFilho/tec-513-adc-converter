#include <avr/io.h>
#include <avr/interrupt.h>

#define BUFFER_SIZE 1024

volatile uint8_t buffer[BUFFER_SIZE];
volatile uint16_t index_buffer = 0;
volatile bool buffer_cheio = false;
volatile bool coletando = false;

void iniciar_captura() {
  index_buffer = 0;
  buffer_cheio = false;
  coletando = true;
  
  // Habilita interrupção do ADC e força o disparo imediato da conversão
  ADCSRA |= (1 << ADIE) | (1 << ADSC);
}

void setup() {
  Serial.begin(115200);

  // AVCC (5V), alinhamento à esquerda (8 bits em ADCH) e Canal A0
  ADMUX = (0 << REFS1) | (1 << REFS0) | (1 << ADLAR);

#ifdef MUX5
  ADCSRB &= ~(1 << MUX5);
#endif

  // Desativa buffer digital do pino A0
  DIDR0 |= (1 << ADC0D);

  // ADEN=1, ADATE=1 (Free Running), Prescaler 16 (Clock do ADC = 1 MHz)
  ADCSRA = (1 << ADEN) | (1 << ADATE) | (1 << ADPS2) | (0 << ADPS1) | (0 << ADPS0);

  // Trigger em Free Running (ADTS2:0 = 000 em ADCSRB)
  ADCSRB &= ~((1 << ADTS2) | (1 << ADTS1) | (1 << ADTS0));

  sei(); // Habilita interrupções globais
}

ISR(ADC_vect) {
  if (coletando) {
    buffer[index_buffer++] = ADCH;
    
    if (index_buffer >= BUFFER_SIZE) {
      coletando = false;
      buffer_cheio = true;
      ADCSRA &= ~(1 << ADIE); // Pausa interrupção
    }
  }
}

void loop() {
  if (Serial.available() > 0) {
    char comando = Serial.read();

    if (comando == 'S') {
      iniciar_captura();

      while (!buffer_cheio);

      Serial.write((const uint8_t*)buffer, BUFFER_SIZE);
    }
  }
}