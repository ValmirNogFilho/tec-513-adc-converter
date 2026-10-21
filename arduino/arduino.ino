#include <avr/io.h>
#include <avr/interrupt.h>

// Tamanho do buffer na SRAM (1024 amostras = 1 KB)
#define BUFFER_SIZE 1024

volatile uint8_t buffer[BUFFER_SIZE];
volatile uint16_t index_buffer = 0;
volatile bool buffer_cheio = false;
volatile bool coletando = false;

void iniciar_captura() {
  index_buffer = 0;
  buffer_cheio = false;
  coletando = true;
  
  // Habilita a interrupção do ADC e dispara a conversão inicial
  ADCSRA |= (1 << ADIE) | (1 << ADSC);
}

void setup() {
  Serial.begin(115200);

  // 1. Configura Referência em AVCC (5V), alinhamento à esquerda (8 bits em ADCH) e Canal A0
  ADMUX = (0 << REFS1) | (1 << REFS0) | (1 << ADLAR);

#ifdef MUX5
  ADCSRB &= ~(1 << MUX5);
#endif

  // Desativa buffer digital no pino A0 para eliminar ruído
  DIDR0 |= (1 << ADC0D);

  // 2. Configura ADCSRA:
  // ADEN  = 1 (Habilita ADC)
  // ADATE = 1 (Habilita disparo automático / Free Running)
  // ADIE  = 1 (Habilita interrupção de conversão concluída)
  // ADPS2:0 = 100 (Prescaler 16 -> Clock do ADC em 1 MHz)
  ADCSRA = (1 << ADEN) | (1 << ADATE) | (1 << ADIE) |
           (1 << ADPS2) | (0 << ADPS1) | (0 << ADPS0);

  // Define fonte de disparo automático como Free Running (ADTS2:0 = 000 em ADCSRB)
  ADCSRB &= ~((1 << ADTS2) | (1 << ADTS1) | (1 << ADTS0));

  sei(); // Habilita interrupções globais

  // Dispara a primeira rodada de captura
  iniciar_captura();
}

// Interrupção executada exatamente no 13º ciclo de cada conversão (~13 us)
ISR(ADC_vect) {
  if (coletando) {
    buffer[index_buffer++] = ADCH; // Salva os 8 bits no buffer
    
    if (index_buffer >= BUFFER_SIZE) {
      coletando = false;
      buffer_cheio = true;
      ADCSRA &= ~(1 << ADIE); // Desabilita a interrupção temporariamente
    }
  }
}

void loop() {
  if (buffer_cheio) {
    // Transmite o bloco de 1024 bytes brutos diretamente pela serial para o Octave
    Serial.write((const uint8_t*)buffer, BUFFER_SIZE);

    // Aguarda um pequeno tempo antes do próximo ciclo de captura
    delay(500);

    // Reinicia a coleta para o próximo bloco de dados
    iniciar_captura();
  }
}