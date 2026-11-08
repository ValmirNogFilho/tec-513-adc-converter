void setup() {
  Serial.begin(115200);

  // Configura AVCC, alinhamento à esquerda e canal A0
  ADMUX = (0 << REFS1) | (1 << REFS0) | (1 << ADLAR);
#ifdef MUX5
  ADCSRB &= ~(1 << MUX5);
#endif
  // Prescaler 16 (clock do ADC = 1 MHz)
  ADCSRA = (1 << ADEN) | (1 << ADPS2) | (0 << ADPS1) | (0 << ADPS0);
  DIDR0 |= (1 << ADC0D);

  // Primeiramente executa uma conversão fictícia (dummy read de 25 ciclos)
  ADCSRA |= (1 << ADSC);
  while (ADCSRA & (1 << ADSC));

  // Mede o tempo exato para realizar 1000 conversões de 13 ciclos
  unsigned long tempo_inicio = micros();
  
  for (int i = 0; i < 1000; i++) {
    ADCSRA |= (1 << ADSC);             // Dispara conversão
    while (ADCSRA & (1 << ADSC));      // Aguarda concluir os 13 ciclos
    uint8_t resultado = ADCH;          // Lê o registrador
  }
  
  unsigned long tempo_fim = micros();

  float tempo_total_us = tempo_fim - tempo_inicio;
  float tempo_medio_por_amostra = tempo_total_us / 1000.0;

  Serial.print("Tempo total para 1000 conversoes: ");
  Serial.print(tempo_total_us);
  Serial.println(" us");

  Serial.print("Tempo medio por conversao: ");
  Serial.print(tempo_medio_por_amostra, 2);
  Serial.println(" us");
}

void loop() {
  // Nada a executar no loop
}