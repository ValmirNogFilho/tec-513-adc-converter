import serial
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM10'  # Ajuste para a sua porta (ex: 'COM3', 'COM4' ou '/dev/ttyACM0')
baud_rate = 115200
tamanho_buffer = 1024

# Abre a conexão serial
ser = serial.Serial(porta_serial, baud_rate, timeout=2)
ser.reset_input_buffer() # Descarta dados antigos acumulados

# Configura o modo de gráfico interativo do Matplotlib
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
linha, = ax.plot(np.zeros(tamanho_buffer), color='blue', linewidth=1.5)

ax.set_ylim(-0.2, 5.2)  # Fixa o eixo Y entre 0V e 5V
ax.set_xlabel('Amostras (0 a 1023)')
ax.set_ylabel('Tensão (V)')
ax.set_title('Leitura em Tempo Real do Potenciômetro - ATmega2560')
ax.grid(True)

print("Lendo serial em tempo real... Para encerrar, feche a janela do gráfico ou pressione Ctrl+C.")

try:
    while True:
        # Lê o bloco de 1024 bytes brutos enviado pelo Arduino
        dados_brutos = ser.read(tamanho_buffer)
        
        if len(dados_brutos) == tamanho_buffer:
            # Converte bytes para array de inteiros (0-255) e depois para Tensão (0-5V)
            dados = np.frombuffer(dados_brutos, dtype=np.uint8)
            tensao = (dados * 5.0) / 256.0
            
            # Atualiza os dados da linha sem recriar a janela
            linha.set_ydata(tensao)
            plt.draw()
            plt.pause(0.01) # Pequena pausa para processar o evento de tela

except (KeyboardInterrupt, plt.Series):
    print("\nEncerrado pelo usuário.")
finally:
    ser.close()
    print("Porta serial fechada.")
