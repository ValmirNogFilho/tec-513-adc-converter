import serial
import time
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM9'  # Ajuste para a sua porta (ex: 'COM3', 'COM4' ou '/dev/ttyACM0')
baud_rate = 115200
tamanho_buffer = 1024

print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)

# Aguarda 3 segundos para garantir que o bootloader do Arduino concluiu o reset
time.sleep(3)
ser.reset_input_buffer()

# Configura o gráfico
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
linha, = ax.plot(np.zeros(tamanho_buffer), color='blue', linewidth=1.5)

ax.set_ylim(-0.2, 5.2)
ax.set_xlabel('Amostras (0 a 1023)')
ax.set_ylabel('Tensão (V)')
ax.set_title('Leitura em Tempo Real via Handshake - ATmega2560')
ax.grid(True)

print("Sincronizado! Solicitando dados ao Arduino...")

try:
    while plt.fignum_exists(fig.number):
        # Limpa resíduos antigos do buffer de entrada
        ser.reset_input_buffer()
        
        # Envia a solicitação 'S' e FORÇA o envio imediato pela USB
        ser.write(b'S')
        ser.flush()

        # Lê os 1024 bytes do Arduino
        dados_brutos = ser.read(tamanho_buffer)

        # Checagem de diagnóstico
        if len(dados_brutos) < tamanho_buffer:
            print(f"Aviso: Recebeu {len(dados_brutos)} bytes de {tamanho_buffer}. Aguardando próximo ciclo...")
            time.sleep(0.1)
            continue

        # Converte os dados brutos de 8 bits (0-255) para Tensão (0-5V)
        dados = np.frombuffer(dados_brutos, dtype=np.uint8)

        # Trigger: começa o gráfico onde a onda cruza o meio da escala subindo
        i = np.arange(10, tamanho_buffer - 11)
        subida = i[(dados[i] < 128) & (dados[i + 1] >= 128) & (dados[i + 10] > dados[i - 10])]
        if len(subida) > 0:
            dados = np.roll(dados, -subida[0])

        tensao = (dados * 5.0) / 256.0

        # Atualiza o gráfico
        linha.set_ydata(tensao)
        plt.draw()
        plt.pause(0.01)

except Exception as e:
    print(f"\nErro no programa: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("Porta serial fechada com segurança.")
