import serial
import time
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM5'  # Ajuste para a sua porta
baud_rate = 115200
tamanho_buffer = 4096

nivel_trigger = 0   # Meio da escala (em 8 bits, 0-255)
tamanho_janela = 1024  # Quantidade de pontos que vai aparecer na tela
# NOVA CONFIGURAÇÃO DO TRIGGER
margem_busca = tamanho_buffer - tamanho_janela  

from config import f_tri, fs_nominal as fs   # Frequência do gerador e fs: mudar só no config.py
from ao_vivo import resumo                     # Erros calculados durante a captura

print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)
time.sleep(3)
ser.reset_input_buffer()

# Configura o gráfico
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
# Agora a linha tem o tamanho da janela (800) e não do buffer inteiro (1024)
linha, = ax.plot(np.zeros(tamanho_janela), color='blue', linewidth=1.5, label='Arduino')

# Triangular ideal começando no vale, igual ao ponto do trigger
n = np.arange(tamanho_janela)
ideal = 255 * (1 - np.abs(2 * ((f_tri * n / fs) % 1) - 1))
ax.plot(n, ideal, color='red', linestyle='--', linewidth=1, label='Ideal')
ax.legend(loc='upper right')

ax.set_ylim(-0.2, 256)
ax.set_xlim(0, tamanho_janela) # Trava o eixo X no tamanho da janela
ax.set_xlabel(f'Amostras (0 a {tamanho_janela-1})')
ax.set_ylabel('Tensão (V)')
ax.set_title('Osciloscópio - Modo Auto (Gatilho em 2.5V)')
ax.grid(True)

print("Sincronizado! Solicitando dados ao Arduino...")
capturas = []

try:
    while plt.fignum_exists(fig.number):
        ser.reset_input_buffer()
        ser.write(b'S')
        ser.flush()

        dados_brutos = ser.read(tamanho_buffer)

        if len(dados_brutos) < tamanho_buffer:
            print(f"Aviso: Recebeu {len(dados_brutos)} bytes. Aguardando...")
            time.sleep(0.1)
            continue

        dados = np.frombuffer(dados_brutos, dtype=np.uint8)

        # Salva os últimos 50 blocos recebidos para o analise.py
        capturas.append(dados)
        np.save(f'captura_{f_tri:g}Hz.npy', np.array(capturas[-50:]))

        # Requisito 2 ao vivo: quantos dos 256 códigos apareceram neste bloco
        ax.set_title(f'Osciloscópio | f_tri = {f_tri:g} Hz | este bloco: {len(np.unique(dados))}/256 códigos')

        # A cada 10 blocos, os erros estáticos somando os blocos até agora
        if len(capturas) % 10 == 0:
            print(f"recebidos {len(capturas)} | usando os últimos " + resumo(capturas[-50:]))

        # --- LÓGICA DO TRIGGER (MODO AUTO) ---
        # Pega apenas a margem inicial para procurar a borda de subida
        area_busca = dados[:margem_busca]
        
        # Procura onde o ponto anterior é menor que 128 e o atual é maior ou igual
        cruzamentos = np.where((area_busca[:-1] <= nivel_trigger) & (area_busca[1:] >= nivel_trigger))[0]

        if len(cruzamentos) > 0:
            # Encontrou o trigger! O índice é o primeiro cruzamento (+1 por causa do offset do array)
            indice_trigger = cruzamentos[0] + 1
        else:
            # MODO AUTO: Não encontrou o cruzamento (pode ser uma linha reta ou sinal muito fraco)
            # Desenha a partir do zero de qualquer forma para a tela não congelar
            indice_trigger = 0 

        # Corta a janela exata de dados contínuos a partir do trigger
        dados_janela = dados[indice_trigger : indice_trigger + tamanho_janela]

        # Converte para tensão
        tensao = (dados_janela * 5.0) / 256.0

        # Atualiza o gráfico
        linha.set_ydata(dados_janela)
        plt.draw()
        plt.pause(0.01)

except Exception as e:
    print(f"\nErro no programa: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("Porta serial fechada com segurança.")
