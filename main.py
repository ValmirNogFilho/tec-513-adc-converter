import serial
import time
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM3'  # Ajuste para a sua porta
baud_rate = 115200
tamanho_buffer = 4096

tamanho_janela = 1024                           # pontos mostrados na tela
margem_busca = tamanho_buffer - tamanho_janela  # o gatilho só vale onde cabe a janela inteira

from config import f_tri, fs_nominal as fs
from ao_vivo import resumo, pior_rampa, instantes

print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)
time.sleep(3)
ser.reset_input_buffer()

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
linha, = ax.plot(np.zeros(tamanho_janela), color='blue', linewidth=1.5, label='Arduino')

# Triangular ideal começando no meio da escala subindo (fase 0,25), igual ao gatilho
n = np.arange(tamanho_janela)


def triangular_ideal(periodo):
    return 255 * (1 - np.abs(2 * ((n / periodo + 0.25) % 1) - 1))


linha_ideal, = ax.plot(n, triangular_ideal(fs / f_tri), color='red', linestyle='--', linewidth=1, label='Ideal')
ax.legend(loc='upper right')

ax.set_ylim(-0.2, 256)
ax.set_xlim(0, tamanho_janela)
ax.set_xlabel(f'Amostras (0 a {tamanho_janela-1})')
ax.set_ylabel('Código (0 a 255)')
ax.set_title('Osciloscópio - Modo Auto (gatilho no meio da escala, subindo)')
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

        # Salva os últimos 50 blocos para o analise.py
        capturas.append(dados)
        np.save(f'captura_{f_tri:g}Hz.npy', np.array(capturas[-50:]))

        # Requisito 2: códigos no bloco e na pior subida/descida
        pior = pior_rampa(dados)
        texto_rampa = f'{pior}/256' if pior is not None else 'sem rampa completa'
        ax.set_title(f'Osciloscópio | f_tri = {f_tri:g} Hz | bloco: {len(np.unique(dados))}/256 | pior rampa: {texto_rampa}')

        # A cada 10 blocos, erros estáticos dos blocos recebidos
        if len(capturas) % 10 == 0:
            print(f"recebidos {len(capturas)} | usando os últimos " + resumo(capturas[-50:]))

        # Gatilho: 1º cruzamento do meio da escala (128) subindo; sem cruzamento, começa no 0
        subidas = instantes(dados, True)
        cabem = subidas[subidas < margem_busca]
        indice_trigger = int(round(cabem[0])) if len(cabem) > 0 else 0

        # Ideal com o período medido no bloco (distância entre duas subidas)
        if len(subidas) >= 2:
            linha_ideal.set_ydata(triangular_ideal(np.mean(np.diff(subidas))))

        dados_janela = dados[indice_trigger : indice_trigger + tamanho_janela]

        linha.set_ydata(dados_janela)
        plt.draw()
        plt.pause(0.01)

except Exception as e:
    print(f"\nErro no programa: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("Porta serial fechada com segurança.")
