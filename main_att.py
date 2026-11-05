import serial
import time
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM3'  # Ajuste para a sua porta
baud_rate = 115200
tamanho_buffer = 4096

tamanho_janela = 1024  # Quantidade de pontos que vai aparecer na tela
# O trigger procura o começo da janela só onde ainda cabe a janela inteira depois dele
margem_busca = tamanho_buffer - tamanho_janela

from config import f_tri, fs_nominal as fs        # Frequência do gerador e fs: mudar só no config.py
from ao_vivo import resumo, pior_rampa, instantes  # Erros e cruzamentos do meio da escala

print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)
time.sleep(3)
ser.reset_input_buffer()

# Configura o gráfico
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
linha, = ax.plot(np.zeros(tamanho_janela), color='blue', linewidth=1.5, label='Arduino')

# Triangular ideal começando no meio da escala subindo, igual ao ponto do trigger.
# Fase 0.25 = um quarto de período depois do vale, onde a triangular passa por 127,5 subindo.
n = np.arange(tamanho_janela)


def triangular_ideal(periodo):
    return 255 * (1 - np.abs(2 * ((n / periodo + 0.25) % 1) - 1))


linha_ideal, = ax.plot(n, triangular_ideal(fs / f_tri), color='red', linestyle='--', linewidth=1, label='Ideal')
ax.legend(loc='upper right')

ax.set_ylim(-0.2, 256)
ax.set_xlim(0, tamanho_janela)  # Trava o eixo X no tamanho da janela
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

        # Salva os últimos 50 blocos recebidos para o analise.py
        capturas.append(dados)
        np.save(f'captura_{f_tri:g}Hz.npy', np.array(capturas[-50:]))

        # Requisito 2 ao vivo: códigos no bloco inteiro e na subida/descida mais pobre do bloco
        pior = pior_rampa(dados)
        texto_rampa = f'{pior}/256' if pior is not None else 'sem rampa completa'
        ax.set_title(f'Osciloscópio | f_tri = {f_tri:g} Hz | bloco: {len(np.unique(dados))}/256 | pior rampa: {texto_rampa}')

        # A cada 10 blocos, os erros estáticos somando os blocos até agora
        if len(capturas) % 10 == 0:
            print(f"recebidos {len(capturas)} | usando os últimos " + resumo(capturas[-50:]))

        # --- LÓGICA DO TRIGGER (MODO AUTO) ---
        # Como num osciloscópio: começa a janela onde a onda cruza o meio da escala (128)
        # subindo. O instantes() só aceita o cruzamento se a onda estiver subindo de verdade,
        # então o ruído perto do 128 não engana. O meio é cruzado em qualquer frequência,
        # mesmo que a onda não chegue no 0 (o trigger no vale falhava nesses casos).
        subidas = instantes(dados, True)
        cabem = subidas[subidas < margem_busca]

        if len(cabem) > 0:
            indice_trigger = int(round(cabem[0]))
        else:
            # MODO AUTO: Não encontrou o cruzamento (pode ser uma linha reta ou sinal muito fraco)
            # Desenha a partir do zero de qualquer forma para a tela não congelar
            indice_trigger = 0

        # A ideal usa o período medido neste bloco (distância entre duas subidas), então
        # acompanha o gerador mesmo que a frequência mude sem mexer no config.py
        if len(subidas) >= 2:
            linha_ideal.set_ydata(triangular_ideal(np.mean(np.diff(subidas))))

        # Corta a janela exata de dados contínuos a partir do trigger
        dados_janela = dados[indice_trigger : indice_trigger + tamanho_janela]

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
