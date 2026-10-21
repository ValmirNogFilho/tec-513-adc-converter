import serial
import time
import numpy as np
import matplotlib.pyplot as plt

porta_serial = 'COM9'  # Ajuste para a sua porta (ex: 'COM3', 'COM4' ou '/dev/ttyACM0')
baud_rate = 115200
tamanho_buffer = 1024
n_capturas = 200       # 200 x 1024 amostras ~ 800 amostras por código
n_codigos = 256        # ADLAR = 1 -> 8 bits em ADCH

# Gerador: triangular de 0 a Vref (Load = HighZ). Use ~100-250 Hz, para que cada
# captura de 1024 amostras (~13 ms) cubra pelo menos um período inteiro; com
# frequência muito baixa cada captura vê só um trecho da rampa e o histograma fica torto.

print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)

# Aguarda 3 segundos para garantir que o bootloader do Arduino concluiu o reset
time.sleep(3)
ser.reset_input_buffer()

# Histograma acumulado: quantas vezes cada código (0-255) apareceu
contagem = np.zeros(n_codigos, dtype=np.int64)

try:
    capturas_ok = 0
    while capturas_ok < n_capturas:
        ser.reset_input_buffer()
        ser.write(b'S')
        ser.flush()

        dados_brutos = ser.read(tamanho_buffer)
        if len(dados_brutos) < tamanho_buffer:
            print(f"Aviso: Recebeu {len(dados_brutos)} bytes de {tamanho_buffer}. Repetindo...")
            continue

        dados = np.frombuffer(dados_brutos, dtype=np.uint8)
        contagem += np.bincount(dados, minlength=n_codigos)
        capturas_ok += 1
        print(f"\rCaptura {capturas_ok}/{n_capturas}", end="")
    print()
finally:
    if ser.is_open:
        ser.close()
        print("Porta serial fechada com segurança.")

# Códigos que nunca apareceram
codigos = np.arange(n_codigos)
presentes = codigos[contagem > 0]
faltando = codigos[contagem == 0]

print(f"\nTotal de amostras: {contagem.sum()}")
print(f"Faixa observada: código {presentes.min()} a {presentes.max()}")
print(f"Códigos presentes: {len(presentes)} de {n_codigos}")
if len(faltando) == 0:
    print("Nenhum código faltando: todos os 256 níveis foram capturados.")
else:
    print(f"Códigos faltando ({len(faltando)}): {faltando.tolist()}")

# DNL pelo método da densidade de códigos (code density / histograma):
# com uma triangular, a tensão fica o mesmo tempo em cada faixa, então cada
# código deveria aparecer o mesmo número de vezes. As pontas (0 e 255) são
# excluídas porque acumulam tudo o que fica abaixo/acima da faixa do ADC.
internos = slice(1, n_codigos - 1)
media_ideal = contagem[internos].mean()
dnl = contagem[internos] / media_ideal - 1      # em LSB
inl = np.cumsum(dnl)                            # INL ~ soma das DNLs

print(f"DNL: mín {dnl.min():+.2f} LSB, máx {dnl.max():+.2f} LSB")
print(f"INL: mín {inl.min():+.2f} LSB, máx {inl.max():+.2f} LSB")

# Gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.bar(codigos, contagem, width=1.0, color='blue')
ax1.bar(faltando, np.full(len(faltando), contagem.max()), width=1.0, color='red', alpha=0.4,
        label=f'Faltando ({len(faltando)})')
ax1.set_ylabel('Ocorrências')
ax1.set_title('Histograma de códigos do ADC (entrada triangular)')
ax1.legend()
ax1.grid(True)

ax2.plot(codigos[internos], dnl, color='blue', label='DNL')
ax2.plot(codigos[internos], inl, color='orange', label='INL')
ax2.axhline(-1, color='red', linestyle='--', linewidth=1)  # DNL = -1 LSB -> código faltando
ax2.set_xlabel('Código (0 a 255)')
ax2.set_ylabel('Erro (LSB)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
