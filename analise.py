# analise.py - Requisitos 5 e 6 do Problema 2 (erros estáticos e características dinâmicas)
#
# Fluxo de uso:
#   1. Rodar o main.py com a triangular ligada; ele salva os últimos 50 blocos em
#      captura_<f_tri>Hz.npy.
#   2. Preencher os parâmetros abaixo com o que foi usado/medido no laboratório.
#   3. Rodar este script: ele imprime a comparação medido x datasheet e abre 5 gráficos.
#
# Abreviações das fontes:
#   DS     = datasheet ATmega2560 (Microchip 2549Q)
#   AVR120 = application note "Characterization and Calibration of the ADC on an AVR"
#   AVR127 = application note "Understanding ADC Parameters"
#   EN     = enunciado do Problema 2
#   [nossa] = dedução minha, sem fonte direta
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------- Parâmetros (preencher no laboratório) ----------
arquivo = 'captura_150.2Hz.npy'  # Arquivo salvo pelo main.py
f_tri = 150.2                    # Frequência configurada no gerador (Hz)

# Mínimo e máximo da triangular. Com o gerador em HighZ, o valor da tela é a tensão
# que chega no pino (anotação do lab de 21/10): 5,020 Vpp com offset de 2,520 V.
v_min = 2.520 - 5.020 / 2        # Offset - Vpp/2
v_max = 2.520 + 5.020 / 2        # Offset + Vpp/2

# Vref = AVCC, porque o código usa REFS0 = 1 (DS Tab. 26-3, p. 281). Na placa ligada
# pela USB, AVCC são os 5 V do notebook, que variam de um notebook para outro.
vref = 5.0                       # MEDIR com o multímetro no pino 5V da placa

# Lemos só o ADCH com ADLAR = 1, ou seja, os 8 bits de cima (DS p. 286).
n_bits = 8

# Taxa de amostragem nominal: clock de 16 MHz ÷ prescaler 16 = 1 MHz no ADC (DS Tab. 26-5,
# p. 285), e cada conversão leva 13 ciclos (DS p. 271): 1 MHz / 13 = 76 923 amostras/s.
fs_nominal = 16e6 / 16 / 13

q = vref / 2**n_bits             # 1 LSB em volts: Vref / 2^n (DS p. 277: "2^n steps (LSBs)")
codigos = np.arange(2**n_bits)   # Códigos possíveis: 0 a 255


# ---------- 1. Leitura dos dados ----------
# O arquivo tem até 50 blocos; cada bloco é uma captura contínua do buffer do Arduino.
capturas = np.load(arquivo).astype(int)
if capturas.ndim == 1:
    capturas = capturas[np.newaxis, :]
dados = capturas[0]              # Bloco usado nos gráficos, na SINAD e na FFT
B, N = capturas.shape


# ---------- 2. Período medido e taxa de amostragem (requisito 6) ----------
# O EN pede a taxa "com base no datasheet e medições experimentais". Como a frequência
# do gerador é conhecida, basta medir quantas amostras cabem em 1 período:
#   fs = (amostras por período) x f_tri  [nossa]
# O período é a distância entre dois cruzamentos do meio da escala (código 128) no
# mesmo sentido. O meio é usado porque fica longe do pico e do vale, onde a onda pode
# estar saturada.
w = 40   # Janela (em amostras) usada para confirmar o sentido e refinar o instante


def instantes(bloco, subida):
    # Candidatos: o código cruza 128 entre uma amostra e a seguinte.
    # A condição bloco[i + w] > bloco[i - w] exige que a onda esteja subindo de verdade
    # (ou descendo, no outro caso); sem ela, o ruído cria cruzamentos falsos perto do 128.
    i = np.arange(w, N - w)
    if subida:
        cruz = i[(bloco[i] < 128) & (bloco[i + 1] >= 128) & (bloco[i + w] > bloco[i - w])]
    else:
        cruz = i[(bloco[i] >= 128) & (bloco[i + 1] < 128) & (bloco[i + w] < bloco[i - w])]
    t, ultimo = [], -N
    for c in cruz:
        if c - ultimo < 4 * w:   # Mesmo cruzamento repetido pelo ruído: ignora
            continue
        ultimo = c
        # Instante refinado: numa janela em volta do cruzamento, as amostras abaixo de
        # 128 são as de antes do cruzamento (na subida). Contar elas dá a posição do
        # cruzamento com a média do ruído, em vez de depender de uma amostra só. [nossa]
        trecho = bloco[c - w:c + w]
        antes = np.sum(trecho < 128) if subida else np.sum(trecho >= 128)
        t.append(c - w + antes)
    return np.array(t, dtype=float)


periodo_nominal = fs_nominal / f_tri
periodos = np.concatenate([np.diff(instantes(b, s)) for b in capturas for s in (True, False)])
periodos = periodos[np.abs(periodos - periodo_nominal) < 0.2 * periodo_nominal]

if len(periodos) > 0:
    periodo = periodos.mean()
else:
    periodo = periodo_nominal
    print("Aviso: não achei 2 cruzamentos no mesmo sentido; usando o período nominal.")

fs_medida = periodo * f_tri


# ---------- 3. Transições da curva de quantização (base do requisito 5) ----------
# Todos os erros estáticos do DS são definidos pelas transições T(k), a tensão em que o
# código passa de k-1 para k (DS §26.6.4, p. 277-279).
# Método: AVR120 §1.7.1, p. 10: "a ramp input voltage is applied and all output value
# transitions are recorded". A triangular é essa rampa.
#
# Como achar T(k) [nossa]: a triangular sobe e desce em linha reta, então a fração do
# tempo em que ela fica abaixo de uma tensão V é (V - v_min)/(v_max - v_min). Se o
# código ficou abaixo de k, a tensão estava abaixo de T(k). Logo:
#   T(k) = v_min + (v_max - v_min) x (amostras com código < k) / (total de amostras)
# Isso só vale num número inteiro de períodos, por isso cada bloco é cortado em M amostras.
# É o princípio do histograma (nível largo recebe mais amostras), com a escala de tensão
# vinda da triangular simulada, como pede o EN req. 5.
n_periodos = int(N // periodo)
if n_periodos == 0:
    raise SystemExit(f"O bloco tem {N} amostras e o período tem {periodo:.0f}: aumente a f_tri.")

M = int(round(n_periodos * periodo))
contagem = sum(np.bincount(b[:M], minlength=2**n_bits) for b in capturas)  # amostras por código
abaixo = np.cumsum(contagem)[:-1]                     # amostras com código < k, para k = 1..255
T = (v_min + (v_max - v_min) * abaixo / (B * M)) / q   # T(k) em LSB de 8 bits


# ---------- 4. Erros estáticos (requisito 5) ----------
# Escada ideal dos nossos 8 bits [nossa]: no ADC de 10 bits ideal, a transição j fica em
# j - 0,5 LSB (1ª em 0,5 LSB, DS p. 277). Lendo só o ADCH, o código de 8 bits k começa
# quando o de 10 bits chega em 4k, então: T_ideal(k) = (4k - 0,5)/4 = k - 0,125 LSB(8).
# Sem esse ajuste, um ADC perfeito mostraria um "offset" de 0,375 LSB só por cortar os bits.
T_ideal = codigos[1:] - 0.125

# Offset: "The deviation of the first transition (0x000 to 0x001) compared to the ideal
# transition (at 0.5 LSB)" (DS p. 277). Aqui a ideal é a ajustada acima (0,875).
offset = T[0] - T_ideal[0]

# Ganho: "After adjusting for offset, the Gain Error is found as the deviation of the last
# transition (0x3FE to 0x3FF) compared to the ideal transition (at 1.5 LSB below
# maximum)" (DS p. 278). Por isso o offset é tirado antes de comparar.
ganho = (T[-1] - offset) - T_ideal[-1]

# DNL e INL devem ser medidas depois de compensar offset e ganho: "It is important that DNL
# and INL values are measured after offset and gain error compensation" (AVR120 §1.7.3,
# p. 10). Compensar os dois é o mesmo que usar a reta que liga a 1ª e a última transição
# medidas; o "LSB real" é a distância média entre transições nessa reta.
lsb_real = (T[-1] - T[0]) / (len(T) - 1)

# DNL: "The maximum deviation of the actual code width (the interval between two adjacent
# transitions) from the ideal code width (1 LSB)" (DS p. 279).
dnl = np.diff(T) / lsb_real - 1                      # Um valor por código, de 1 a 254

# INL: "After adjusting for offset and gain error, the INL is the maximum deviation of an
# actual transition compared to an ideal transition for any code" (DS p. 278).
# Com o LSB real, a INL é a soma acumulada das DNLs: "INL can be interpreted as a sum of
# DNLs" (AVR120 §1.7.2, p. 10).
inl = (T - (T[0] + np.arange(len(T)) * lsb_real)) / lsb_real

# Código perdido: código que nunca aparece; equivale a DNL = -1. "An ADC with DNL error less
# than ±1LSB guarantees no missing code" (AVR127 §3.4.3, p. 15).
faltando = codigos[contagem == 0]

# Se a triangular não chegou no código 0 ou no 255, a 1ª ou a última transição não foi
# vista, e o offset/ganho não valem (era o problema do Vref do notebook).
pontas_ok = contagem[0] > 0 and contagem[-1] > 0

# Exatidão absoluta: "The maximum deviation of an actual (unadjusted) transition compared to
# an ideal transition for any code. This is the compound effect of offset, gain error,
# differential error, non-linearity, and quantization error. Ideal value: ±0.5 LSB"
# (DS p. 279). É o único valor que a Tab. 31-9 (p. 365) dá a 1 MHz, nosso clock.
# Conta: maior desvio sem corrigir nada, x 4 para LSB de 10 bits, + 0,5 da quantização.
# Só vemos 1 a cada 4 transições de 10 bits, então pode sair um pouco menor. [nossa]
exatidao_10 = 4 * np.max(np.abs(T - T_ideal)) + 0.5


# ---------- 5. Triangular simulada, alinhada com a medida ----------
# EN req. 5: "deve-se gerar por simulação o sinal triangular e compará-lo com a saída obtida
# do conversor A/D". A triangular é gerada com v_min, v_max e o período medido.
n = np.arange(N)


def triangular(fase):
    # Triangular de período "periodo" (em amostras), com o vale em n = -fase.
    x = (n + fase) / periodo
    return v_min + (v_max - v_min) * (1 - np.abs(2 * (x % 1) - 1))


def desvio(fase):
    # Erro quadrático médio entre o medido e a simulada (limitada a 0..255, como o ADC).
    return np.mean((dados - np.clip(triangular(fase) / q, 0, 2**n_bits - 1)) ** 2)


# Não sabemos em que ponto da onda o bloco começou: testa todas as fases e fica com a de
# menor erro. Primeiro de 1 em 1 amostra, depois de 0,05 em 0,05 em volta da melhor. [nossa]
fases = np.arange(0, periodo, 1.0)
fase = fases[np.argmin([desvio(f) for f in fases])]
fases = np.arange(fase - 1, fase + 1, 0.05)
fase = fases[np.argmin([desvio(f) for f in fases])]

v_sim = triangular(fase)                              # Tensão simulada em cada amostra (V)
x = v_sim / q                                         # A mesma tensão em LSB
ideal = np.clip(np.floor(x + 0.125), 0, 2**n_bits - 1)  # O que um ADC ideal leria no ADCH


# ---------- 6. Monotonicidade (requisito 5) ----------
# Nenhum PDF da pasta define monotonicidade de ADC. Usamos a definição matemática: se a
# entrada aumenta, o código não diminui. Para o ruído não atrapalhar, comparamos a tensão
# média que gerou cada código: ela tem que crescer de um código para o seguinte. [nossa]
presentes = [k for k in codigos[1:-1] if np.any(dados == k)]
media_v = np.array([v_sim[dados == k].mean() for k in presentes])
nao_monotonicos = [presentes[j + 1] for j in range(len(presentes) - 1) if media_v[j + 1] <= media_v[j]]


# ---------- 7. SINAD e ENOB: faixa dinâmica medida (requisito 6) ----------
# SINAD = sinal / (ruído + distorção) (AVR127 §4.3, p. 17-18). O erro aqui é o medido menos
# a simulada, depois de tirar offset e ganho com uma reta (polyfit); sobram quantização,
# ruído e não linearidade juntos, por isso é SINAD e não só SNR (AVR127 §4.1).
# Usamos só a parte sem saturação (entre 0,5 e 254,5 LSB).
uso = (x > 0.5) & (x < 2**n_bits - 1.5)


def sinad(y):
    a, b = np.polyfit(x[uso], y[uso], 1)
    erro = y[uso] - (a * x[uso] + b)
    return 10 * np.log10(np.var(a * x[uso]) / np.mean(erro ** 2))


sinad_medida = sinad(dados)
sinad_ideal = sinad(ideal)      # Mesma conta com um ADC perfeito: confere a fórmula teórica

# Teórica: "SQNR(dB) = 1.76 + 6.02b" (Proakis eq. 1.4.32, p. 37; AVR127 §4.1, p. 17).
# O 1,76 é da senoide; o Proakis avisa que a fórmula "was derived for sinusoidal signals".
# Para a triangular, que passa o mesmo tempo em cada valor, fica 6,02 x n. [nossa]
sinad_teorica = 20 * np.log10(2**n_bits)

# ENOB: "the number of bits with which the ADC behaves like a perfect ADC";
# ENOB = (SINAD - 1,76)/6,02 (AVR127 §4.4, p. 18). Sem o 1,76 pelo mesmo motivo. [nossa]
enob = sinad_medida / 6.02


# ---------- 8. Espectro (FFT): comparação na frequência (produto do EN) ----------
# EN, produto: "Comparação (no tempo e na frequência) entre a trajetória ideal (simulada)
# e a real". A FFT calcula a DFT (Lathi §8.5-8.6, p. 702-723). Usamos M amostras, um número
# inteiro de períodos, para as harmônicas caírem nos pontos da FFT, sem vazamento
# (Lathi p. 704 e Exemplo M8.1, p. 724). Resolução: fs/M (Lathi eq. 8.20c, p. 704).
freq = np.arange(M // 2 + 1) * fs_medida / M
X_med = np.abs(np.fft.rfft(dados[:M] - dados[:M].mean()))
X_ideal = np.abs(np.fft.rfft(x[:M] - x[:M].mean()))
ref = X_ideal.max()             # 0 dB = fundamental da triangular simulada


# ---------- 9. Resultados: medido x datasheet ----------
def linha(nome, medido, referencia, fonte):
    print(f"  {nome:<24}{medido:<18}{referencia:<34}{fonte}")


print(f"\nArquivo: {arquivo}  ({B} blocos de {N} amostras)")
print(f"f_tri = {f_tri:g} Hz | triangular de {v_min:.3f} a {v_max:.3f} V | Vref = {vref} V | 1 LSB = {q*1e3:.2f} mV")
print(f"Período medido: {periodo:.1f} amostras ({len(periodos)} medições)")
if not pontas_ok:
    print("Aviso: o sinal não chegou no código 0 ou no 255; offset, ganho e exatidão não valem.")

print("\n=== Requisito 6: características dinâmicas ===")
linha("Item", "Medido", "Datasheet / teórico", "Fonte")
linha("Taxa de amostragem", f"{fs_medida:,.0f} /s", "76.923 /s (1 MHz / 13)", "DS p. 268 e 271")
linha("Tempo de conversão", f"{1e6 / fs_medida:.2f} us", "13 us", "DS p. 268; Tab. 31-9")
linha("SINAD (faixa dinâmica)", f"{sinad_medida:.1f} dB", f"{sinad_teorica:.1f} dB (ADC ideal 8 bits)", "Proakis p. 37; AVR127 §4.1 (DS não dá)")
linha("  ADC ideal simulado", f"{sinad_ideal:.1f} dB", "confere a teórica", "[nossa]")
linha("ENOB", f"{enob:.2f} bits", "8 bits (ideal)", "AVR127 §4.4 (DS não dá)")
print("  O tempo do nciclos.ino (conversão única) é medido à parte, no monitor serial.")

print("\n=== Requisito 5: erros estáticos (LSB de 8 bits; datasheet em LSB de 10 bits / 4) ===")
linha("Item", "Medido", "Datasheet (típico)", "Fonte")
linha("Offset", f"{offset:+.2f}", "-0,5 (-2 LSB10, a 200 kHz)", "DS p. 277; Tab. 31-9")
linha("Ganho", f"{ganho:+.2f}", "0,5 (2 LSB10, a 200 kHz)", "DS p. 278; Tab. 31-9")
linha("DNL (maior |valor|)", f"{np.max(np.abs(dnl)):.2f}", "0,125 (0,5 LSB10, a 200 kHz)", "DS p. 279; Tab. 31-9")
linha("INL (maior |valor|)", f"{np.max(np.abs(inl)):.2f}", "0,31 (1,25 LSB10, a 200 kHz)", "DS p. 278; Tab. 31-9")
linha("Exatidão absoluta", f"{exatidao_10:.2f} LSB10", "3 LSB10 (a 1 MHz)", "DS p. 279; Tab. 31-9")
linha("Códigos perdidos", str(len(faltando)), "0 esperado (DNL < 1)", "AVR127 §3.4.3")
linha("Monotônico", "sim" if not nao_monotonicos else "não", "sim esperado", "[nossa] (DS não dá)")
print(f"\n  1ª transição: {T[0]:.2f} LSB (ideal {T_ideal[0]:.3f}) | última: {T[-1]:.2f} LSB (ideal {T_ideal[-1]:.3f})")
print(f"  DNL: mín {dnl.min():+.2f}, máx {dnl.max():+.2f} | INL: mín {inl.min():+.2f}, máx {inl.max():+.2f}")
if len(faltando):
    print(f"  Códigos que faltaram: {faltando.tolist()}")
if nao_monotonicos:
    print(f"  Códigos fora de ordem: {nao_monotonicos}")
print("  Atenção: a Tab. 31-9 mede offset/ganho/DNL/INL a 200 kHz e Vref = 4 V; nós usamos 1 MHz.")


# ---------- 10. Gráficos ----------
t = n / fs_medida * 1e3          # Tempo em ms: t = n / fs (x[n] = x(nT))
fim = min(N, int(2 * periodo))   # Mostra 2 períodos

# Gráfico 1: comparação no tempo (produto do EN). A escada (step) é a reconstrução por
# retentor de ordem zero, que é o que o ADC entrega.
fig1, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
a1.step(t[:fim], dados[:fim] * q, where='post', label='ADC (medido)')
a1.plot(t[:fim], v_sim[:fim], 'r--', label='Triangular simulada')
a1.set_ylabel('Tensão (V)')
a1.set_title('Comparação no tempo')
a1.legend()
a1.grid(True)
a2.plot(t[:fim], dados[:fim] - x[:fim], linewidth=0.8)   # Degrau constante = offset; inclinação = ganho
a2.set_xlabel('Tempo (ms)')
a2.set_ylabel('Medido - simulado (LSB)')
a2.grid(True)

# Gráfico 2: curva de quantização real x ideal (EN, produto: "comparação entre as curvas de
# quantização real e ideal"). Cada ponto é (tensão simulada, código medido).
fig2, ax = plt.subplots(figsize=(8, 7))
xx = np.linspace(0, 2**n_bits, 5000)
ax.plot(x, dados, '.', markersize=2, label='Real (medido × simulado)')
ax.step(xx, np.clip(np.floor(xx + 0.125), 0, 2**n_bits - 1), 'r', where='post', linewidth=0.8, label='Ideal')
ax.set_xlabel('Tensão de entrada simulada (LSB)')
ax.set_ylabel('Código de saída')
ax.set_title('Curva de quantização: real × ideal')
ax.legend()
ax.grid(True)

# Gráfico 3: DNL e INL por código (como as Figs. 26-13 e 26-14 do DS, p. 279).
# A linha em -1 marca código perdido (AVR127 §3.4.3).
fig3, (b1, b2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
b1.bar(codigos[1:-1], dnl, width=1.0)
b1.axhline(-1, color='r', linestyle='--', linewidth=1)
b1.set_ylabel('DNL (LSB)')
b1.grid(True)
b2.plot(codigos[1:], inl)
b2.set_xlabel('Código')
b2.set_ylabel('INL (LSB)')
b2.grid(True)

# Gráfico 5: os outros erros estáticos.
fig5, ((c1, c2), (c3, c4)) = plt.subplots(2, 2, figsize=(12, 8))
fig5.suptitle('Erros estáticos')
degrau_y = np.r_[codigos, codigos[-1]]   # Escada: sobe 1 código em cada transição

# Offset: zoom na 1ª transição, como na Fig. 26-11 do DS (p. 278).
c1.step(np.r_[0, T, 2**n_bits], degrau_y, where='post', label='Real')
c1.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
c1.set_xlim(0, 6)
c1.set_ylim(-0.5, 5.5)
c1.set_title(f'Offset: 1ª transição deslocada {offset:+.2f} LSB')
c1.set_xlabel('Tensão de entrada (LSB)')
c1.set_ylabel('Código')
c1.legend()
c1.grid(True)

# Ganho: zoom na última transição, já sem o offset, como na Fig. 26-12 do DS (p. 278).
c2.step(np.r_[0, T - offset, 2**n_bits], degrau_y, where='post', label='Real, sem o offset')
c2.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
c2.set_xlim(249, 256)
c2.set_ylim(248.5, 255.5)
c2.set_title(f'Ganho: última transição deslocada {ganho:+.2f} LSB')
c2.set_xlabel('Tensão de entrada (LSB)')
c2.legend()
c2.grid(True)

# Monotonicidade: quanto a tensão média sobe de um código para o próximo; tem que ser > 0.
c3.plot(presentes[1:], np.diff(media_v) / q, '.')
c3.axhline(0, color='r', linestyle='--', linewidth=1)
c3.set_title('Monotonicidade: todos os pontos acima de 0')
c3.set_xlabel('Código')
c3.set_ylabel('Aumento da tensão média (LSB)')
c3.grid(True)

# Códigos perdidos: a contagem de cada código (histograma); os que deram 0 ficam em vermelho.
# A escala usa os códigos do meio, porque o 0 e o 255 acumulam a saturação.
topo = 1.3 * contagem[1:-1].max()
c4.bar(codigos, contagem, width=1.0)
c4.vlines(faltando, 0, topo, colors='r', linewidth=2, label=f'Faltando ({len(faltando)})')
c4.set_ylim(0, topo)
c4.set_title('Códigos perdidos (0 e 255 saturam e podem sair cortados)')
c4.set_xlabel('Código')
c4.set_ylabel('Amostras')
c4.legend()
c4.grid(True)
fig5.tight_layout()

# Gráfico 4: comparação na frequência. A triangular ideal tem só harmônicas ímpares, que
# caem com 1/n² (Lathi Ex. 6.2, p. 538); harmônicas pares ou extras no medido indicam
# não linearidade (AVR127 §4.2). O "chão" do medido é o ruído (o mesmo da SINAD).
fig4, ax = plt.subplots(figsize=(10, 5))
ax.plot(freq, 20 * np.log10(X_med / ref + 1e-12), label='Medido')
ax.plot(freq, 20 * np.log10(X_ideal / ref + 1e-12), 'r--', label='Triangular simulada')
ax.set_xlim(0, min(fs_medida / 2, 40 * f_tri))
ax.set_ylim(-100, 5)
ax.set_xlabel('Frequência (Hz)')
ax.set_ylabel('Magnitude (dB, relativa à fundamental)')
ax.set_title('Comparação na frequência')
ax.legend()
ax.grid(True)

plt.show()
