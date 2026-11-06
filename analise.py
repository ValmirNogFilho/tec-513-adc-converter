# analise.py - Requisitos 2, 5 e 6 do Problema 2 (erros estáticos e características dinâmicas)
#
# Uso: rodar o main_att.py (salva captura_<f_tri>Hz.npy), conferir o config.py e rodar este script.

import numpy as np
import matplotlib.pyplot as plt

from config import f_tri, v_min, v_max, vref, fs_nominal, n_bits

arquivo = f'captura_{f_tri:g}Hz.npy'

q = vref / 2**n_bits             # 1 LSB em volts
codigos = np.arange(2**n_bits)   # 0 a 255


# ---------- 1. Leitura ----------
capturas = np.load(arquivo).astype(int)
if capturas.ndim == 1:
    capturas = capturas[np.newaxis, :]
dados = capturas[0]              # Bloco usado nos gráficos, na SINAD e na FFT
B, N = capturas.shape


# ---------- 2. Período medido e taxa de amostragem (req. 6) ----------
# fs = (amostras por período) x f_tri. O período é a distância entre dois cruzamentos
# do meio da escala (código 128) no mesmo sentido.
w = 40   # Janela (amostras) para confirmar o sentido e refinar o instante


def instantes(bloco, subida):
    # Cruzamentos do código 128; exige que a onda esteja mesmo subindo (ou descendo)
    i = np.arange(w, N - w)
    if subida:
        cruz = i[(bloco[i] < 128) & (bloco[i + 1] >= 128) & (bloco[i + w] > bloco[i - w])]
    else:
        cruz = i[(bloco[i] >= 128) & (bloco[i + 1] < 128) & (bloco[i + w] < bloco[i - w])]
    t, ultimo = [], -N
    for c in cruz:
        if c - ultimo < 4 * w:   # Mesmo cruzamento repetido pelo ruído
            continue
        ultimo = c
        # Refina o instante contando as amostras antes do cruzamento na janela
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


# ---------- 3. Transições T(k) (base do req. 5) ----------
# Método da rampa. Na triangular, a fração de amostras com código < k
# dá a posição de T(k):  T(k) = v_min + (v_max - v_min) x N(<k) / N_total.
# Só vale em períodos inteiros, por isso cada bloco é cortado em M amostras.
n_periodos = int(N // periodo)
if n_periodos == 0:
    raise SystemExit(f"O bloco tem {N} amostras e o período tem {periodo:.0f}: aumente a f_tri.")

M = int(round(n_periodos * periodo))
contagem = sum(np.bincount(b[:M], minlength=2**n_bits) for b in capturas)  # amostras por código
abaixo = np.cumsum(contagem)[:-1]                       # amostras com código < k
T = (v_min + (v_max - v_min) * abaixo / (B * M)) / q   # T(k) em LSB


# ---------- 4. Erros estáticos (req. 5) ----------
# Ideal lendo só o ADCH: T_ideal(k) = (4k - 0,5)/4 = k - 0,125 LSB
T_ideal = codigos[1:] - 0.125

offset = T[0] - T_ideal[0]                     # desvio da 1ª transição
ganho = (T[-1] - offset) - T_ideal[-1]         # desvio da última transição, sem o offset

# DNL e INL depois de tirar offset e ganho: usa o LSB real
lsb_real = (T[-1] - T[0]) / (len(T) - 1)
dnl = np.diff(T) / lsb_real - 1                                    # largura do degrau - 1
inl = (T - (T[0] + np.arange(len(T)) * lsb_real)) / lsb_real       # desvio da reta

faltando = codigos[contagem == 0]              # códigos perdidos
pontas_ok = contagem[0] > 0 and contagem[-1] > 0

# Exatidão absoluta: maior desvio sem correção, em LSB de 10 bits, + 0,5
exatidao_10 = 4 * np.max(np.abs(T - T_ideal)) + 0.5


# ---------- 4b. Requisito 2: cada subida e cada descida ----------
# Uma rampa vai de um vale (código 0) ao pico seguinte (código 255), ou o contrário.
def rampas(bloco):
    picos = np.where((bloco[1:] == 2**n_bits - 1) & (bloco[:-1] < 2**n_bits - 1))[0] + 1
    vales = np.where((bloco[1:] == 0) & (bloco[:-1] > 0))[0] + 1
    ext = sorted([(i, 'p') for i in picos] + [(i, 'v') for i in vales])
    return [bloco[i:j + 1] for (i, a), (j, d) in zip(ext, ext[1:]) if a != d and j - i > periodo / 4]


faltam_rampa = np.array([2**n_bits - len(np.unique(r)) for b in capturas for r in rampas(b)])


# ---------- 5. Triangular simulada (req. 5) ----------
n = np.arange(N)


def triangular(fase):
    # Rampa de 0 a 1 por período, dobrada em triângulo e escalada para v_min..v_max
    x = (n + fase) / periodo
    return v_min + (v_max - v_min) * (1 - np.abs(2 * (x % 1) - 1))


def desvio(fase):
    return np.mean((dados - np.clip(triangular(fase) / q, 0, 2**n_bits - 1)) ** 2)


# Fase de menor erro quadrático: busca de 1 em 1 amostra, depois de 0,05 em 0,05
fases = np.arange(0, periodo, 1.0)
fase = fases[np.argmin([desvio(f) for f in fases])]
fases = np.arange(fase - 1, fase + 1, 0.05)
fase = fases[np.argmin([desvio(f) for f in fases])]

v_sim = triangular(fase)                                  # tensão simulada (V)
x = v_sim / q                                             # em LSB
ideal = np.clip(np.floor(x + 0.125), 0, 2**n_bits - 1)    # leitura de um ADC ideal no ADCH


# ---------- 6. Monotonicidade (req. 5) ----------
# A tensão média de cada código tem que crescer de um código para o próximo.
presentes = [k for k in codigos[1:-1] if np.any(dados == k)]
media_v = np.array([v_sim[dados == k].mean() for k in presentes])
nao_monotonicos = [presentes[j + 1] for j in range(len(presentes) - 1) if media_v[j + 1] <= media_v[j]]


# ---------- 7. SINAD e ENOB (req. 6) ----------
# Erro = medido - reta ajustada (tira offset e ganho); só a faixa sem saturação
uso = (x > 0.5) & (x < 2**n_bits - 1.5)


def sinad(y):
    a, b = np.polyfit(x[uso], y[uso], 1)
    erro = y[uso] - (a * x[uso] + b)
    return 10 * np.log10(np.var(a * x[uso]) / np.mean(erro ** 2))


sinad_medida = sinad(dados)
sinad_ideal = sinad(ideal)                     # ADC perfeito: confere a teórica
sinad_teorica = 20 * np.log10(2**n_bits)       # 6,02n, sem o 1,76 da senoide
enob = sinad_medida / 6.02                     # sem o 1,76, pelo mesmo motivo


# ---------- 8. FFT (comparação na frequência) ----------
# M amostras = períodos inteiros: cada harmônica cai num ponto da FFT, sem vazamento.
# A harmônica h fica no ponto h x n_periodos.
X_med = np.abs(np.fft.rfft(dados[:M] - dados[:M].mean()))
X_ideal = np.abs(np.fft.rfft(x[:M] - x[:M].mean()))
ref = X_ideal.max()                            # fundamental da simulada = 1

# Fase: FFT a partir do 1º cruzamento do meio subindo, para a fase não depender de onde o
# bloco começou. Esperado nas ímpares: -90°, +90°, -90°...
subidas0 = instantes(dados, True)
i0 = int(round(subidas0[0])) if len(subidas0) else 0
n_periodos_f = max(1, int((N - i0) // periodo))
trecho_med = dados[i0:i0 + int(round(n_periodos_f * periodo))]
trecho_ide = x[i0:i0 + len(trecho_med)]
F_med = np.fft.rfft(trecho_med - trecho_med.mean())
F_ide = np.fft.rfft(trecho_ide - trecho_ide.mean())


# ---------- 9. Resultados no terminal ----------
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

print("\n=== Requisito 2: níveis capturados em cada subida/descida (sem somar) ===")
if len(faltam_rampa) == 0:
    print("  Não achei rampas completas: o sinal precisa chegar no código 0 e no 255.")
else:
    print(f"  Rampas analisadas: {len(faltam_rampa)} | códigos faltando por rampa: média {faltam_rampa.mean():.1f}, "
          f"mín {faltam_rampa.min()}, máx {faltam_rampa.max()} | rampas com os 256: {np.sum(faltam_rampa == 0)}")
    print(f"  Amostras por nível: {periodo / 2 / ((v_max - v_min) / q):.2f} "
          f"(com folga de 2x, f_tri <= {fs_medida / 2**(n_bits + 2):.0f} Hz)")

print("\n=== Comparação na frequência: altura das harmônicas (relativa à fundamental) ===")
linha("Harmônica", "Medido", "Triangular simulada", "Teoria (1/n², só ímpares)")
for h in range(1, 11):
    k = h * n_periodos
    teoria = f"{1 / h**2:.4f}" if h % 2 else "0"
    linha(f"{h}ª ({h * f_tri:.0f} Hz)", f"{X_med[k] / ref:.4f}", f"{X_ideal[k] / ref:.4f}", teoria)


# ---------- 10. Gráficos ----------
t = n / fs_medida * 1e3          # tempo em ms
fim = min(N, int(2 * periodo))   # 2 períodos
degrau_y = np.r_[codigos, codigos[-1]]

# 1: medido x triangular simulada no tempo
fig1, a1 = plt.subplots(figsize=(10, 5), num='1 - Comparação no tempo')
a1.step(t[:fim], dados[:fim] * q, where='post', label='ADC (medido)')
a1.plot(t[:fim], v_sim[:fim], 'r--', label='Triangular simulada')
a1.set_xlabel('Tempo (ms)')
a1.set_ylabel('Tensão (V)')
a1.set_title('Comparação no tempo')
a1.legend()
a1.grid(True)

# 2: curva de quantização real x ideal; cada ponto é (tensão simulada, código medido)
fig2, ax = plt.subplots(figsize=(8, 7), num='2 - Curva de quantização')
xx = np.linspace(0, 2**n_bits, 5000)
ax.plot(x, dados, '.', markersize=2, label='Real (medido × simulado)')
ax.step(xx, np.clip(np.floor(xx + 0.125), 0, 2**n_bits - 1), 'r', where='post', linewidth=0.8, label='Ideal')
ax.set_xlabel('Tensão de entrada simulada (LSB)')
ax.set_ylabel('Código de saída')
ax.set_title('Curva de quantização: real × ideal')
ax.legend()
ax.grid(True)

# 5: offset, ganho, monotonicidade e códigos perdidos
fig5, ((c1, c2), (c3, c4)) = plt.subplots(2, 2, figsize=(12, 8), num='5 - Erros estáticos')
fig5.suptitle('Erros estáticos')

c1.step(np.r_[0, T, 2**n_bits], degrau_y, where='post', label='Real')
c1.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
c1.set_xlim(0, 6)
c1.set_ylim(-0.5, 5.5)
c1.set_title(f'Offset: 1ª transição deslocada {offset:+.2f} LSB')
c1.set_xlabel('Tensão de entrada (LSB)')
c1.set_ylabel('Código')
c1.legend()
c1.grid(True)

c2.step(np.r_[0, T - offset, 2**n_bits], degrau_y, where='post', label='Real, sem o offset')
c2.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
c2.set_xlim(249, 256)
c2.set_ylim(248.5, 255.5)
c2.set_title(f'Ganho: última transição deslocada {ganho:+.2f} LSB')
c2.set_xlabel('Tensão de entrada (LSB)')
c2.legend()
c2.grid(True)

c3.plot(presentes[1:], np.diff(media_v) / q, '.')
c3.axhline(0, color='r', linestyle='--', linewidth=1)
c3.set_title('Monotonicidade: todos os pontos acima de 0')
c3.set_xlabel('Código')
c3.set_ylabel('Aumento da tensão média (LSB)')
c3.grid(True)

topo = 1.3 * contagem[1:-1].max()   # escala pelos códigos do meio; 0 e 255 acumulam saturação
c4.bar(codigos, contagem, width=1.0)
c4.vlines(faltando, 0, topo, colors='r', linewidth=2, label=f'Faltando ({len(faltando)})')
c4.set_ylim(0, topo)
c4.set_title('Códigos perdidos (0 e 255 saturam e podem sair cortados)')
c4.set_xlabel('Código')
c4.set_ylabel('Amostras')
c4.legend()
c4.grid(True)
fig5.tight_layout()

# 6: requisito 2, códigos pulados em cada rampa (todas deveriam ficar em 0)
if len(faltam_rampa) > 0:
    fig6, ax6 = plt.subplots(figsize=(10, 4), num='6 - Níveis por rampa (req. 2)')
    ax6.bar(np.arange(len(faltam_rampa)), faltam_rampa, width=1.0)
    ax6.axhline(faltam_rampa.mean(), color='r', linestyle='--', linewidth=1, label=f'Média: {faltam_rampa.mean():.1f}')
    ax6.set_xlabel('Rampa (subida ou descida, na ordem da captura)')
    ax6.set_ylabel('Códigos que faltaram')
    ax6.set_title(f'Requisito 2: códigos pulados em cada rampa ({np.sum(faltam_rampa == 0)} de {len(faltam_rampa)} rampas completas)')
    ax6.legend()
    ax6.grid(True)

# 7: espectro discreto nas harmônicas, módulo e fase
n_harm = 20
h = np.arange(1, n_harm + 1)
k_h = h * n_periodos
med_lin = X_med[k_h] / ref
ide_lin = X_ideal[k_h] / ref

fig7, (ax7, ax7f) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, num='7 - Espectro discreto (módulo e fase)')
ax7.stem(h * f_tri, med_lin, basefmt=' ', label='Medido')
ax7.plot(h * f_tri, ide_lin, 'rx', markersize=9, mew=2, label='Triangular simulada')
ax7.set_ylim(0, 1.05)
ax7.set_ylabel('Amplitude (relativa à fundamental)')
ax7.set_title('Espectro discreto: altura de cada harmônica')
ax7.legend()
ax7.grid(True)

h_imp = h[(h % 2 == 1) & (h * n_periodos_f < len(F_med))]   # fase só nas ímpares
fase_med = np.degrees(np.angle(F_med[h_imp * n_periodos_f]))
fase_ide = np.degrees(np.angle(F_ide[h_imp * n_periodos_f]))
ax7f.stem(h_imp * f_tri, fase_med, basefmt=' ', label='Medido')
ax7f.plot(h_imp * f_tri, fase_ide, 'rx', markersize=9, mew=2, label='Triangular simulada')
ax7f.set_ylim(-200, 200)
ax7f.set_yticks([-180, -90, 0, 90, 180])
ax7f.set_xticks(h * f_tri)
ax7f.set_xticklabels([f'{v}f' for v in h], fontsize=8)
ax7f.set_xlabel(f'Harmônica (f = {f_tri:g} Hz)')
ax7f.set_ylabel('Fase (graus)')
ax7f.set_title('Fase das harmônicas ímpares (a partir do meio da escala, subindo)')
ax7f.legend()
ax7f.grid(True)
fig7.tight_layout()

# 9: triangular ideal na frequência: só ímpares, caindo com 1/n²
fig9, ax9 = plt.subplots(figsize=(10, 5), num='9 - Triangular ideal na frequência')
ax9.stem(h, ide_lin, linefmt='r-', markerfmt='ro', basefmt=' ', label='FFT da triangular simulada')
h_teo = h[h % 2 == 1]
ax9.plot(h_teo, 1 / h_teo**2, 'kx', markersize=9, mew=2, label='Teoria: 1/n² (só ímpares)')
for v in h_teo[:4]:
    ax9.annotate(f'1/{v * v}' if v > 1 else '1', (v, 1 / v**2), textcoords='offset points', xytext=(8, 4))
ax9.set_xticks(h)
ax9.set_xticklabels([f'{v}f' for v in h], fontsize=8)
ax9.set_ylim(0, 1.1)
ax9.set_xlabel(f'Harmônica (f = {f_tri:g} Hz)')
ax9.set_ylabel('Amplitude (relativa à fundamental)')
ax9.set_title('Triangular ideal na frequência: só harmônicas ímpares, caindo com 1/n²')
ax9.legend()
ax9.grid(True)

# 8: DNL e INL no estilo do datasheet, zoom no pior ponto, escada real sem offset e ganho
T_corr = T_ideal[0] + (T - T[0]) / lsb_real
fig8, (e1, e2) = plt.subplots(1, 2, figsize=(13, 5), num='8 - DNL e INL no estilo do datasheet')

k_inl = int(np.argmax(np.abs(inl)))          # transição de maior |INL|
e1.step(np.r_[0, T_corr, 2**n_bits], degrau_y, where='post', label='Real (sem offset e ganho)')
e1.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
e1.annotate('', xy=(T_corr[k_inl], k_inl + 0.5), xytext=(T_ideal[k_inl], k_inl + 0.5),
            arrowprops=dict(arrowstyle='<->', color='k'))
e1.text((T_corr[k_inl] + T_ideal[k_inl]) / 2, k_inl + 0.7, f'INL = {inl[k_inl]:+.2f} LSB', ha='center')
e1.set_xlim(T_ideal[k_inl] - 4, T_ideal[k_inl] + 4)
e1.set_ylim(k_inl - 4, k_inl + 5)
e1.set_title(f'INL (Fig. 26-13): maior desvio, na transição para o código {k_inl + 1}')
e1.set_xlabel('Tensão de entrada (LSB)')
e1.set_ylabel('Código')
e1.legend(loc='upper left')
e1.grid(True)

j = int(np.argmax(np.abs(dnl)))              # degrau de maior |DNL|
e2.step(np.r_[0, T_corr, 2**n_bits], degrau_y, where='post', label='Real (sem offset e ganho)')
e2.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
e2.annotate('', xy=(T_corr[j + 1], j + 1.15), xytext=(T_corr[j], j + 1.15), arrowprops=dict(arrowstyle='<->', color='C0'))
e2.text((T_corr[j] + T_corr[j + 1]) / 2, j + 1.3, f'real: {1 + dnl[j]:.2f} LSB', ha='center', color='C0')
e2.annotate('', xy=(T_ideal[j + 1], j + 0.85), xytext=(T_ideal[j], j + 0.85), arrowprops=dict(arrowstyle='<->', color='r'))
e2.text((T_ideal[j] + T_ideal[j + 1]) / 2, j + 0.55, 'ideal: 1 LSB', ha='center', color='r')
e2.set_xlim(T_ideal[j] - 3, T_ideal[j] + 4)
e2.set_ylim(j - 3, j + 5)
e2.set_title(f'DNL (Fig. 26-14): degrau do código {j + 1}, DNL = {dnl[j]:+.2f} LSB')
e2.set_xlabel('Tensão de entrada (LSB)')
e2.legend(loc='upper left')
e2.grid(True)
fig8.tight_layout()

plt.show()
