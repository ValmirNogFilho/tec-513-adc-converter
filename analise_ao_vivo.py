# =============================================================================
# analise_ao_vivo.py - recalcula tudo a cada bloco recebido do Arduino e atualiza os
# gráficos do analise.py em duas janelas, para ver na hora o efeito de mudar a
# frequência no gerador.
#
# Janela 1 (a cada bloco):  resumo, sinal no tempo, erro, requisito 2, módulo e fase.
# Janela 2 (a cada 3 blocos): curva de quantização e erros estáticos (dependem da média).
#
# - A frequência da triangular é medida no próprio sinal (não precisa mudar o config.py).
# - Quando a frequência muda mais de 1%, a média dos erros recomeça do zero.
# - As contas são as mesmas do analise.py (as fontes de cada fórmula estão comentadas lá).
#   Do config.py só são usados a amplitude do gerador (v_min, v_max), o vref, a fs e n_bits.
# - Fechar a janela 1 encerra; fechar a janela 2 só para de atualizá-la.
# - Não salva nada: para guardar uma captura e rodar o analise.py, use o main.py.
# =============================================================================

import serial
import time
import numpy as np
import matplotlib.pyplot as plt

from config import v_min, v_max, vref, fs_nominal, n_bits
from ao_vivo import instantes

porta_serial = 'COM3'    # Ajuste para a sua porta
baud_rate = 115200
tamanho_buffer = 4096    # Igual ao BUFFER_SIZE do arduino.ino
max_blocos = 10          # Blocos somados nos erros estáticos (recomeça quando a frequência muda)
mudanca = 0.01           # Mudança de frequência (1%) que faz a média recomeçar: permite ajuste fino
lentos_a_cada = 3        # A janela 2 (depende da média) atualiza a cada 3 blocos
n_harm = 20              # Harmônicas no espectro discreto

q = vref / 2**n_bits                  # 1 LSB em volts
codigos = np.arange(2**n_bits)
T_ideal = codigos[1:] - 0.125         # Transições ideais lendo só o ADCH (analise.py, seção 4)
degrau_y = np.r_[codigos, codigos[-1]]
n = np.arange(tamanho_buffer)
t_ms = n / fs_nominal * 1e3


# ---------- Contas (as mesmas do analise.py) ----------
def periodo_dos_blocos(blocos):
    # Distância entre cruzamentos do meio da escala no mesmo sentido (analise.py, seção 2).
    p = np.concatenate([np.diff(instantes(b, s)) for b in blocos for s in (True, False)])
    if len(p) == 0:
        return None
    p = p[np.abs(p - np.median(p)) < 0.2 * np.median(p)]   # descarta medidas fora do padrão
    return p.mean()


def erros_estaticos(blocos, periodo):
    # Transições pela contagem em períodos inteiros e os erros do DS (analise.py, seções 3 e 4).
    n_periodos = int(tamanho_buffer // periodo)
    M = int(round(n_periodos * periodo))
    contagem = sum(np.bincount(b[:M], minlength=2**n_bits) for b in blocos)
    T = (v_min + (v_max - v_min) * np.cumsum(contagem)[:-1] / (len(blocos) * M)) / q
    offset = T[0] - T_ideal[0]
    ganho = (T[-1] - offset) - T_ideal[-1]
    lsb_real = (T[-1] - T[0]) / (len(T) - 1)
    dnl = np.diff(T) / lsb_real - 1
    inl = (T - (T[0] + np.arange(len(T)) * lsb_real)) / lsb_real
    return dict(contagem=contagem, T=T, offset=offset, ganho=ganho, lsb_real=lsb_real,
                dnl=dnl, inl=inl, M=M, n_periodos=n_periodos)


def triangular_alinhada(bloco, periodo, subidas):
    # Triangular simulada encaixada no bloco pela fase de menor erro (analise.py, seção 5).
    # Para ser rápido: o 1º cruzamento do meio subindo dá a fase aproximada (a triangular
    # simulada cruza o meio subindo em 1/4 do período), e só se refina em volta dela.
    def tri(fase):
        x = (n + fase) / periodo
        return v_min + (v_max - v_min) * (1 - np.abs(2 * (x % 1) - 1))

    def desvio(fase):
        return np.mean((bloco - np.clip(tri(fase) / q, 0, 2**n_bits - 1)) ** 2)

    fase0 = 0.25 * periodo - subidas[0] if len(subidas) else 0.0
    fases = np.arange(fase0 - 5, fase0 + 5, 0.1)
    fase = fases[np.argmin([desvio(f) for f in fases])]
    return tri(fase)


def rampas(bloco, periodo):
    # Cada subida/descida, de um vale (código 0) ao pico seguinte (255) (analise.py, seção 4b).
    picos = np.where((bloco[1:] == 2**n_bits - 1) & (bloco[:-1] < 2**n_bits - 1))[0] + 1
    vales = np.where((bloco[1:] == 0) & (bloco[:-1] > 0))[0] + 1
    ext = sorted([(i, 'p') for i in picos] + [(i, 'v') for i in vales])
    return [bloco[i:j + 1] for (i, a), (j, d) in zip(ext, ext[1:]) if a != d and j - i > periodo / 4]


# ---------- As duas janelas ----------
plt.ion()
fig1, A = plt.subplots(3, 2, figsize=(15, 10), num='1 - Sinal e requisito 2 (ao vivo)')
fig2, B = plt.subplots(3, 3, figsize=(16, 10), num='2 - Erros estáticos (ao vivo)')
ja_ajustada = set()


def terminar(fig):
    # tight_layout é lento: só na primeira vez que a janela é desenhada
    if fig.number not in ja_ajustada:
        fig.tight_layout()
        ja_ajustada.add(fig.number)
    fig.canvas.draw_idle()


print("Conectando à porta e aguardando inicialização do Arduino...")
ser = serial.Serial(porta_serial, baud_rate, timeout=3)
time.sleep(3)
ser.reset_input_buffer()
print("Sincronizado! Mude a frequência no gerador e acompanhe as janelas.")

blocos, periodo_ref, contador = [], None, 0

try:
    while plt.fignum_exists(fig1.number):
        ser.reset_input_buffer()
        ser.write(b'S')
        ser.flush()
        bruto = ser.read(tamanho_buffer)
        if len(bruto) < tamanho_buffer:
            print(f"Aviso: Recebeu {len(bruto)} bytes. Aguardando...")
            time.sleep(0.1)
            continue
        bloco = np.frombuffer(bruto, dtype=np.uint8).astype(int)

        # Frequência medida neste bloco; se mudou mais que 'mudanca', recomeça a média
        p_bloco = periodo_dos_blocos([bloco])
        if p_bloco is None:
            fig1.suptitle('Não achei o período: a triangular está ligada e cruzando o meio da escala?')
            plt.pause(0.1)
            continue
        if periodo_ref is None or abs(p_bloco - periodo_ref) > mudanca * periodo_ref:
            blocos = []
        blocos = (blocos + [bloco])[-max_blocos:]
        periodo = periodo_dos_blocos(blocos)
        periodo_ref = periodo
        contador += 1

        f_medida = fs_nominal / periodo
        por_nivel = periodo / 2 / ((v_max - v_min) / q)
        E = erros_estaticos(blocos, periodo)
        subidas = instantes(bloco, True)
        v_sim = triangular_alinhada(bloco, periodo, subidas)
        x = v_sim / q

        # SINAD e ENOB (analise.py, seção 7)
        uso = (x > 0.5) & (x < 2**n_bits - 1.5)
        a, b = np.polyfit(x[uso], bloco[uso], 1)
        erro = bloco[uso] - (a * x[uso] + b)
        sinad = 10 * np.log10(np.var(a * x[uso]) / np.mean(erro ** 2))

        # Requisito 2 por rampa, em todos os blocos acumulados (analise.py, seção 4b)
        faltam = np.array([2**n_bits - len(np.unique(r)) for bl in blocos for r in rampas(bl, periodo)])

        # Espectro (analise.py, seção 8). FFT complexa: módulo (abs) e fase (angle).
        # A FFT começa no 1º cruzamento do meio da escala subindo (como o trigger do
        # osciloscópio), e não no começo do bloco: assim todo bloco "começa" no mesmo ponto da
        # onda e a fase fica parada (deslocar no tempo muda a fase: Lathi eq. 8.33, p. 714).
        i0 = int(round(subidas[0])) if len(subidas) else 0
        n_periodos = max(1, int((tamanho_buffer - i0) // periodo))
        M = int(round(n_periodos * periodo))
        trecho_med, trecho_ide = bloco[i0:i0 + M], x[i0:i0 + M]
        M = len(trecho_med)
        F_med = np.fft.rfft(trecho_med - trecho_med.mean())
        F_ide = np.fft.rfft(trecho_ide - trecho_ide.mean())
        ref = np.abs(F_ide).max()
        h = np.arange(1, n_harm + 1)
        h = h[h * n_periodos < len(F_med)]
        k_h = h * n_periodos

        # Trecho de 2 períodos a partir do meio da escala subindo (trigger)
        cabem = subidas[subidas < tamanho_buffer - 2 * periodo]
        ini = int(round(cabem[0])) if len(cabem) else 0
        fim = min(tamanho_buffer, ini + int(2 * periodo))
        tt = t_ms[ini:fim] - t_ms[ini]

        # ================= JANELA 1: sinal e requisito 2 (a cada bloco) =================
        for ax in A.flat:
            ax.cla()

        # Resumo com os números
        ax = A[0, 0]
        ax.axis('off')
        aviso = '' if E['contagem'][0] > 0 and E['contagem'][-1] > 0 else '\nAVISO: não chegou no 0 ou no 255'
        ax.text(0.0, 1.0,
                f'Frequência medida:  {f_medida:.2f} Hz\n'
                f'Amostras por nível: {por_nivel:.2f}  (>= 2 para a folga)\n'
                f'Blocos na média:    {len(blocos)}\n'
                f'Pior rampa:         {256 - faltam.max() if len(faltam) else "-"}/256   '
                f'(rampas completas: {np.sum(faltam == 0)}/{len(faltam)})\n'
                f'Códigos perdidos:   {int(np.sum(E["contagem"] == 0))}\n'
                f'Offset {E["offset"]:+.2f} | Ganho {E["ganho"]:+.2f} | '
                f'DNL {np.max(np.abs(E["dnl"])):.2f} | INL {np.max(np.abs(E["inl"])):.2f} LSB\n'
                f'SINAD {sinad:.1f} dB (ideal 48,2) | ENOB {sinad / 6.02:.2f} bits' + aviso,
                va='top', family='monospace', fontsize=11)
        ax.set_title('Resumo', loc='left')

        # Sinal no tempo: medido × simulado
        ax = A[0, 1]
        ax.step(tt, bloco[ini:fim] * q, where='post', label='ADC (medido)')
        ax.plot(tt, v_sim[ini:fim], 'r--', label='Triangular simulada')
        ax.set_title('Comparação no tempo')
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('Tensão (V)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        # Erro no tempo, com a média móvel (tendência = offset e ganho)
        ax = A[1, 0]
        erro_t = bloco[ini:fim] - x[ini:fim]
        jm = max(5, int(periodo / 20))
        ax.plot(tt, erro_t, linewidth=0.6, label='Erro de cada amostra')
        ax.plot(tt, np.convolve(erro_t, np.ones(jm) / jm, mode='same'), 'r', linewidth=2, label='Média móvel')
        ax.set_title('Medido - simulado')
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('LSB')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        # Requisito 2: códigos que faltaram em cada subida/descida
        ax = A[1, 1]
        if len(faltam):
            ax.vlines(np.arange(len(faltam)), 0, faltam, linewidth=2)
            ax.set_ylim(0, max(5, 1.1 * faltam.max()))
            ax.set_title(f'Requisito 2: {np.sum(faltam == 0)} de {len(faltam)} rampas com os 256 códigos')
        else:
            ax.set_title('Requisito 2: sem rampa completa (o sinal chega no 0 e no 255?)')
        ax.set_xlabel('Rampa (subida ou descida)')
        ax.set_ylabel('Códigos que faltaram')
        ax.grid(True)

        # Espectro discreto: módulo (linear). Ideal: só ímpares, com 1/h² (Lathi Ex. 6.2, p. 538).
        ax = A[2, 0]
        ax.vlines(h, 0, np.abs(F_med[k_h]) / ref, linewidth=2)
        ax.plot(h, np.abs(F_med[k_h]) / ref, 'o', color='C0', label='Medido')
        ax.plot(h, np.abs(F_ide[k_h]) / ref, 'rx', markersize=9, mew=2, label='Triangular simulada')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(h)
        ax.set_xticklabels([f'{v}f' for v in h], fontsize=7)
        ax.set_title('Espectro discreto: módulo')
        ax.set_ylabel('Amplitude (relativa à fundamental)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        # Espectro discreto: fase, só nas ímpares (nas pares o módulo ideal é ~0 e a fase não
        # tem sentido). Comparável porque a simulada está encaixada no mesmo bloco (Lathi eq. 8.33).
        ax = A[2, 1]
        hi = h[h % 2 == 1]
        fase_med = np.degrees(np.angle(F_med[hi * n_periodos]))
        fase_ide = np.degrees(np.angle(F_ide[hi * n_periodos]))
        ax.vlines(hi, 0, fase_med, linewidth=2)
        ax.plot(hi, fase_med, 'o', color='C0', label='Medido')
        ax.plot(hi, fase_ide, 'rx', markersize=9, mew=2, label='Triangular simulada')
        ax.set_ylim(-200, 200)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_xticks(h)
        ax.set_xticklabels([f'{v}f' for v in h], fontsize=7)
        ax.set_title('Espectro discreto: fase (só ímpares)')
        ax.set_ylabel('Fase (graus)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        fig1.suptitle(f'f ≈ {f_medida:.1f} Hz | {por_nivel:.2f} amostras por nível')
        terminar(fig1)

        # ================= JANELA 2: erros estáticos (a cada 'lentos_a_cada' blocos) =================
        if plt.fignum_exists(fig2.number) and (len(blocos) == 1 or contador % lentos_a_cada == 0):
            T, dnl, inl, offset, ganho = E['T'], E['dnl'], E['inl'], E['offset'], E['ganho']
            T_corr = T_ideal[0] + (T - T[0]) / E['lsb_real']   # sem offset e ganho (DS p. 278)
            for ax in B.flat:
                ax.cla()

            # Curva de quantização: (tensão simulada, código medido) de cada amostra do bloco
            ax = B[0, 0]
            ax.plot(x, bloco, '.', markersize=2, label='Real')
            ax.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r', where='post', linewidth=0.8, label='Ideal')
            ax.set_title('Curva de quantização')
            ax.set_xlabel('Tensão de entrada (LSB)')
            ax.set_ylabel('Código')
            ax.legend(fontsize=8)
            ax.grid(True)

            # DNL por código (linha em -1 = código perdido)
            ax = B[0, 1]
            ax.vlines(codigos[1:-1], 0, dnl, linewidth=1.5)
            ax.axhline(-1, color='r', linestyle='--', linewidth=1)
            ax.set_title(f'DNL ({len(blocos)} blocos): máx {np.max(np.abs(dnl)):.2f} LSB')
            ax.set_xlabel('Código')
            ax.set_ylabel('LSB')
            ax.grid(True)

            # INL por código
            ax = B[0, 2]
            ax.plot(codigos[1:], inl)
            ax.set_title(f'INL: máx {np.max(np.abs(inl)):.2f} LSB')
            ax.set_xlabel('Código')
            ax.set_ylabel('LSB')
            ax.grid(True)

            # Offset: zoom na 1ª transição (Fig. 26-11)
            ax = B[1, 0]
            ax.step(np.r_[0, T, 2**n_bits], degrau_y, where='post', label='Real')
            ax.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
            ax.set_xlim(0, 6)
            ax.set_ylim(-0.5, 5.5)
            ax.set_title(f'Offset: {offset:+.2f} LSB')
            ax.set_xlabel('Tensão de entrada (LSB)')
            ax.set_ylabel('Código')
            ax.legend(fontsize=8)
            ax.grid(True)

            # Ganho: zoom na última transição, sem o offset (Fig. 26-12)
            ax = B[1, 1]
            ax.step(np.r_[0, T - offset, 2**n_bits], degrau_y, where='post', label='Real, sem offset')
            ax.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
            ax.set_xlim(249, 256)
            ax.set_ylim(248.5, 255.5)
            ax.set_title(f'Ganho: {ganho:+.2f} LSB')
            ax.set_xlabel('Tensão de entrada (LSB)')
            ax.legend(fontsize=8)
            ax.grid(True)

            # Monotonicidade: quanto a tensão média sobe de um código para o próximo (> 0)
            ax = B[1, 2]
            soma = np.bincount(bloco, weights=v_sim, minlength=2**n_bits)
            cont_b = np.bincount(bloco, minlength=2**n_bits)
            pres = codigos[1:-1][cont_b[1:-1] > 0]
            media_v = soma[pres] / cont_b[pres]
            ax.plot(pres[1:], np.diff(media_v) / q, '.')
            ax.axhline(0, color='r', linestyle='--', linewidth=1)
            ax.set_title('Monotonicidade: pontos acima de 0')
            ax.set_xlabel('Código')
            ax.set_ylabel('Aumento da tensão média (LSB)')
            ax.grid(True)

            # Códigos perdidos: contagem de cada código (0 e 255 saturam)
            ax = B[2, 0]
            contagem = E['contagem']
            faltando = codigos[contagem == 0]
            topo = 1.3 * contagem[1:-1].max()
            ax.vlines(codigos, 0, contagem, linewidth=2)
            ax.vlines(faltando, 0, topo, colors='r', linewidth=2)
            ax.set_ylim(0, topo)
            ax.set_title(f'Códigos perdidos: {len(faltando)}')
            ax.set_xlabel('Código')
            ax.set_ylabel('Amostras')
            ax.grid(True)

            # INL no estilo da Fig. 26-13: zoom no pior ponto
            ax = B[2, 1]
            k = int(np.argmax(np.abs(inl)))
            ax.step(np.r_[0, T_corr, 2**n_bits], degrau_y, where='post', label='Real (sem offset e ganho)')
            ax.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
            ax.annotate('', xy=(T_corr[k], k + 0.5), xytext=(T_ideal[k], k + 0.5), arrowprops=dict(arrowstyle='<->'))
            ax.set_xlim(T_ideal[k] - 4, T_ideal[k] + 4)
            ax.set_ylim(k - 4, k + 5)
            ax.set_title(f'INL (Fig. 26-13): código {k + 1}, {inl[k]:+.2f} LSB')
            ax.set_xlabel('Tensão de entrada (LSB)')
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True)

            # DNL no estilo da Fig. 26-14: o degrau de maior |DNL|
            ax = B[2, 2]
            j = int(np.argmax(np.abs(dnl)))
            ax.step(np.r_[0, T_corr, 2**n_bits], degrau_y, where='post', label='Real (sem offset e ganho)')
            ax.step(np.r_[0, T_ideal, 2**n_bits], degrau_y, 'r--', where='post', label='Ideal')
            ax.annotate('', xy=(T_corr[j + 1], j + 1.15), xytext=(T_corr[j], j + 1.15), arrowprops=dict(arrowstyle='<->', color='C0'))
            ax.text((T_corr[j] + T_corr[j + 1]) / 2, j + 1.3, f'{1 + dnl[j]:.2f} LSB', ha='center', color='C0')
            ax.set_xlim(T_ideal[j] - 3, T_ideal[j] + 4)
            ax.set_ylim(j - 3, j + 5)
            ax.set_title(f'DNL (Fig. 26-14): código {j + 1}, {dnl[j]:+.2f} LSB')
            ax.set_xlabel('Tensão de entrada (LSB)')
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True)

            fig2.suptitle(f'Erros estáticos: média de {len(blocos)} blocos (f ≈ {f_medida:.1f} Hz)')
            terminar(fig2)

        plt.pause(0.01)

except Exception as e:
    print(f"\nErro no programa: {e}")
finally:
    if ser.is_open:
        ser.close()
        print("Porta serial fechada com segurança.")
