# Cálculos rápidos, sem gráficos, para o main.py mostrar enquanto captura.
# São as mesmas contas das seções 2 a 4 do analise.py; as fontes de cada fórmula estão lá.
# A SINAD, a FFT e a monotonicidade ficam só no analise.py, porque precisam alinhar a
# triangular simulada com o bloco, o que é lento demais para fazer a cada captura.

import numpy as np
from config import f_tri, v_min, v_max, vref, fs_nominal, n_bits

w = 40


def instantes(bloco, subida):
    # Instantes em que a onda cruza o código 128 (subindo ou descendo); igual ao analise.py.
    N = len(bloco)
    i = np.arange(w, N - w)
    if subida:
        cruz = i[(bloco[i] < 128) & (bloco[i + 1] >= 128) & (bloco[i + w] > bloco[i - w])]
    else:
        cruz = i[(bloco[i] >= 128) & (bloco[i + 1] < 128) & (bloco[i + w] < bloco[i - w])]
    t, ultimo = [], -N
    for c in cruz:
        if c - ultimo < 4 * w:
            continue
        ultimo = c
        trecho = bloco[c - w:c + w]
        antes = np.sum(trecho < 128) if subida else np.sum(trecho >= 128)
        t.append(c - w + antes)
    return np.array(t, dtype=float)


def pior_rampa(bloco):
    # Requisito 2: quantos códigos aparecem na subida/descida mais pobre do bloco (igual à
    # seção 4b do analise.py). Uma rampa vai de um vale (código 0) ao pico seguinte (255).
    bloco = np.asarray(bloco).astype(int)
    picos = np.where((bloco[1:] == 2**n_bits - 1) & (bloco[:-1] < 2**n_bits - 1))[0] + 1
    vales = np.where((bloco[1:] == 0) & (bloco[:-1] > 0))[0] + 1
    ext = sorted([(i, 'p') for i in picos] + [(i, 'v') for i in vales])
    minimo = fs_nominal / f_tri / 4
    contagens = [len(np.unique(bloco[i:j + 1])) for (i, a), (j, d) in zip(ext, ext[1:]) if a != d and j - i > minimo]
    return min(contagens) if contagens else None


def resumo(capturas):
    capturas = np.asarray(capturas).astype(int)
    B, N = capturas.shape
    q = vref / 2**n_bits

    # Período em amostras e taxa de amostragem (analise.py, seção 2)
    periodo_nominal = fs_nominal / f_tri
    periodos = np.concatenate([np.diff(instantes(b, s)) for b in capturas for s in (True, False)])
    periodos = periodos[np.abs(periodos - periodo_nominal) < 0.2 * periodo_nominal]
    if len(periodos) == 0:
        return f"{B} blocos | não achei o período: confira o gerador e o f_tri do config.py"
    periodo = periodos.mean()
    n_periodos = int(N // periodo)
    if n_periodos == 0:
        return f"{B} blocos | o período não cabe no bloco: aumente a f_tri"

    # Transições pela contagem em períodos inteiros (analise.py, seção 3)
    M = int(round(n_periodos * periodo))
    contagem = sum(np.bincount(b[:M], minlength=2**n_bits) for b in capturas)
    abaixo = np.cumsum(contagem)[:-1]
    T = (v_min + (v_max - v_min) * abaixo / (B * M)) / q

    # Erros estáticos (analise.py, seção 4)
    T_ideal = np.arange(1, 2**n_bits) - 0.125
    offset = T[0] - T_ideal[0]
    ganho = (T[-1] - offset) - T_ideal[-1]
    lsb_real = (T[-1] - T[0]) / (len(T) - 1)
    dnl = np.diff(T) / lsb_real - 1
    inl = (T - (T[0] + np.arange(len(T)) * lsb_real)) / lsb_real
    faltando = int(np.sum(contagem == 0))

    return (f"{B} blocos | fs {periodo * f_tri:,.0f}/s | faltando {faltando} | "
            f"offset {offset:+.2f} | ganho {ganho:+.2f} | DNL {np.max(np.abs(dnl)):.2f} | "
            f"INL {np.max(np.abs(inl)):.2f} LSB")
