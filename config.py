# Parâmetros do experimento, usados pelo main.py, pelo ao_vivo.py e pelo analise.py.
# Se mudar algo no gerador ou na placa, mude só aqui.

# Frequência configurada no gerador (Hz). O main.py salva a captura como captura_<f_tri>Hz.npy.
f_tri =75

amp = 5
offset = 2.5
amp_margem = 0.045
offset_margem = 0.05

# Mínimo e máximo da triangular. Com o gerador em HighZ, o valor da tela é a tensão
# que chega no pino (anotação do lab de 21/10): 5,020 Vpp com offset de 2,520 V.
v_min = (offset+offset_margem) - (amp + amp_margem) / 2        # Offset - Vpp/2
v_max = (offset+offset_margem) + (amp + amp_margem) / 2        # Offset + Vpp/2

# Vref = AVCC, porque o código usa REFS0 = 1 (DS Tab. 26-3, p. 281). Na placa ligada
# pela USB, AVCC são os 5 V do notebook, que variam de um notebook para outro.
# Pode ser medido no pino AREF (DS p. 275) ou no pino 5V.
vref = amp+amp_margem                       # Não medido: valor nominal

# Taxa de amostragem nominal: clock de 16 MHz ÷ prescaler 16 = 1 MHz no ADC (DS Tab. 26-5,
# p. 285), e cada conversão leva 13 ciclos (DS p. 271): 1 MHz / 13 = 76 923 amostras/s.
fs_nominal = 16e6 / 16 / 13

# Lemos só o ADCH com ADLAR = 1, ou seja, os 8 bits de cima (DS p. 286).
n_bits = 8
