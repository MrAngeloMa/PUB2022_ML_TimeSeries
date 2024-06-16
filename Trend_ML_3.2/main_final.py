# Importação e bibliotecas
from functions import *

reset = '\033[0m'        # Reseta a formatação
red = '\033[31m'         # Texto em vermelho
green = '\033[32m'       # Texto em verde
yellow = '\033[33m'      # Texto em amarelo
blue = '\033[34m'        # Texto em azul
magenta = '\033[35m'     # Texto em magenta
cyan = '\033[36m'        # Texto em cyan
white = '\033[37m'       # Texto em branco

# Previsão de têndencia com base nos indicadores
# Treina um modelo para calcular tendência com base nos indicadores

# Treina milhares de modelos variando seus atributos
features=['Volume', 'Open', 'rsi', 'ema_5_t', 'sma_5_t', 'ema_10_t', 'sma_10_t',
           'stoc_t', 'bol_t', 'adx_t', 'macd_t', 'macd', 'chai_t']

print((features))

# Aplica o Modelo

#model_name = 'ml_rf_Interval.in_45_minute_ac_52.96'
#ativo = 'ITUB4'

#print(pred_ac(model_name, ativo))
