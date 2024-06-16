
import numpy as np
import pandas as pd
from scipy.stats import randint
import time
import datetime as dt

import yfinance as yf
import talib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from tvdatafeed_lib.main import TvDatafeed, Interval
from sklearn.model_selection import RandomizedSearchCV

import logging
import pickle

logging.basicConfig(filename='Resultados/Registro.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt = '%m-%d %H:%M')

logging.getLogger("yfinance").setLevel(logging.INFO)


"""
## Funções dos Indicadores Técnicos e suas tendências

A função deve receber um dataframe e um período, então calculo o indicador e devolve a tendência

Sugestões de indicaores:
- Média Móvel Simples (SMA) de diferentes períodos ✓
- Média Móvel Exponencial (EMA) de diferentes períodos ✓
- Índice de Força Relativa (RSI) ✓
- Bandas de Bollinger ✓
- MACD (Moving Average Convergence Divergence)✓
- Estocástico ✓
- Volume ✓
- Oscilador Chaikin ✓
- Fibonacci Retracement

"""
reset = '\033[0m'        # Reseta a formatação
red = '\033[31m'         # Texto em vermelho
green = '\033[32m'       # Texto em verde
yellow = '\033[33m'      # Texto em amarelo
blue = '\033[34m'        # Texto em azul
magenta = '\033[35m'     # Texto em magenta
cyan = '\033[36m'        # Texto em cyan
white = '\033[37m'       # Texto em branco

"""### profit ✓"""

def get_PROFIT(df):
    retorno = df['Close'].pct_change() * 100

    trend = 0 * retorno
    #trend[ retorno > (retorno.mean() + 0.24*retorno.std() ) ] = 1
    #trend[ retorno < (retorno.mean() - 0.24*retorno.std() )] = -1
    trend[ retorno >= 0.0 ] = 1
    trend[ retorno < 0.0 ] = -1

    return pd.DataFrame({'profit': retorno.round(2), 'profit_t': trend})

"""### Simple Moving Average (SMA) ✓"""

def get_SMA(df, window):
    sma = df['Close'].rolling(window).mean()

    trend = 0 * sma
    trend[ df['Close'] > sma] = 1
    trend[ df['Close'] < sma] = -1

    return pd.DataFrame({f'sma_{window}': sma.round(2), f'sma_{window}_t': trend})

"""### Exponential Moving Average (EMA) ✓"""

def get_EMA(df, window):
    close = df['Close']
    ema = df['Close'].ewm(span=window, adjust=False).mean()

    trend = 0 * ema
    trend[ close > ema] = 1
    trend[ close < ema] = -1

    return pd.DataFrame({f'ema_{window}': ema.round(2), f'ema_{window}_t': trend})

"""### Relative Strength Index (RSI) ✓"""

def get_RSI(df, window = 14):
    sma = df['Close'].rolling(window).mean()
    rsi = talib.RSI(df['Close'].values, timeperiod=window)

    trend = 0 * sma
    trend[rsi < 30] = 1
    trend[rsi > 70] = -1


    return pd.DataFrame({'rsi': rsi.round(2), 'rsi_t': trend})

"""### Bollinger Bands (BOL) ✓"""

def get_BOL(df, window = 20, n_std = 2):
    # Cálculo das bandas de Bollinger
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * n_std)
    lower_band = rolling_mean - (rolling_std * n_std)

    # Identificação da tendência
    """
    trend = []
    for i in range(len(df)):
        if df['Close'][i] > upper_band[i]:
            trend.append(1)  # Tendência de alta
        elif df['Close'][i] < lower_band[i]:
            trend.append(-1)  # Tendência de baixa
        else:
            trend.append(0)  # Sem tendência definida
    """

    trend = rolling_mean * 0
    trend[df['Close'] > upper_band] = 1
    trend[df['Close'] < lower_band] = -1
    trend[(df['Close'] < upper_band) & (df['Close'] > lower_band)] = 0

    # Criação de um novo dataframe com a coluna "tendencia"
    df_trend = pd.DataFrame({ 'bol_t': trend})
    return df_trend

"""### Moving Average Convergence Divergence (MACD) ✓"""

def get_MACD(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    trend = 0 * macd
    trend[macd > signal_line] = 1
    trend[macd < signal_line] = -1

    return pd.DataFrame({'macd': macd.round(2), 'macd_t': trend})

"""### Stochastic Oscillators (STOC) ✓"""

def get_STOC(df, janela=14, suavizacao=3, sobrevenda=20, sobrecompra=80):
    # Cálculo das linhas %K e %D
    high_n = df['High'].rolling(window=janela).max()
    low_n = df['Low'].rolling(window=janela).min()
    k_percent = 100 * ((df['Close'] - low_n) / (high_n - low_n))
    d_percent = k_percent.rolling(window=suavizacao).mean()

    # Identificação da tendência
    tendencia = []
    for i in range(len(df)):
        if k_percent.iloc[i] < sobrevenda and d_percent.iloc[i] < sobrevenda and k_percent.iloc[i] > d_percent.iloc[i]:
            tendencia.append(1)  # Tendência de alta
        elif k_percent.iloc[i] > sobrecompra and d_percent.iloc[i] > sobrecompra and k_percent.iloc[i] < d_percent.iloc[i]:
            tendencia.append(-1)  # Tendência de baixa
        else:
            tendencia.append(0)  # Tendência neutra

    return pd.DataFrame(tendencia, index=df.index, columns=['stoc_t'])

"""### **Oscilador Chaikin (CHAI) ✓"""

def get_CHAI(df):
    money_flow_multiplier = 2 * ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    adl = money_flow_volume.cumsum()
    chaikin = pd.DataFrame({'chai': adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()})
    chaikin['chai_t'] = chaikin['chai'].apply(lambda x: 1 if x > 0 else -1)

    return chaikin

"""### **Average Directional Index (ADX) ✓"""

def get_ADX(data, adx_period=14, adx_threshold=25):

    df = data.copy()
    # Calcula o True Range (TR) para cada dia
    df['TR'] = np.nan
    df['TR'] = np.maximum(df['High'] - df['Low'], df['High'] - df['Close'].shift(1))
    df['TR'] = np.maximum(df['TR'], df['Close'].shift(1) - df['Low'])

    # Calcula o Directional Movement (DM) para cada dia
    df['DMplus'] = np.nan
    df['DMminus'] = np.nan
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                            df['High'] - df['High'].shift(1), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                             df['Low'].shift(1) - df['Low'], 0)

    # Calcula o True Directional Indicator (DI) para cada dia
    df['DIplus'] = np.nan
    df['DIminus'] = np.nan
    df['DIplus'] = 100 * (df['DMplus'].rolling(window=14).sum() / df['TR'].rolling(window = adx_period).sum())
    df['DIminus'] = 100 * (df['DMminus'].rolling(window=14).sum() / df['TR'].rolling(window = adx_period).sum())

    # Calcula o Average Directional Index (ADX) para cada dia
    df['DX'] = np.nan
    df['DX'] = 100 * np.abs((df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus']))
    df['adx'] = np.nan
    df['adx'] = df['DX'].rolling(window=14).mean()

    #########################################################

    adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=adx_period)
    plus_di = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=adx_period)
    minus_di = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=adx_period)

    df['adx'] = adx
    df['PlusDI'] = plus_di
    df['MinusDI'] = minus_di

    df.loc[(adx > adx_threshold) & (plus_di > minus_di), 'adx_t'] = 1
    df.loc[(adx > adx_threshold) & (plus_di < minus_di), 'adx_t'] = -1
    df.loc[adx <= adx_threshold, 'adx_t'] = 0

    return df[['adx',  'adx_t']]


def get_ac(ativo, start, end, intervalo = '1d'):
  df = yf.download(tickers=ativo, interval=intervalo, start=start, end=end)
  df.head()
  return df

"""### Calculo Indicadores e União"""

def mg_ind(data):
  # Rastreadores
  ema10_df = get_EMA(data, 10)
  sma10_df = get_SMA(data, 10)
  ema5_df  = get_EMA(data, 5)
  sma5_df  = get_SMA(data, 5)
  macd_df   = get_MACD(data)

  stoc_df  = get_STOC(data)
  bol_df   = get_BOL(data)
  adx_df   = get_ADX(data)
  chai_df  = get_CHAI(data)

  rsi_df    = get_RSI(data, 14)
  profit_df = get_PROFIT(data)

  # Adiciona os indicadores no Dataframe principal
  data = data.join([rsi_df, sma10_df, ema10_df, sma5_df, ema5_df, macd_df, profit_df, stoc_df, bol_df, adx_df, chai_df])
  #data = data.join([rsi_df, sma10_df, ema10_df, sma5_df, ema5_df, macd_df, profit_df])

  trend_df = pd.concat([rsi_df, sma10_df, ema10_df, sma5_df, ema5_df, macd_df, profit_df, stoc_df, bol_df, adx_df, chai_df])
  #trend_df = pd.concat([rsi_df, sma10_df, ema10_df, sma5_df, ema5_df, macd_df, profit_df])

  return data

"""### Tratamento"""
def process(data):
  data.dropna(inplace=True)

  # Rastreadores
  data['sma_5_t']  = data['sma_5_t'].astype(int)
  data['ema_5_t']  = data['ema_5_t'].astype(int)
  data['sma_10_t'] = data['sma_10_t'].astype(int)
  data['ema_10_t'] = data['ema_10_t'].astype(int)
  data['macd_t']   = data['macd_t'].astype(int)

  # Osciladores
  data['chai_t'] = data['chai_t'].astype(int) # ?
  data['stoc_t'] = data['stoc_t'].astype(int)
  data['adx_t']  = data['adx_t'].astype(int) # ?
  data['rsi_t']  = data['rsi_t'].astype(int)

  # Indicadores de Volatilidade
  data['bol_t'] = data['bol_t'].astype(int)

  # Tendência real
  data['profit_t'] = data['profit_t'].astype(int)

  data['profit2'] = data['profit'].shift(-1)  
  data['profit2_t'] = data['profit_t'].shift(-1)       

  return data

### Separação de Dados
def split(data):
  Y = data['profit_t']
  features=['Volume', 'rsi', 'ema_5_t', 'sma_5_t', 'ema_10_t', 'sma_10_t', 'stoc_t', 'bol_t', 'adx_t', 'macd_t','macd', 'chai_t']
  X = data[features]

  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

  return x_train, x_test, y_train, y_test

def split_custom(X,Y):

  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

  return x_train, x_test, y_train, y_test

def train_RF(x_train, y_train, **kwargs):
  
    n_e = kwargs.get('n_estimators', 64)
    n_j = kwargs.get('n_jobs', 5)
    max_d = kwargs.get('max_depth', 9)
    max_f = kwargs.get('max_features', 1)
    min_l = kwargs.get('min_samples_leaf', 3)
    min_s = kwargs.get('min_samples_split', 5)

    # Calculou-se os melhores hiparâmetros considerando todas as features e estabeleceu-se como o padrão

    ml = RandomForestClassifier(n_estimators = n_e, n_jobs = n_j, max_depth = max_d,
        max_features = max_f, min_samples_leaf = min_l, min_samples_split = min_s)

    ml.fit(x_train, y_train)

    return ml

def input_ac():
    name = input("Nome da ação: ")
    start = input("Início: ")
    return name, start

def data_inter(intervalo, n = 5000, ativo = 'PETR3'):

    tv = TvDatafeed()

    ativos_org_var = {}
    ativos_org_var[ativo] = 'BMFBOVESPA'

    for symb_dict in ativos_org_var.items():
        data = tv.get_hist(*symb_dict, n_bars = n,
                        interval = intervalo).reset_index()

    data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'}, errors="raise", inplace=True)


    #### altera os indices do dataframe para as datas de cada linha
    data = data.set_index('datetime')

    ### Calculo Indicadores e União
    data = mg_ind(data)

    ## Tratamento
    data = process(data)
    data.dropna(inplace=True)

    return data

def performance(X, x_train, y_train, y_test, predictions, ml_rf):

    print('\n\nClassification Report')
    print(classification_report(y_test, predictions))

    rf_matrix = confusion_matrix(y_test, predictions)
    print('\nConfusion_matrix')
    print(rf_matrix)

    # Print the Accuracy of our Model.
    print('\nCorrect Prediction (%): ', accuracy_score(y_test, predictions, normalize = True) * 100.0)

    true_negatives  = rf_matrix[0][0]
    false_negatives = rf_matrix[1][0]
    true_positives  = rf_matrix[1][1]
    false_positives = rf_matrix[0][1]

    accuracy    = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    percision   = true_positives / (true_positives + false_positives)
    recall      = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    print('\nAccuracy.....: {}'.format(float(accuracy)))
    print('Percision....: {}'.format(float(percision)))
    print('Recall.......: {}'.format(float(recall)))
    print('Specificity..: {}'.format(float(specificity)))


    # Calculate feature importance and store in pandas series
    feature_imp = pd.Series(ml_rf.feature_importances_, index = X.columns).sort_values(ascending=False)
    print('\nFeature importance')
    print( feature_imp )

    # Por outro método

    # Substitua o nome do seu modelo e dos dados
    sel = SelectFromModel(ml_rf, threshold=0.04)
    sel.fit(x_train, y_train)

    # Substitua o nome das suas colunas de features
    selected_feat= x_train.columns[(sel.get_support())]
    print(selected_feat)

def best_hp(x_train, y_train):
    ## Define a grade de possíveis valores para os hiperparâmetros
    param_distribs = {
        'n_estimators': randint(low=10, high=80),
        'max_features': randint(low=1, high=3),
        'max_depth': randint(low=1, high=8),
        'min_samples_split': randint(low=2, high=8),
        'n_jobs': randint(low=2, high=8),
        'min_samples_leaf': randint(low=1, high=8)
    }

    # Cria um modelo com o estimador RandomForestClassifier
    rnd_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_distribs,
                                    n_iter=25, cv=4, scoring='accuracy', random_state=42)

    # Treina o modelo com a busca aleatória dos hiperparâmetros
    rnd_search.fit(x_train, y_train)

    return rnd_search.best_params_

def recursive_test(features):
    print(f"{yellow}Horário de Início{reset}: {red}{dt.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}{reset}")
    logging.info(f"Horário de Início: {dt.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    start_time = time.time()

    results_f = pd.DataFrame(columns=['Selected_Features', 'Accuracy', 'Normalize','Intervalo', 'HyperP', 'Model'])
    results_f.to_csv('Resultados/Resultado.csv', index=False)
    feature_combinations = []
    n_total = 7814

    modelos_salvos = pd.DataFrame(columns=['Selected_Features', 'Accuracy', 'Normalize','Intervalo', 'HyperP', 'Model_Name'])
    modelos_salvos.to_csv('Resultados/Modelos_Salvos/Modelos_Salvos.csv', index=False)

    for num_features in range(4, len(features)+1):
        feature_combinations.extend(combinations(features, num_features))

    
    intervals = [Interval.in_1_minute, Interval.in_3_minute, Interval.in_5_minute, Interval.in_15_minute, Interval.in_30_minute, Interval.in_45_minute]
#                ,Interval.in_1_hour, Interval.in_2_hour, Interval.in_3_hour, Interval.in_daily, Interval.in_weekly, Interval.in_monthly]
    
    for intervalo in list(intervals):
        print(f"Treinando {yellow}{intervalo}{reset}")
        logging.info(f"Treinando{intervalo}")

        inter = pd.DataFrame(columns=['Selected_Features', 'Accuracy', 'Normalize','Intervalo', 'HyperP', 'Model'])

        start_int = time.time()
        data = data_inter(intervalo)
        n = 0

        for selected_features in feature_combinations:
            results = []
            Y = data['profit2_t']
            X = data[list(selected_features)]
            x_train, x_test, y_train, y_test = split_custom(X,Y)
            results_f = pd.read_csv('Resultados/Resultado.csv')

            norm_op = True
            if norm_op:
                # fit scaler on your training data
                norm = MinMaxScaler().fit(x_train)

                # transform your training data
                X_train_norm = norm.transform(x_train)

                # transform testing database
                X_test_norm = norm.transform(x_test)

                #best_hparams = best_hp(X_train_norm, y_train)
                best_hparams = {'max_depth': 9, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 64, 'n_jobs': 5}

                ml_rf = train_RF(X_train_norm, y_train, **best_hparams)
                
                # Avaliação do modelo
                predictions = ml_rf.predict(X_test_norm)

                accuracy = accuracy_score(y_test, predictions, normalize = True) * 100.0
                result = {
                    'Selected_Features': ', '.join(selected_features),
                    'Accuracy': accuracy,
                    'Normalize': f"{norm_op}",
                    'Intervalo': f"{intervalo}",
                    'HyperP': best_hparams,
                    'Model': ml_rf
                }

                results.append(result)
                ml_rf = 0

                norm_op = False      

            if norm_op == False:
                # Treinamento do modelo
                #best_hparams = best_hp(x_train, y_train)
                best_hparams = {'max_depth': 9, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 64, 'n_jobs': 5}

                ml_rf = train_RF(x_train, y_train, **best_hparams)
                    
                # Avaliação do modelo
                predictions = ml_rf.predict(x_test)

                accuracy = accuracy_score(y_test, predictions, normalize = True) * 100.0
                result = {
                    'Selected_Features': ', '.join(selected_features),
                    'Accuracy': f"{accuracy:.2f}",
                    'Normalize': f"{norm_op}",
                    'Intervalo': f"{intervalo}",
                    'HyperP': best_hparams,
                    'Model': ml_rf
                }
                results.append(result)
                ml_rf = 0
            n += 1

            if (n%400) == 0 or n == 0:
                logging.info(f"{((n/n_total)*100):.2f}% do intervalo concluido")
                print(f"{((n/n_total)*100):.2f}% do intervalo concluido")

            results = pd.DataFrame(results)
            inter = pd.concat([inter, results]).reset_index(drop=True)
            inter['Accuracy'] = inter['Accuracy'].astype(float)
            inter = inter[inter['Accuracy'] == inter['Accuracy'].max()].head(1)

            results_f = pd.concat([results_f, results]).reset_index(drop=True)
            results_f = results_f.drop('Model', axis=1)
            results_f.to_csv('Resultados/Resultado.csv', index=False)

        results_f = pd.read_csv('Resultados/Resultado.csv')
        modelos_salvos = pd.read_csv('Resultados/Modelos_Salvos/Modelos_Salvos.csv')

        """
        best_model = inter[inter['Accuracy'] == inter['Accuracy'].max()]['Model'].values[0]
        ac_best_model = inter[inter['Accuracy'] == inter['Accuracy'].max()]['Accuracy'].values[0]

        modelos_salvos = pd.concat([modelos_salvos, inter[inter['Accuracy'] == inter['Accuracy'].max()]]).reset_index(drop=True)
        modelos_salvos.loc[modelos_salvos.index[-1], 'Model_Name'] = f'ml_rf_{intervalo}_ac_{ac_best_model:.2f}'
        """

        best_model = inter['Model'].values[0]
        ac_best_model = inter['Accuracy'].values[0]

        modelos_salvos = pd.concat([modelos_salvos, inter]).reset_index(drop=True)
        modelos_salvos.loc[modelos_salvos.index[-1], 'Model_Name'] = f'ml_rf_{intervalo}_ac_{ac_best_model:.2f}'

        modelos_salvos = modelos_salvos.drop('Model', axis=1)

        results_f.to_csv('Resultados/Resultado.csv', index=False)
        modelos_salvos.to_csv('Resultados/Modelos_Salvos/Modelos_Salvos.csv', index=False)


        # Finaliza o processamento do intervalo

        logging.info(f"{intervalo} Finalizado")

        end_int = time.time()
        logging.info(results_f.tail())
        print(results_f.tail())

        print(f"Intervalo {yellow}{intervalo}{reset} finalizado")
        logging.info(f"Intervalo {intervalo} finalizado")

        with open(f'Resultados/Modelos_Salvos/ml_rf_{intervalo}_ac_{ac_best_model:.2f}.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        print(f"Melhor modelo Salvo")
        logging.info(f"Melhor modelo Salvo")       

        print(f"Tempo de execução: {red}{((end_int - start_int)/60):.2f} minutos{reset}")
        logging.info(f"Tempo de execução: {((end_int - start_int)/60):.2f} minutos")
        
        print(f'Finalizao às: {red}{dt.datetime.now().strftime("%H:%M:%S")}{reset}')
        logging.info(f'Finalizao às:{dt.datetime.now().strftime("%H:%M:%S")}')

        print("")
        logging.info("")
    
    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"{yellow}Tempo total de execução{reset}: {red}{execution_time:.2f} minutos{reset}")
    logging.info(f"Tempo total de execução: {execution_time:.2f} minutos")

    return results_f

def pred_ac(model_name, ativo = 'PETR3'):
    intervals = [Interval.in_1_minute, Interval.in_3_minute, Interval.in_5_minute, Interval.in_15_minute, Interval.in_30_minute, Interval.in_45_minute
            ,Interval.in_1_hour, Interval.in_2_hour, Interval.in_3_hour, Interval.in_daily, Interval.in_weekly, Interval.in_monthly]

    # Abre o modelo salvo
    with open(f'Resultados/Modelos_Salvos/{model_name}.pkl', 'rb') as file:
        model = pickle.load(file)

    # Abre as informações dos modelos salvos
    modelos_salvos = pd.read_csv('Resultados/Modelos_Salvos/Modelos_Salvos.csv')
    features = (modelos_salvos.loc[(modelos_salvos['Model_Name'] == model_name), 'Selected_Features'].values[0]).split(', ')
    intervalo = modelos_salvos.loc[(modelos_salvos['Model_Name'] == model_name), 'Intervalo'].values[0]

    # Garante que intervalo não será uma string
    for i in list(intervals):
        if str(i) in str(intervalo):
            intervalo = i

    # Processa os dados para o intervalo requerido
    data = data_inter(intervalo, 500, ativo)

    # Calcula a ultima tendência com base nas features requeridas
    trend = model.predict(data[features].iloc[-1].values.reshape(1, -1))
    return trend[0]
