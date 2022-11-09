import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN
from tensorflow import keras


def get_ac(ativo, start, end, intervalo = '1d'):
  df = yf.download(tickers=ativo, interval=intervalo, start=start, end=end)
  return df

def call():
    name = input("Nome da ação: ")
    start = input("Início: ")
    end = input("Fim: ")
    return name, start, end

def df_Nan(df):   
    df = df.fillna(method='ffill', inplace=True)
    df = df.dropna()
    return df

# Ajuste de data
def parse_date(df):
  df['Date'] = pd.to_datetime(df.index, format = '%Y%m%d')
  return df

# Considerando apenas o preço de fechamento
# Considerando apenas o preço de fechamento
days_time_step = 15
def ttv(df): # ttv = Teste, Treino e Validação
    df_features = df[['Date', 'Close']]
    df_features.drop('Date', axis = 1, inplace = True)

    training_size = int(len(df_features) * 0.8)
    test_size = len(df_features) - training_size

    # Normalização
    scaler = StandardScaler()
    prices_sc = scaler.fit_transform(df_features)

    train_data = prices_sc[:training_size]
    test_data = prices_sc[training_size: training_size + test_size]
    val_data = prices_sc[training_size - days_time_step:]

    #return train_data, test_data

    return train_data, test_data, val_data, scaler

def create_df(df, steps=1):
  dataX, dataY = [], []
  for i in range(len(df)-steps-1):
    a = df[i:(i+steps), 0]
    dataX.append(a)
    dataY.append(df[i + steps, 0])
  return np.array(dataX), np.array(dataY)

def Processing(train_data, test_data, val_data, days_time_step):
    #treino
    X_train, Y_train = create_df(train_data, days_time_step)
    X_test, Y_test = create_df(test_data, days_time_step)
    X_val, Y_val = create_df(val_data, days_time_step)

    #converter tudo pra matriz numpy

    X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val= X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
   
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def lstm(days_time_step):
    model = Sequential()
    model.add(LSTM(100, return_sequences=False, input_shape=(days_time_step, 1)))
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.compile(loss='mse', optimizer='adam')

    return model

def validation(model, X_train, Y_train, X_val, Y_val):
    val = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = 30, batch_size = 32, verbose = 2)
    return val

def plot_val(validation):
    plt.plot(validation.history["loss"], label='loss')
    plt.plot(validation.history["val_loss"], label='val_loss')
    plt.legend()

def pred(model, X_test, test_data, scaler):
    predict = model.predict(X_test)

    #transformação inversa do normalizador, pra que possamos plotar no gráfico os resultados
    predict = scaler.inverse_transform(predict)
    real = scaler.inverse_transform(test_data)
    return predict, real

def plot_pred(predict, real, df_t):
    plt.figure(figsize = (18,9))
    plt.plot(real, color = 'green', label = 'real')
    plt.plot(predict, color = 'red', label = 'previsão')
    #plt.xticks(range(0, len(real), 50), df_t['Date'], rotation=45)
    plt.xlabel('Datas', fontsize=18)
    plt.ylabel('Preço Médio', fontsize=18)
    plt.title("Projeção de Preço AAPL", fontsize=30)
    plt.legend()
    plt.show()

def last_days(test_data, days_time_step):
    lenght_test = len(test_data)
    days_input_steps = lenght_test - days_time_step

    input_steps = test_data[days_input_steps:]
    input_steps = np.array(input_steps).reshape(1,-1)

    # Transformar em lista
    list_output_steps = list(input_steps)
    list_output_steps = list_output_steps[0].tolist()

    return list_output_steps, input_steps

def forecast(f_days, input_steps, list_output_steps, days_time_step, pred, model):
    pred_output = []
    i = 0
    n_future = f_days
    while (i < n_future):

      if(len(list_output_steps) > days_time_step):
        
        input_steps = np.array(list_output_steps[1:])
        print('{} dia. valores de entrada -> {}'.format(i, input_steps))
        input_steps = input_steps.reshape(1, -1)
        input_steps = input_steps.reshape((1, days_time_step, 1))
        # print(imput_steps)
        pred = model.predict(input_steps, verbose = 0)
        print('{} dia. valores previsto -> {}'.format(i, pred))
        list_output_steps.extend(pred[0].tolist())
        list_output_steps = list_output_steps[1:]
        # print (list_output_steps)
        pred_output.extend(pred.tolist())
        i = i + 1
      else:
        input_steps = input_steps.reshape((1, days_time_step, 1))
        pred = model.predict(input_steps, verbose = 0)
        print(pred[0])
        list_output_steps.extend(pred[0].tolist())
        print(len(list_output_steps))
        pred_output.extend(pred.tolist())
        i = i + 1

        print(pred_output)

    return pred_output

def tf_forecast(pred_output, scaler):
    # Transforma a saída
    prev = scaler.inverse_transform(pred_output)
    prev = np.array(prev).reshape(1, -1)
    list_output_prev = list(prev)
    list_output_prev = prev[0].tolist()

    print(list_output_prev)

    return list_output_prev

# Formata a saida e cria DataFrame
def format_forecast(df, list_output_prev):
    dates = pd.to_datetime(df['Date'])
    predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods = 10, freq = 'b').tolist()
    predict_dates

    forecast_dates = []
    for i in predict_dates:
      forecast_dates.append(i.date())

    df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Close': list_output_prev})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    df_forecast = df_forecast.set_index(pd.DatetimeIndex(df_forecast['Date'].values))
    df_forecast.drop('Date', axis = 1, inplace= True)

    print(df_forecast)

    return df_forecast

def plot_forecast(df, df_forecast):
    # Plotar gráfico
    ma100 = df.rolling(10).mean()
    plt.figure( figsize = (16,8))
    plt.plot(df['Close'])
    plt.plot(df_forecast['Close'],'r')
    plt.legend(['Preço fechamento', 'Preço Previsto'])
    #plt.plot(ma100, 'r')
    plt.show()

# Função Final
def main():
    days_time_step = 15
    f_days = 10
    name, start, end = call()
    df = get_ac(name, start, end)
    df = parse_date(df)

    train_data, test_data, val_data,scaler = ttv(df)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = Processing(train_data, test_data, val_data, days_time_step)

    model = lstm(days_time_step)

    val = validation(model, X_train, Y_train, X_val, Y_val)

    plot_val(val)

    # Formatação

    predict, real = pred(model, X_test, test_data, scaler)

    plot_pred(predict, real, df)

    # Previsão Próximos dias

    list_output_steps, input_steps = last_days(test_data, days_time_step)

    pred_output = forecast(f_days, input_steps, list_output_steps, days_time_step, pred, model)

    list_output_prev = tf_forecast(pred_output, scaler)

    df_forecast = format_forecast(df, list_output_prev)

    plot_forecast(df, df_forecast)

def teste():
    days_time_step = 15
    f_days = 10
    name = 'aapl'
    start= '2013-09-15'
    end = '2021-11-15'

    df = get_ac(name, start, end)
    df = parse_date(df)

    train_data, test_data, val_data,scaler = ttv(df)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = Processing(train_data, test_data, val_data, days_time_step)

    model = lstm(days_time_step)

    val = validation(model, X_train, Y_train, X_val, Y_val)

    plot_val(val)

    # Formatação

    predict, real = pred(model, X_test, test_data, scaler)

    plot_pred(predict, real, df)

    # Previsão Próximos dias

    list_output_steps, input_steps = last_days(test_data, days_time_step)

    pred_output = forecast(f_days, input_steps, list_output_steps, days_time_step, pred, model)

    list_output_prev = tf_forecast(pred_output, scaler)

    df_forecast = format_forecast(df, list_output_prev)

    plot_forecast(df, df_forecast)

#main()

teste()