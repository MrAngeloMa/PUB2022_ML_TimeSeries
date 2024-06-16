# Uso de técnicas de Machine Learning para previsão de Séries Temporais Financeiras *(Guia)*

Esse guia tem como objetivo organizar os códigos até então desenvolvidos para melhor identifica-los, saber suas utilidades e características.

## [1_Séries_Temporais_Indicadores](https://github.com/MrAngeloMa/PUB2022_ML_TimeSeries/blob/main/1_S%C3%A9ries_Temporais_Indicadores.ipynb)
Foi o primeiro código desenvolvido, nele encontram-se:

- Teoria de alguns indicadores (MACD, LTA, LTB, Nuvem de Ichimoku, Suporte e Resistência, Bandas de Bollinger, RSI);
- Cálculos de indicadores voltado para o estudo do mesmo;
- Visualização de dados;
- Teste de um modelo ARIMA.

## [2_RN](https://github.com/MrAngeloMa/PUB2022_ML_TimeSeries/blob/main/2_RN.ipynb)
Esse Colab teve como objetivo o estudo e o teste de algorítmos de Machine Learning, nele encontram-se:

- Um teste de comparação entre Regressão Linear, Rede Neural e RN com ajuste de hiperparâmetro para verificar qual era possivelmente melhor para a tarefa desejada;
- Um pouco sobre Engenharia de Dados para ativos financeiros;
- Teste de uma Rede Neural LSTM (Não obteve resultado tão desejável)

Todos os códigos apresentados utilizam apenas o fechamento como parâmetro e tentam prever o preço, algo que se mostrou inviável.

## [3_PUB_MachineLearning]
Esse código foi melhor desenvolvido e organizado, está presente nele uma grande quantidade de materiais de apoio para a execução do projeto como um todo. Ele foi desenvolvido como uma análise final a respeito de qual atitude tomar quanto ao desenvolvimento do algoritmo desejado.

#### Características:

- Aqui os códigos são completamente automatizados e organizados em funções;
- Utilizou-se uma Rede Neural LSTM através da biblioteca Keras;
- Chegou-se a conclusão que tentar prever o fechamento era invíavel e não apresentaria muita utilidade, mesmo que funcionasse. Sendo assim, tomou-se a alternativa de calcular tendência;
- No final do código são apresentadas alternativas que poderiam ser implementadas para melhorar a previsão do algorítmo.

## [4_Análise_Técnica_Machine_Learning](https://github.com/MrAngeloMa/PUB2022_ML_TimeSeries/blob/main/4_An%C3%A1lise_T%C3%A9cnica_Machine_Learning.ipynb)
Esse código consulta a internet a respeito das 100 maiores empresas atuais com ações no mercado. Sendo assim, ele faz uma análise geral de 100 ações simultaneamente e retorna as melhores opções. Isso seria útil para a criação de portifólios por uma pessoa que não planeja analisar todas as ações individualmente.

#### Característica:

- Features: Dados básicos + Indicadores técnicos;
- Random Forest;
- Coleta dados da internet;
- Retorna as melhores ações;
- Utiliza Clusters para dividir as ações em grupos de comportamento comuns;
- Prevê tendência;

Nesse código, surgiu a possibilidade do programa final ter 2 opções:

- Análise Geral de ações (Como apresentado);
- Análise Específica.

O programa forneceria as duas opções ao cliente, ele poderia formar um portfólio com base nas ações indicadas na Análise Geral ou executar uma Análise e Previsão de um ativo em específico da sua escolha. Para a ultima parte os códigos anteriores não funcionariam, pois não preveem tendência e nem usam indicadores técnicos para treinar o algorítmo. Sendo assim, o próximo código é voltado para o Treinamento de um Algorítmo que utilize esses Indicadores e que se aplique a uma única ação.
