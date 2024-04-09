# importação das bibliotecas necessárias. deverá instalar, usando pip, as que não estiverem disponíveis

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# lê o banco de dados em csv
df = pd.read_csv("nano2.csv")

# importa ARIMA (modelo autoregressivo de médias móveis)
from statsmodels.tsa.arima.model import ARIMA

# chama o modelo ARIMA e imprime sumário (quadro com valores)
arima_results = ARIMA(df["Y"], df[["T","D","P"]], order=(1,0,0)).fit()
print(arima_results.summary())

# define último mês sem intervenção e último mês com intervenção
start = 10
end = 20

# calcula predições a partir dos resultados do modelo ARIMA
predictions = arima_results.get_prediction(0, end-1)
summary = predictions.summary_frame(alpha=0.05)

# elabora modelo ARIMA durante intervenção
arima_cf = ARIMA(df["Y"][:start], df["T"][:start], order=(1,0,0)).fit()

# Modela as predições das médias 
y_pred = predictions.predicted_mean

# Calcula a média do contrafactual (o que aconteceria se não houvesse intervenção), com IC 95%
y_cf = arima_cf.get_forecast(10, exog=df["T"][start:]).summary_frame(alpha=0.05)

# Seleciona o tipo de gráfico e o seu tamanho
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16,10))

# Plota os dados da incidência de VAP
ax.scatter(df["T"], df["Y"], facecolors='none', edgecolors='steelblue', label="VAP incidence", linewidths=2)

# Plota a predição da média de VAP
ax.plot(df["T"][:start], y_pred[:start], 'b-', label="model prediction")
ax.plot(df["T"][start:], y_pred[start:], 'b-')

# Plota a média do contrafactual de VAP com IC 95%
ax.plot(df["T"][start:], y_cf["mean"], 'k.', label="counterfactual")
ax.fill_between(df["T"][start:], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="counterfactual 95% CI");


# Plota a linha do momento da intervenção
ax.axvline(x = 10.5, color = 'r', label = 'nanotechnology')

ax.legend(loc='best')
plt.ylim([-1, 12])
plt.xlabel("Months")
plt.ylabel("VAP rate (per 1000)");