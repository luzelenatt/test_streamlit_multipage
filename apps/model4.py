import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader as data
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px

def app():
    st.title('Model - Decision Tree')

    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'GC=F')

    df = data.DataReader(user_input, 'yahoo', start, end)

    # Describiendo los datos

    st.subheader('Datos del 2004 al 2022') 
    st.write(df.describe())
    
    # Candlestick chart
    st.subheader('Gráfico Financiero') 
    candlestick = go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close']
                            )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        width=800, height=600,
        title=user_input,
        yaxis_title='Precio'
    )
    
    st.plotly_chart(fig)
    
    
    

    # Añadiendo indicadores para el modelo
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    
    
    # Modelo SVC
    
    ## Variables predictoras
    X = df[['Open-Close', 'High-Low']]
    ## Variable objetivo
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    ## División data de entrenamiento y prueba
    split_percentage = 0.8
    split = int(split_percentage*len(df))
    ## Entrenando el dataset
    X_train = X[:split]
    y_train = y[:split]
    ## Testeando el dataset
    X_test = X[split:]
    y_test = y[split:]
    ## Creación del modelo
    cls = DecisionTreeRegressor().fit(X_train, y_train)
    ## Predicción del test
    y_pred = cls.predict(X_test)
    
    
    # Señal de predicción 
    
    df['Predicted_Signal'] = cls.predict(X)
    ## Añadiendo columna condicional
    conditionlist = [
    (df['Predicted_Signal'] == 1) ,
    (df['Predicted_Signal'] == 0)]
    choicelist = ['Comprar','Vender']
    df['Decision'] = np.select(conditionlist, choicelist)
    st.subheader('Predicción de señal de compra o venta') 
    st.write(df)    
    
    
    
    # Estrategia de Implementación
    
    # Cálculo de las devoluciones diarias
    df['Return'] = df.Close.pct_change()
    # Cálculo de los rendimientos de la estrategia
    df['Strategy_Return'] = df.Return*df.Predicted_Signal.shift(1)
    # Cálculo de los rendimientos acumulativos
    df['Cum_Ret'] = df['Return'].cumsum()
    # Cálculo de los rendimientos acumulativos de la estrategia
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    # Retornos de la estrategia de trama vs rendimientos originales
    st.subheader('Retornos de la estrategia de trama vs. Rendimientos originales') 
    fig = px.line(df,y=['Cum_Ret', 'Cum_Strategy'])
    st.plotly_chart(fig)
    
    
    
   ##########PLANTILLA####################
    # Evaluación del modelo
    
    st.title('Evaluación del Modelo Decision Tree')
    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo Decision Tree",
        color="metrica"
    )
    st.plotly_chart(fig)
