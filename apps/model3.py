import numpy as np
import pandas as pd
import pandas_datareader as data2
import plotly.graph_objects as go
import plotly.express as px
from sklearn import neighbors
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score 
import streamlit as st

def app():
    st.title('Model - KNN')

    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'GC=F')

    df = data2.DataReader(user_input, 'yahoo', start, end)

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
    
    
    

    # Cree un nuevo marco de datos con solo la columna 'Close'
    data = df.filter(['Close'])

    # Convierte el marco de datos en una matriz numpy
    dataset = data.values

    # Obtenga el número de filas para entrenar el modelo
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # Escalando la data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    training_size=int(len(scaled_data)*0.65)
    test_size=len(scaled_data)-training_size
    train_data,test_data=scaled_data[0:training_size,:],scaled_data[training_size:len(scaled_data),:1]
    
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    K = time_step
    neighbor = neighbors.KNeighborsRegressor(n_neighbors = K)
    neighbor.fit(X_train, y_train)

    train_predict=neighbor.predict(X_train)
    test_predict=neighbor.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


    st.title('Evaluación del Modelo KNN')
    ## Métricas
    MAE= mean_absolute_error(original_ytrain,train_predict)
    MSE= mean_squared_error(original_ytrain,train_predict)
    RMSE= math.sqrt(mean_squared_error(original_ytrain,train_predict))

    MAET= mean_absolute_error(original_ytest,test_predict)
    MSET= mean_squared_error(original_ytest,test_predict)
    RMSET= math.sqrt(mean_squared_error(original_ytest,test_predict))

    RS = explained_variance_score(original_ytrain, train_predict)
    RST = explained_variance_score(original_ytest, test_predict)
    
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','Mean Absolute Error TestData','Mean Squared Error TestData','Root Mean Squared Error TestData'],
        'valor': [MAE, MSE, RMSE, MAET, MSET, RMSET]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo FeedForward Neural Network",
        color="metrica"
    )
    st.plotly_chart(fig)


    score = {
        'metrica' : ['Regression Score', 'Regression Score Trained'],
        'valor': [RS,RST]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas del score') 
    fig2 = px.bar(        
        score,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo FeedForward Neural Network",
        color="metrica"
    )
    st.plotly_chart(fig2)

