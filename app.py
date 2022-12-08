import streamlit as st
from multiapp import MultiApp
from apps import home, model1, model2, model3, model4 # import your app modules here

app = MultiApp()

st.markdown("""
# Multi-Page App

This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo SVR", model1.app)
app.add_app("Modelo LR", model2.app)
app.add_app("Modelo GRU", model3.app)
app.add_app("Modelo ARIMA", model4.app)
# The main app
app.run()
