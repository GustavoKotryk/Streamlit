import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Previsão de Progressão do Diabetes")
st.title("Previsão de Progressão do Diabetes")
st.markdown("Informe os valores das características clínicas do paciente para prever a **progressão da doença (índice)**.")

model = joblib.load('modelo.pkl')

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Idade (age)", -0.1, 0.1, 0.0, format="%.3f")
    sex = st.slider("Sexo (sex)", -0.1, 0.1, 0.0, format="%.3f")
    bmi = st.slider("Índice de Massa Corporal (bmi)", -0.1, 0.1, 0.0, format="%.3f")
    bp = st.slider("Pressão arterial média (bp)", -0.1, 0.1, 0.0, format="%.3f")
    s1 = st.slider("Soro lipídico (s1)", -0.1, 0.1, 0.0, format="%.3f")

with col2:
    s2 = st.slider("Soro lipídico (s2)", -0.1, 0.1, 0.0, format="%.3f")
    s3 = st.slider("Soro lipídico (s3)", -0.1, 0.1, 0.0, format="%.3f")
    s4 = st.slider("Soro lipídico (s4)", -0.1, 0.1, 0.0, format="%.3f")
    s5 = st.slider("Soro lipídico (s5)", -0.1, 0.1, 0.0, format="%.3f")
    s6 = st.slider("Glicose (s6)", -0.1, 0.1, 0.0, format="%.3f")

if st.button("Prever progressão"):
    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    pred = model.predict(input_data)[0]
    st.success(f"Progressão prevista da doença: **{pred:.1f}**")
    st.metric("Valor previsto", f"{pred:.1f}", delta=None)