import streamlit as st
import pandas as pd
import numpy as np

st.title("Studium Dashboard: Hello World")
st.write("Hier ist ein kleiner Test mit Pandas:")

df = pd.DataFrame(
    np.random.randn(10, 2),
    columns=['Note A', 'Note B']
)

st.line_chart(df)
st.dataframe(df)