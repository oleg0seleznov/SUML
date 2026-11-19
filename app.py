import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()

filename = 'model.h5'
model = pickle.load(open(filename, 'rb'))

sex_d = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d = {0: 'Pierwsza', 1: 'Druga', 2: 'Trzecia'}
embarked_d = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}

def main():
    st.set_page_config(page_title="Czy przebyłbyś katastrofę?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image('https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG')
    

if main() == "__main__":
    main()