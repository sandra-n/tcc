import streamlit as st
import pandas as pd
import numpy as np
import translators as tss
import random

st.title('Racismo Amarelo')
st.header('Uma análise sobre discursos de ódio contemporâneos através de aprendizagem de máquina')

st.subheader('')
st.subheader('Motivação')
st.write('O racismo ainda é um tema muito relevante atualmente, presente não apenas no contexto de povos com descendência africana, mas também contra populações provenientes da Ásia e de cultura indígena. Esse tipo de discurso de ódio direcionados à população asiática se tornou mais visível por meio do crescente número de comentários xenofóbicos em redes sociais durante os Jogos Olímpicos de Tóquio 2021 e o início do período da pandemia da Covid-19.'
'O trabalho aqui apresentado é um estudo em processamento de linguagem natural e machine learning, com o intuito de identificar comentários racistas contra asiáticos e seus descendentes na rede social Twitter. Portanto, procura-se por analisar sentimentos presentes nos comentários, assim como elementos de microagresão, de forma que o contexto e a forma como certas palavras foram usadas nas frases em estudo indiquem a presença ou não de injúrias raciais.'
'Com a esperança de se trazer uma conscientização, ou ao menos um meio para que a população consiga se educar, os modelos treinados e especificados ao longo do trabalho foram integrados a uma aplicação em forma de bot do Twitter, por meio do qual usuários tem a possibilidade de chamá-lo e verificar se uma mensagem possui alguma conotação racista contra asiáticos.'
'Vale destacar que o trabalho possui um enfoque ao racismo contra povos e descendentes da Ásia, uma vez que pouco se encontrou trabalhos com essa temática. Ademais, dado a sensibilidade do tema, achou-se sensato concentrar-se numa etnia apenas.')

st.subheader('O modelo')
st.text_input('Digite uma sentença para o modelo verificar se é racista ou não:', 
            help='A sentença deve ser escrita em inglês')
if st.button('Verificar'):
    random.seed()
    valor = random.random()
    if(valor > 0.5):
        st.success(valor)
    else:
        st.error(valor)

st.write('O modelo apresenta um score com valores entre 0 e 1.')
st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')
st.write('')
st.write('Se sentir alguma dificuldade em lembrar-se de algum termo em inglês, use o tradutor abaixo!')

input_text = st.text_input(label='texto a ser traduzido', label_visibility='hidden')
#Translates user input and creates text to speech audio
 
if st.button('Traduzir'):    
    result = tss.google(input_text, from_language='pt', to_language='en')
    st.success(result)