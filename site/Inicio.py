import streamlit as st
import translators as tss
from lstm_functions import load_lstm_model, lstm_pre_process_tweet
from bayes_functions import load_bayes_model, bayes_predict
from cnn_functions import load_cnn_model, cnn_pre_process_tweet
from lstm_emb_functions import load_lstm_emb_model, lstm_emb_pre_process_tweet
from cnn_emb_functions import load_cnn_emb_model, cnn_emb_pre_process_tweet
from lstm_cnn_functions import load_lstm_cnn_model, lstm_cnn_pre_process_tweet
from lstm_bi_cnn_functions import load_lstm_bi_cnn_model, lstm_bi_cnn_pre_process_tweet
import numpy as np
import tensorflow as tf

st.title('Racismo Amarelo')
st.header('Uma análise sobre discursos de ódio contemporâneos através de aprendizagem de máquina')

st.subheader('')
st.subheader('Motivação')
st.write('O racismo ainda é um tema muito relevante atualmente, presente não apenas no contexto de povos com descendência africana, mas também contra populações provenientes da Ásia e de cultura indígena. Esse tipo de discurso de ódio direcionados à população asiática se tornou mais visível por meio do crescente número de comentários xenofóbicos em redes sociais durante os Jogos Olímpicos de Tóquio 2021 e o início do período da pandemia da Covid-19.'
'O trabalho aqui apresentado é um estudo em processamento de linguagem natural e machine learning, com o intuito de identificar comentários racistas contra asiáticos e seus descendentes na rede social Twitter. Portanto, procura-se por analisar sentimentos presentes nos comentários, assim como elementos de microagresão, de forma que o contexto e a forma como certas palavras foram usadas nas frases em estudo indiquem a presença ou não de injúrias raciais.'
'Com a esperança de se trazer uma conscientização, ou ao menos um meio para que a população consiga se educar, os modelos treinados e especificados ao longo do trabalho foram integrados a uma aplicação em forma de bot do Twitter, por meio do qual usuários tem a possibilidade de chamá-lo e verificar se uma mensagem possui alguma conotação racista contra asiáticos.'
'Vale destacar que o trabalho possui um enfoque ao racismo contra povos e descendentes da Ásia, uma vez que pouco se encontrou trabalhos com essa temática. Ademais, dado a sensibilidade do tema, achou-se sensato concentrar-se numa etnia apenas.')

st.subheader('O modelo')

modelo_selecionado = st.selectbox(label='Escolha o tipo de modelo a ser utilizado', options=('Bayes',
                                                                                             'LSTM',
                                                                                             'LSTM Embedded',
                                                                                             'CNN',
                                                                                             'CNN Embedded', 
                                                                                             'LSTM + CNN',
                                                                                             'LSTM Bidirecional + CNN'))

frase_analisada = st.text_input('Digite uma sentença para o modelo verificar se é racista ou não:',
                                help='A sentença deve ser escrita em inglês')



if modelo_selecionado == 'Bayes':
    if st.button('Verificar'):
        model = load_bayes_model()
        if model:
            #with st.spinner(text='Processando os dados'):
            prediction = bayes_predict(frase_analisada, model)
            if prediction < 0.5:
                st.error('A frase analisada foi classificada como racista')
            if prediction >= 0.5:
                st.success('A frase analisada foi classificada como não racista')
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo avalia de forma binária, se é ou não racista.')
    st.write('Não indica se a frase apresenta uma tendência racista.')

elif modelo_selecionado == 'LSTM':
    if st.button('Verificar'):
        model = load_lstm_model()
        if model:
            frase_processada = lstm_pre_process_tweet(frase_analisada)
            prediction = model(frase_processada)
            prediction = ((prediction[: ,0])[0]).item()
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(prediction))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(prediction))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')

elif modelo_selecionado == 'LSTM Embedded':
    if st.button('Verificar'):
        model = load_lstm_emb_model()
        if model:
            frase_processada = lstm_emb_pre_process_tweet(frase_analisada)
            prediction = model(frase_processada)
            prediction = ((prediction[: ,0])[0]).item()
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(prediction))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(prediction))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')

elif modelo_selecionado == 'CNN':
    if st.button('Verificar'):
        model = load_cnn_model()
        if model:
            frase_processada = cnn_pre_process_tweet(frase_analisada)
            prediction = model(frase_processada)[0][0]
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(float(prediction)))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(float(prediction)))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')

elif modelo_selecionado == 'CNN Embedded':
    if st.button('Verificar'):
        model = load_cnn_emb_model()
        if model:
            frase_processada = cnn_emb_pre_process_tweet(frase_analisada)
            prediction = model(frase_processada)
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(float(prediction)))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(float(prediction)))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')

elif modelo_selecionado == 'LSTM + CNN':
    if st.button('Verificar'):
        model = load_lstm_cnn_model()
        if model:
            frase_processada = lstm_cnn_pre_process_tweet(frase_analisada)
            prediction = model(frase_processada)
            prediction = ((prediction[: ,0])[0]).item()
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(prediction))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(prediction))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')

elif modelo_selecionado == 'LSTM Bidirecional + CNN':
    if st.button('Verificar'):
        model = load_lstm_bi_cnn_model()
        if model:
            frase_processada = lstm_bi_cnn_pre_process_tweet(frase_analisada)
            st.info(frase_processada)
            prediction = model(frase_processada)
            prediction = ((prediction[: ,0])[0]).item()
            if prediction < 0.5:
                st.error('A frase analisada tende a ser racista, com score de: ' + str(prediction))
            if prediction >= 0.5:
                st.success('A frase analisada tende a ser não racista, com score de: ' + str(prediction))
        else:
            st.error("Tente novamente mais tarde")
    st.write('O modelo apresenta um score com valores entre 0 e 1.')
    st.write('A frase apresenta-se com maior tendência racista conforme o  valor do score se aproxime de 0.')


else:
    st.error('Nenhum modelo foi especificado')
  

st.write('')
st.write('Se sentir alguma dificuldade em lembrar-se de algum termo em inglês, use o tradutor abaixo!')

input_text = st.text_input(label='coloque um texto para traduzir', placeholder='Coloque um texto para traduzir', label_visibility='hidden')

if st.button('Traduzir'):    
    if input_text != "":
        result = tss.google(input_text, from_language='pt', to_language='en')
        st.success(result)
    else:
        st.error("Nenhum texto foi colocado para ser traduzido")
