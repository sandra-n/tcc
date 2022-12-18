import streamlit as st
import translators as tss
import random
from PIL import Image
from lstm_functions import load_lstm_model, lstm_pre_process_tweet
from bayes_functions import load_bayes_model, bayes_predict
from cnn_functions import load_cnn_model, cnn_pre_process_tweet
from lstm_emb_functions import load_lstm_emb_model, lstm_emb_pre_process_tweet
from cnn_emb_functions import load_cnn_emb_model, cnn_emb_pre_process_tweet
from lstm_cnn_functions import load_lstm_cnn_model, lstm_cnn_pre_process_tweet
from lstm_bi_cnn_functions import load_lstm_bi_cnn_model, lstm_bi_cnn_pre_process_tweet
from lstm_bi_cnn_emb_functions import load_lstm_bi_cnn_emb_model, lstm_bi_cnn_emb_pre_process_tweet
from lstm_cnn_emb_functions import load_lstm_cnn_emb_model, lstm_cnn_emb_pre_process_tweet
from lstm_bi_emb_functions import load_lstm_bi_emb_model, lstm_bi_emb_pre_process_tweet
from treat_tweets import treating_tweet
import os
from pathlib import Path

path = os.path.dirname(__file__)
path = Path(path)
parent_dir = str(path.parent.absolute())

with open(parent_dir + '/.streamlit/style.css') as f:
    st.markdown(
        f'<style>{f.read()}</style>',
        unsafe_allow_html=True,
    )


image = Image.open(str(path)+'/images/site-image.jpeg')
st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

st.title('Racismo Amarelo')
st.header('Uma análise sobre discursos de ódio contemporâneos através de aprendizagem de máquina')

st.subheader('')
st.subheader('Motivação')
st.write('O racismo ainda é um tema muito relevante atualmente, presente não apenas no contexto de povos com descendência africana, mas também contra populações provenientes da Ásia e de cultura indígena. Esse tipo de discurso de ódio direcionados à população asiática se tornou mais visível por meio do crescente número de comentários xenofóbicos em redes sociais durante os Jogos Olímpicos de Tóquio 2021 e o início do período da pandemia da Covid-19. \n\n'
'O trabalho aqui apresentado é um estudo em processamento de linguagem natural e machine learning, com o intuito de identificar comentários racistas contra asiáticos e seus descendentes na rede social Twitter. \n\n'
'Com a esperança de se trazer uma conscientização, ou ao menos um meio para que a população consiga se educar, os modelos treinados e especificados ao longo do trabalho foram integrados a uma aplicação em forma de bot do Twitter, por meio do qual usuários tem a possibilidade de chamá-lo e verificar se uma mensagem possui alguma conotação racista contra asiáticos.')

st.subheader('O modelo')

modelo_selecionado = st.selectbox(label='Escolha o tipo de modelo a ser utilizado', options=('Bayes',
                                                                                             'LSTM',
                                                                                             'LSTM com Embedding',
                                                                                             'CNN',
                                                                                             'CNN com Embedding', 
                                                                                             'LSTM + CNN',
                                                                                             'LSTM + CNN com Embedding',
                                                                                             'LSTM Bidirecional com Embedding',
                                                                                             'LSTM Bidirecional + CNN', 
                                                                                             'LSTM Bidirecional + CNN com Embedding'))

frase_input = st.text_input('Digite uma sentença para o modelo verificar se é racista ou não:',
                                help='A sentença deve ser escrita em inglês')
frase_analisada = treating_tweet(frase_input)


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

elif modelo_selecionado == 'LSTM com Embedding':
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

elif modelo_selecionado == 'CNN com Embedding':
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

elif modelo_selecionado == 'LSTM Bidirecional com Embedding':
    if st.button('Verificar'):
        model = load_lstm_bi_emb_model()
        if model:
            frase_processada = lstm_bi_emb_pre_process_tweet(frase_analisada)
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

elif modelo_selecionado == 'LSTM Bidirecional + CNN com Embedding':
    if st.button('Verificar'):
        model = load_lstm_bi_cnn_emb_model()
        if model:
            frase_processada = lstm_bi_cnn_emb_pre_process_tweet(frase_analisada)
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

elif modelo_selecionado == 'LSTM + CNN com Embedding':
    if st.button('Verificar'):
        model = load_lstm_cnn_emb_model()
        if model:
            frase_processada = lstm_cnn_emb_pre_process_tweet(frase_analisada)
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
