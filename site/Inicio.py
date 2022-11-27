import streamlit as st
import translators as tss
from lstm_model_functions import load_lstm_model, pre_process_tweet
from bayes_model_functions import load_bayes_model, bayes_predict

st.title('Racismo Amarelo')
st.header('Uma análise sobre discursos de ódio contemporâneos através de aprendizagem de máquina')

st.subheader('')
st.subheader('Motivação')
st.write('O racismo ainda é um tema muito relevante atualmente, presente não apenas no contexto de povos com descendência africana, mas também contra populações provenientes da Ásia e de cultura indígena. Esse tipo de discurso de ódio direcionados à população asiática se tornou mais visível por meio do crescente número de comentários xenofóbicos em redes sociais durante os Jogos Olímpicos de Tóquio 2021 e o início do período da pandemia da Covid-19.'
'O trabalho aqui apresentado é um estudo em processamento de linguagem natural e machine learning, com o intuito de identificar comentários racistas contra asiáticos e seus descendentes na rede social Twitter. Portanto, procura-se por analisar sentimentos presentes nos comentários, assim como elementos de microagresão, de forma que o contexto e a forma como certas palavras foram usadas nas frases em estudo indiquem a presença ou não de injúrias raciais.'
'Com a esperança de se trazer uma conscientização, ou ao menos um meio para que a população consiga se educar, os modelos treinados e especificados ao longo do trabalho foram integrados a uma aplicação em forma de bot do Twitter, por meio do qual usuários tem a possibilidade de chamá-lo e verificar se uma mensagem possui alguma conotação racista contra asiáticos.'
'Vale destacar que o trabalho possui um enfoque ao racismo contra povos e descendentes da Ásia, uma vez que pouco se encontrou trabalhos com essa temática. Ademais, dado a sensibilidade do tema, achou-se sensato concentrar-se numa etnia apenas.')

st.subheader('O modelo')

modelo_selecionado = st.selectbox(label='Escolha o tipo de modelo a ser utilizado', options=('Bayes', 'LSTM'))

frase_analisada = st.text_input('Digite uma sentença para o modelo verificar se é racista ou não:',
                                help='A sentença deve ser escrita em inglês')



if modelo_selecionado == 'Bayes':
    if st.button('Verificar'):
        model = load_bayes_model()
        if model:
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
            frase_processada = pre_process_tweet(frase_analisada)
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
