import streamlit as st
from PIL import Image


with open('.streamlit/style.css') as f:
    st.markdown(
        f'<style>{f.read()}</style>',
        unsafe_allow_html=True,
    )

st.title('Sobre o projeto')

st.subheader('')
st.subheader('Objetivo')
st.write('O objetivo do projeto de formatura é desenvolver um modelo de Processamento de Linguagem Natural capaz de identificar a presença de comentários racistas em redes sociais como Twitter.  \n'
' O desenvolvimento de uma conta automatizada (bot) em tal plataforma, capaz de informar os usuários sobre a presença de textos discriminatórios e promover a conscientização a cerca do tema, respeitando o espaço de cada um e as regras do Twitter, também é uma das metas do grupo.  \n\n'
' Como futuros engenheiros, nosso papel é de encontrar soluções a problemas e desafios do cotidiano. Com esse projeto, buscamos levantar a discussão para a questão do racismo, especialmente contra asiáticos, que é tão presente no mundo atual, porém pouco abordada nas mídias.  \n')

st.subheader('Arquitetura')
st.write('Os dados utilizados para treinamento, validação e teste de modelos de Machine Learning e Deep Learning desenvolvidos foram obtidos de duas fontes: o dataset público Covid Hate, criado por pesquisadores do Instituto de Tecnologia da Georgia e da Virginia Tech, e pela própria API do Twitter.  \n\n'
' O Covid Hate consiste num dataset com mais de 200 milhões de tweets coletados durante a pandemia de Covid-19, classificados como "racistas", "antirracistas" e "neutros". Já os dados coletados via API contém dados referentes ao autor e data da postagem, mas sem classificação de racismo.  \n'
' Esses tweets foram coletados entre maio e julho de 2022 e, para serem incluídos no dataset do projeto, precisaram ser rotulados. A rotulação foi feita baseada na rotulação manual de uma amostra de todos os tweets seguida da aplicação do modelo BERT pré-treinado. Com ele, os outros tweets puderam ser classificados.  \n\n'
' Ao final, a quantidade total de tweets utilizáveis para o estudo foi 1459393. Todos os dados, independentemente da fonte da qual vieram, passaram pelos mesmos processos de limpeza e tratamento, antes de serem inseridos nos modelos.  \n')
image = Image.open('images/Data flowchart.jpg')
st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

st.write('Foram testados 11 modelos distintos: **Naive Bayes**, **LSTM** (Long-Short Term Memory), **LSTM bidirecional**, **CNN-1D** (Convolutional Neural Network), e combinações de LSTM e CNN, e LSTM bidirecional e CNN.'
' Variações destes modelos aplicando **LSI** (Latent Semantic Indexing) de antemão também foram testadas, além das versões iniciais utilizano apenas "tokenização".'
' Para todos eles, a pipeline seguida foi a mesma: uma vez treinados, validados e testandos os modelos desenvolvidos, eles são salvos para uso externo futuro. Este uso é tanto na própria website do projeto, quanto no bot, que faz requisições a Cloud Functions do Google Cloud Platform.')

st.subheader('Metodologia')
st.write('A metodologia utilizada para o desenvolvimento do projeto se baseia no ciclo de vida de um projeto de ciência de dados concebida pela Microsoft, o TDSP (Team Data Science Process). Trata-se de um modelo de trabalho bem amplo que apresenta dentro da concepção de um projeto quatro estágios principais: '
        'entendimento do negócio; coleta e entendimento de dados; modelagem; e deploy. \n\n'
        'Desse modo, dentro do contexto do projeto em questão, o seguinte fluxograma com a metodologia aplicada foi elaborado:')
tdsp = Image.open('images/fluxogramametodologia.jpg')
st.image(tdsp, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
st.write('Primeiramente, entendeu-se o problema, no caso o racismo contra populações asiáticas, quais suas peculiaridades e quais os possíveis meios de estudo e abordagem do mesmo. Dentro desse passo, além do estudo de conceitos de Machine Learning e Processamento de Linguagem Natural, o grupo realizou reuniões e conversas com Lucas Lago e David Nemer. '
        'O primeiro para compreender quais as limitações e características principais de bots e da rede social Twitter.'
        'Já o segundo é um influente antropólogo, especializado em antropologia da informática, que melhor explicou a problemática do racismo contra povos asiáticos dentro da internet. \n\n'
        'Em seguida, foram feitas pesquisas e coletas de bases de dados que pudessem auxiliar na criação de modelos em machine learning. Os dados obtidos, então, foram tratados e classificados. Após isso, avaliou-se a qualidade e quantidade daquilo que se coletou. Caso necessário, volta-se ao passo de entendimento do problema, dessa vex com uma melhor copmreensão de limitações em termos de dados. \n\n'
        'Por fim, foi feita a modeloagem de modelos e avaliados, de acordo com os desempenhos em métricas conhecidas, como: acurácia, recall, F1-score e precisão.')

st.subheader('Conclusão')
st.write('Escrever sobre conclusão')
