import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import streamlit.components.v1 as components


def cropped_image(photo):
    img = Image.open(photo)
    height,width = img.size
    lum_img = Image.new('L', [height,width] , 0)
    
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0), (height,width)], 0, 360, 
                fill = 255, outline = "white")
    img_arr = np.array(img)
    lum_img_arr = np.array(lum_img)
    final_img_arr = np.dstack((img_arr,lum_img_arr))
    return Image.fromarray(final_img_arr)


with open('.streamlit/style.css') as f:
    st.markdown(
        f'<style>{f.read()}</style>',
        unsafe_allow_html=True,
    )

st.title('Quem somos')

st.header('Integrantes')


st.subheader("Gabriel Oga Sanefuji")
(col1, col2) = st.columns([2,1])
with col1:
    #st.image("stuff in 1")
    st.write('Estudante do último ano de Engenharia de Computação (cooperativo) Poli-USP')
with col2:
    st.image(cropped_image('images/foto-gabriel.jpg'), caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")


st.subheader("Sandra Ayumi Nihama")
(col1, col2) = st.columns([1,2])
with col1:
    st.image(cropped_image('images/foto-sandrinha.jpeg'), caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
with col2:
    st.write('Estudante do último ano de Engenharia de Computação (cooperativo) Poli-USP, com Duplo Diploma da CentraleSupélec, França.  \n'
    'Membro da equipe de Tênis de Mesa da Poli desde 2017, fez parte também do Cursinho da Poli-USP durante a graduação.  \n'
    'Além de Dados e Machine Learning, suas maiores paixões são passar tempo com a família e amigos, descobrir músicas novas, desenhar e buscar aprender coisas novas :)')

st.write('\n\n')
st.header('Professor orientador')
st.subheader("Ricardo Luis de Azevedo da Rocha")
(col1, col2) = st.columns([2,1])
with col1:
    #st.image("stuff in 1")
    st.write('Estudante do último ano de Engenharia de Computação (cooperativo) Poli-USP')
with col2:
    st.image(cropped_image('images/foto-sandrinha.jpeg'), caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")


st.header('Twitter bot')
st.write('Acompanhe as postagens do nosso bot [@AreYou_Racist](https://twitter.com/AreYou_Racist) no Twitter:')

components.html(
    """
        <a class="twitter-timeline" data-width="400" data-height="600" href="https://twitter.com/Anitta?ref_src=twsrc%5Etfw">Tweets by Anitta</a> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """,
    height=500,
)
# Página do Twitter: https://twitter.com/AreYou_Racist