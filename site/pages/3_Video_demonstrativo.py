import streamlit as st
import os
from pathlib import Path

path = os.path.dirname(__file__)
path = Path(path)
parent_dir = str(path.parent.absolute())

with open(str(parent_dir) + '/../.streamlit/style.css') as f:
    st.markdown(
        f'<style>{f.read()}</style>',
        unsafe_allow_html=True,
    )
    
st.title('Vídeo demonstrativo')
st.write('O vídeo apresentado a seguir contém a explicação contextual do projeto assim como uma demonstração prática dos modelos implementados e do bot desenvolvido:')

video_file = open(parent_dir + "/video/MyGameList.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)
st.video(data="https://www.youtube.com/watch?v=7jHcszxmKRw", format="video/mp4", start_time=0)
