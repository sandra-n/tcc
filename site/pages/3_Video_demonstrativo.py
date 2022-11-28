import streamlit as st

st.title('Vídeo demosntrativo')
st.write('O vídeo apresentado a seguir contém a explicação contextual do projeto assim como uma demonstração prática dos modelos implementados e do bot desenvolvido:')

video_file = open("video/MyGameList.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)
st.video(data="https://www.youtube.com/watch?v=7jHcszxmKRw", format="video/mp4", start_time=0)
