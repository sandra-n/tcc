import streamlit as st
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

st.title('Quem somos')

st.header('Integrantes')

st.header('Professor orientador')

st.header('Vídeo')
video_file = open("video/MyGameList.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)
st.video(data="https://www.youtube.com/watch?v=7jHcszxmKRw", format="video/mp4", start_time=0)

st.header('Twitter bot')