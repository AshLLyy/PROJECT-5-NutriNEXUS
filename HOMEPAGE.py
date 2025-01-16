#%%
import streamlit as st
import os

from PIL import Image
# %%
st.set_page_config(
    page_title="main page"
)

st.title("NUTRINexus Bot")
st.sidebar.success("NutriNEXUS Main Page")

IMAGE_PATH = os.path.join(os.getcwd(), 'static', 'download.jpeg')

image = Image.open(IMAGE_PATH)
st.image(image, use_column_width=True)

st.write("NUTRINexus Bot is a smart, interactive nutrition assistant designed to provide personalized dietary guidance and support. It analyzes individual preferences, and goals to create tailored meal plans, suggest healthy recipes, and classify the food is healthy or unhealthy. NutriNexus promotes better nutrition habits, helps manage weight, improves overall well-being, and addresses specific needs like diabetes or heart health.")
st.write("With real-time advice, reminders, and motivation, NutriNexus empowers users to make informed food choices and achieve long-term health goals.")

st.title("NUTRITION")

st.write("Good nutrition is essential for health and development. It improves child and maternal health, strengthens the immune system, supports safer pregnancies, reduces the risk of diseases like diabetes and heart conditions, and promotes longer life.")
st.write("Malnutrition, including undernutrition and obesity, poses major health risks worldwide, especially in low- and middle-income countries. It leads to serious, lasting impacts on individuals, families, communities, and nations.")
st.write("Sorced by: World Health Organization(WHO)")
    
VIDEO_URL = "https://www.youtube.com/watch?v=TsZlTe_4LTU&ab_channel=WorldHealthOrganizationRegionalOfficefortheWesternPacific"
st.video(VIDEO_URL)

st.title("“People who laugh actually live longer than those who don’t laugh. Few persons realize that health actually varies according to the amount of laughter.” – James J. Walsh")