from config import *
import streamlit as st
from web_pages._home_page import home_page, load_animation
from web_pages._chatbot_page import chatbot_page
from web_pages._predict_page import prediction_page 
from web_pages._about_us import about_us
from web_pages._mri_predict_page import mri_predict_page

st.set_page_config(page_title = "Alzheimer Prediction", page_icon = ":tada:", layout = "wide")

st.sidebar.title(":blue[Alzheimer Predictive System]")
pick_web = st.sidebar.selectbox("You are welcome to pick any of these website",
                        ["Home page", "Chatbot", "Alzheimer Prediction Data Info", "Alzheimer Prediction MRI","About us"])

st.sidebar.write("""
# :orange[Disclaimer!]
The prediction is not used on medical purpose, instead it is to use for informational purpose.\n

Thank you for paying attention!
""")
    
def main():
    if pick_web == "Home page":
        home_page()

    if pick_web == "Chatbot":
        chatbot_page()

    if pick_web == "Alzheimer Prediction Data Info":
        prediction_page()

    if pick_web == "Alzheimer Prediction MRI":
        mri_predict_page()

    if pick_web == "About us":
        about_us()

if __name__ == "__main__":
    main()
    

