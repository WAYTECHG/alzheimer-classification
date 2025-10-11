import requests
import streamlit as st
from streamlit_lottie import st_lottie

def load_animation(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def home_page():
    animation = load_animation("https://lottie.host/2ace4c1e-f991-4f3c-93a8-873cc96fe5b9/w5mhsViRT4.json")
    with st.container():
        # st.write("---")
        left, right = st.columns(2)
        with left:
            st.title(":blue[Alzheimer System Prediction]")
            # st.write("###")
            tab1, tab2, tab3 = st.tabs([":blue[Definition]", ":blue[Reason]", ":blue[Aim]"])
            with tab1:
                st.subheader("What is Alzheimer?")
                st.write("Alzheimer's disease is a progressive neurological disorder that affects the brain, causing problems with memory, thinking, and behavior. It is the most common cause of dementia, a general term for memory loss and other cognitive abilities serious enough to interfere with daily life.")
                st.write("##") 
            with tab2:
                st.subheader("Why detection is needed?")
                st.write("Early detection of Alzheimer's disease is crucial for timely intervention, improving patient outcomes, and enhancing quality of life. It allows for prompt medical treatment, better care planning, and the opportunity to participate in clinical trials. Early diagnosis also reduces healthcare costs and provides access to support services for patients and caregivers, ultimately helping to manage symptoms and maintain independence for as long as possible.")
                st.write("##")
            with tab3:
                st.subheader("The aim of this project")
                st.write("The aim of the Alzheimer's prediction system project is to develop a reliable and efficient tool for the early detection of Alzheimer's disease. This system seeks to enable timely medical intervention, improve patient outcomes, and enhance quality of life. Additionally, it aims to support better resource allocation for patient care and provide valuable data for ongoing research and clinical trials.")
                st.write("##")
            st.write("Click learn more to know more about Alzheimer.")
            st.write("[Learn More >](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447)")
            st.write("Credit goes to :orange[MayoClinic.]")

        with right:
            st_lottie(animation, height = 300, key = "coding")
    
