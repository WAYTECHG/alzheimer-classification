import requests
import streamlit as st
from streamlit_lottie import st_lottie

def load_animation(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def about_us():
    animation = load_animation("https://lottie.host/308a3c5a-6d72-4153-b9f6-3b7a69459121/exYSxMZhgD.json")
    # animation = load_animation("https://lottie.host/b4b816f8-600b-4978-99fb-8af267ed48dc/Z7PObTYybV.json")
    
    st.markdown(f"<h1 style= 'text-align:center; color:#60b4ff'>Group Members</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    group_members = [
        {"name": "Oh Kai Wen", "roles": "Feature Engineering and Data Balancing, Model Training and Evaluation, Hyperparameter Tuning, Report Writing"},
        {"name": "Wilbert Andrew Yonathan", "roles": "Model Training and Evaluation, Hyperparameter Tuning, Streamlit Application Development, Report Writing"},
        {"name": "Alsthania", "roles": "Data Collection and Cleaning, Exploratory Data Analysis, Hyperparameter Tuning, MRI Image Analysis, Report Writing"},
        {"name": "Lu Zhuo Qing", "roles": "Exploratory Data Analysis, Hyperparameter Tuning, MRI Image Analysis, Report Writing"}
    ]

    left_col, center_col, right_col = st.columns([1, 2, 1])

    for i, member in enumerate(group_members):
        if i % 2 == 0:
            with left_col:
                st.markdown(f"""
                    <div style='background-color:rgba(0, 0, 0, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <h2 style='text-align:center; color:#B5B5D0;'>{member['name']}</h2>
                        <p style='text-align:center; color:#B0B0B0;'>Roles: {member['roles']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with right_col:
                st.markdown(f"""
                    <div style='background-color:rgba(0, 0, 0, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                        <h2 style='text-align:center; color:#B5B5D0;'>{member['name']}</h2>
                        <p style='text-align:center; color:#B0B0B0;'>Roles: {member['roles']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with center_col:
        st_lottie(animation, height = 350, key = "coding")