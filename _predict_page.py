import time
import joblib
import pandas as pd
from config import *
import streamlit as st
import plotly.express as px


def prediction_page():
    df = pd.read_csv('./data/alzheimer_dataset.csv')

    def convert_to_one_hot(selected_category, all_categories):
            one_hot = [True if category == selected_category else False for category in all_categories]
            for value in one_hot:
                user_input.append(value)

    def predict_alzheimer(input_data):
        loaded_model = joblib.load('./models/alzheimer_info_model.pkl')
        predictions = loaded_model.predict(input_data)

        return predictions

    st.title(":blue[Alzheimer Prediction Based On Data Info]")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:      
        st.write("## Physical Information")
        age = st.number_input("Age", min_value=0, max_value=122, step=1, value=65)
        gender = st.selectbox("Gender", ("Male", "Female"))
        education = st.number_input("Years of Education", min_value=0, value=12)

        st.write("<br>", unsafe_allow_html=True)

        st.write("## Demographics")
        ethnicity = st.radio("Ethnicity", ("Hisp/Latino", "Not Hisp/Latino", "Unknown"))
        race_cat = st.radio("Race Category", ("White", "Black", "Asian"))

        st.write("<br>", unsafe_allow_html=True)

        

    with col2:
        st.write("## Genetic Information")
        apoe_allele_type = st.selectbox("APOE Allele Type", ["APOE4_0", "APOE4_1", "APOE4_2"])
        st.markdown("""
        - APOE4_0: No copies of the APOE4 allele
        - APOE4_1: One copy of the APOE4 allele
        - APOE4_2: Two copies of the APOE4 allele
        """)
        
        apoe_genotype = st.selectbox("APOE4 Genotype", ("(2,2)", "(2,3)", "(2,4)", "(3,3)", "(3,4)", "(4,4)"))
        st.markdown("""
        - (2,2): Two copies of the APOE2 allele
        - (2,3): One copy of the APOE2 allele and one copy of the APOE3 allele
        - (2,4): One copy of the APOE2 allele and one copy of the APOE4 allele
        - (3,3): Two copies of the APOE3 allele
        - (3,4): One copy of the APOE3 allele and one copy of the APOE4 allele
        - (4,4): Two copies of the APOE4 allele
        """)
        
        imputed_genotype = st.selectbox("Imputed Genotype", ("True", "False"))
        st.markdown("""
        This indicates whether the genotype has been imputed. Imputation is a statistical method used to predict unobserved genotypes based on observed data.
        """)

        st.write("<br>", unsafe_allow_html=True)

    st.write("---")
    col3, col4 = st.columns(2)

    with col3:
        st.write("## Cognitive Assessment")
        # Adding the MMSE score input with slider
        mmse = st.slider("MMSE Score", min_value=20, max_value=30, step=1)
    with col4:
        st.write("##")
        st.markdown("""
        The Mini-Mental State Examination (MMSE) is a 30-point questionnaire used extensively in clinical and research settings to measure cognitive impairment. It is commonly used to screen for dementia.
        You can take the MMSE test.
        """)

    st.write("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict")
    st.write("")


    loading_bar = st.empty()
    if age and education and mmse and apoe_genotype and race_cat and gender and apoe_allele_type and imputed_genotype and ethnicity and predict_button:
        loading_bar.write("Thank you for entering the patient's information.")
        progress_text = "Please wait, we're predicting your clinical condition..."
        my_bar = loading_bar.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text)

        user_input = [age, education, mmse]

        convert_to_one_hot("PTRACCAT_" + race_cat, PTRACCAT_CATEGORIES)
        convert_to_one_hot("APOE Genotype_" + apoe_genotype, APOE_CATEGORIES)
        convert_to_one_hot("PTETHCAT_" + ethnicity, PTHETHCAT_CATEGORIES)
        convert_to_one_hot(apoe_allele_type, APOE4_CATEGORIES)
        convert_to_one_hot("PTGENDER_" + gender, PTGENDER_CATEGORIES)
        convert_to_one_hot("imputed_genotype_" + imputed_genotype, IMPUTED_CATEGORIES)

        data = pd.DataFrame([user_input])
        predicted_condition = predict_alzheimer(data)
        
        loading_bar.empty()
        
        st.write("## :blue[Predicted Clinical Condition:]", unsafe_allow_html=True)
        st.write(f"## <b>{ABBREVIATION[predicted_condition[0]]}</b> ({predicted_condition[0]})", unsafe_allow_html=True)
        st.write(f"{CONDITION_DESCRIPTION[predicted_condition[0]]}", unsafe_allow_html=True)
        st.write("##")

    # Define categorical and numerical columns and their descriptions
    cat_cols = ['Gender', 'Ethnicity', 'APOE4_Status', 'Imputed_Genotype', 'APOE_Genotype']
    cat_cols_dict = {
        'Gender': 'The gender of the person',
        'Ethnicity': 'The ethnicity of the person',
        'APOE4_Status': 'APOE status of the person',
        'Imputed_Genotype': 'Imputed genotype of the person',
        'APOE_Genotype': 'APOE genotype of the person'
    }
    num_cols = ['Age', 'Education_Year', 'MMSE']
    num_cols_dict = {
        'Age': 'The age of the person',
        'Education_Year': 'Number of years of education',
        'MMSE': 'Mini-Mental State Examination score'
    }

    # Ensure columns are treated as strings
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Color map for the classes
    color_discrete_map = {
        "AD": "#d26a3e",      # Alzheimer's Disease
        "LMCI": "#268d87",    # Late Mild Cognitive Impairment
        "CN": "#a2c5ac"       # Control Normal
    }

    # Creating the container for categorical feature visualization
    with st.expander('Look at the percentage of Alzheimer categories among categorical features'):
        # Creating a selectbox dropdown with the categorical features to choose from
        cat_option = st.selectbox('Select a feature to examine', cat_cols, key='cat_cols1')

        # The function to run the categorical plot
        def percentage_plot(col):
            # Ensure columns are treated as strings
            df[col] = df[col].astype(str)

            # Creates a temporary dataframe to get the percentages
            temp_df = df.groupby(col)['DX.bl'].value_counts(normalize=True)
            temp_df = temp_df.mul(100).rename('Percent').reset_index()
            temp_df['Percent'] = temp_df['Percent'].round(decimals=1)

            # Plot the percentages with the temporary dataframe
            fig = px.bar(temp_df, x=col, y='Percent', color='DX.bl',
                        barmode="group", text='Percent', color_discrete_map=color_discrete_map,
                        title=f"Percent Alzheimer Category By {col}")

            # Explanation of the features displays along with the graph
            st.markdown('**Explanation of the feature selected:**')
            st.markdown(f'**- {cat_option}: {cat_cols_dict[cat_option]}**')
            return fig

        # Running the function
        st.plotly_chart(percentage_plot(cat_option))

    # Creating the container for numerical feature visualization
    with st.expander('Look at Alzheimer in numerical features plotted together'):
        # Creating selectbox dropdowns with the numerical features to choose from
        num_optionX = st.selectbox('Select a feature for the X-axis', num_cols, key='x1')
        num_optionY = st.selectbox('Select a feature for the Y-axis', num_cols, key='y1')

        # The function to run the numerical plot
        def scatter_plot(x, y):
            # Plot the features selected by the user
            fig = px.scatter(df, x=x, y=y, color="DX.bl", color_discrete_map=color_discrete_map,
                            title=f"{y} vs {x} and Alzheimer Category")

            # Explanation of the features displays along with the graph
            st.markdown('**Explanation of the features selected:**')
            st.markdown(f'**- {num_optionX}: {num_cols_dict[num_optionX]}**')
            st.markdown(f'**- {num_optionY}: {num_cols_dict[num_optionY]}**')
            return fig

        # Running the function
        st.plotly_chart(scatter_plot(num_optionX, num_optionY))

    # Creating a new container for data distribution and correlation matrix
    with st.expander('Look at Data Distribution and Correlation Matrix'):
        st.write("#### Data Distribution")
        st.write("##### Histogram of Selected Features")

        num_optionX = st.selectbox('Select a Numerical Feature', num_cols, key='x3')

        # Updating histograms based on selected options
        fig_hist = px.histogram(df, x=num_optionX, color="DX.bl", nbins=30, title=f"Histogram of {num_optionX}",
                                color_discrete_map=color_discrete_map)
        st.plotly_chart(fig_hist)

        st.write("#### Correlation Matrix")
        corr_matrix = df[num_cols].corr()

        # Plot the correlation matrix using Plotly
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Geyser',
                            title='Correlation Matrix')

        # Show the plot in Streamlit
        st.plotly_chart(fig_corr)


    # Creating the container for the third plot
    with st.expander('Look at Alzheimer in a numerical feature alone or add a categorical feature'):

    # Creating selectbox dropdowns with the features to choose from, 'None' will render by default for the categorical option
        num_optionX2 = st.selectbox('Select a numerical feature for the X-axis', num_cols, key='x2')
        cat_optionY = st.selectbox('Select a categorical feature for the Y-axis if desired', cat_cols, key='y2')

    # The function to run the third plot
        def strip_plot(x, y='None'):

            # Plot the features selected by the user when y is set to 'None'
            if y=='None':
                fig = px.strip(df, x=x, orientation="h", color="DX.bl", color_discrete_sequence=color_discrete_map,
                            title=f'{x} and Depression')

                # Explaination of the feature to display along with the graph
                st.markdown('**Explaination of the feature selected:**')
                st.markdown(f'**- {num_optionX2}: {num_cols_dict[num_optionX2]}**')
                return fig

            # Plot the features selected by the user when y is not 'None'
            else:
                fig = px.strip(df, x=x, y=y, orientation='h', color='DX.bl', color_discrete_sequence=['#d26a3e','#268d87','#a2c5ac'],
                            title=f"{x} with {y} and Depression")

                # Explaination of the features displays along with the graph
                st.markdown('**Explaination of the features selected:**')
                st.markdown(f'**- {num_optionX2}: {num_cols_dict[num_optionX2]}**')
                st.markdown(f'**- {cat_optionY}: {cat_cols_dict[cat_optionY]}**')
                return fig

    # Running the function
        st.plotly_chart(strip_plot(num_optionX2, cat_optionY))




# prediction_page()