import os
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import streamlit as st


# Function to extract text from the column with the highest word count in an HTML table
def extract_column_with_highest_word_count(file_content):
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        table = soup.find('table')
        if not table:
            return None

        rows = table.find_all('tr')
        if not rows:
            return None

        columns_data = []
        for row in rows:
            cells = row.find_all('td')
            for i, cell in enumerate(cells):
                if len(columns_data) <= i:
                    columns_data.append([])
                columns_data[i].append(cell.get_text(separator=' ').strip())

        word_counts = [sum(len(re.findall(r'\b\w+\b', cell)) for cell in column) for column in columns_data]
        max_word_count_index = word_counts.index(max(word_counts))
        highest_word_count_column = columns_data[max_word_count_index]
        filtered_column = [cell for cell in highest_word_count_column if not re.search(r'\d', cell)]

        return ' '.join(filtered_column)
    except Exception as e:
        return None

# Load the saved model
model_path = 'best_model.pkl'
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Assuming the TF-IDF vectorizer was also saved
vectorizer_path = 'tfidf_vectorizer.pkl'
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Function to classify a new HTML file
def classify_html_file(file_content):
    extracted_text = extract_column_with_highest_word_count(file_content)
    if extracted_text:
        # Transform the extracted text using the TF-IDF vectorizer
        transformed_text = vectorizer.transform([extracted_text])
        
        # Predict the class using the loaded model
        prediction = loaded_model.predict(transformed_text)
        return prediction[0], extracted_text
    else:
        return "Unable to extract text", None

# Streamlit app
st.set_page_config(layout="wide")
st.title(":rainbow[_HTML Table Text Extractor and Classifier :red[: ▶]_]")
st.write("")

with st.sidebar:
    Menu=st.sidebar.selectbox(":red[_**Please Select The Menu:-**_]",("Home", "Table Classifier"))
    with st.sidebar:
        st.header(":red[_Skills:_]")
        st.write(':blue[ :star: HTML Parsing]')
        st.write(':blue[ :star: Natural Language Processing]')
        st.write(':blue[ :star: Scikit-learn]')
        st.write(':blue[ :star: Random Forest Classification]')
        st.write(':blue[ :star: TF-IDF Vectorization]')
        st.write(':blue[ :star: Data Cleaning]')
        st.write(':blue[ :star: Model Evaluation]')
        st.write(':blue[ :star: Streamlit]')


if Menu == 'Home':

        st.header("ABOUT THIS PROJECT")

        st.subheader(":orange[1. PROBLEM STATEMENT:]")
        st.write('''***The task is to classify different financial statements (Balance Sheet, Cash Flow, Notes, Income Statement, and Others) from HTML files. Each HTML file contains tables with financial data, and the goal is to accurately classify each document based on its content.***''')

        st.subheader(":orange[2. PROJECT OVERVIEW:]")
        st.write('''***1. Understand the structure and content of the HTML files containing financial data.***''')
        st.write('''***2. Extract text data from the HTML files, focusing on the most relevant columns.***''')
        st.write('''***3. Preprocess the extracted text data to prepare it for model training.***''')
        st.write('''***4. Address class imbalance in the dataset to improve model performance.***''')
        st.write('''***5. Train a machine learning model (Random Forest) to classify the financial statements.***''')
        st.write('''***6. Evaluate and tune the model to achieve the best performance.***''')
        st.write('''***7. Develop a Streamlit application to upload HTML files and classify them using the trained model.***''')

        st.subheader(":orange[3. DATA ENGINEERING:]")
        st.write('''***1. Parse HTML files to extract text data.***''')
        st.write('''***2. Clean and preprocess the text data, including handling missing values and removing irrelevant information.***''')
        
        st.subheader(":orange[4. MODEL DEVELOPMENT:]")
        st.write('''***1. Vectorize the text data using TF-IDF Vectorization.***''')
        st.write('''***2. Address class imbalance using techniques such as SMOTE (Synthetic Minority Over-sampling Technique).***''')
        st.write('''***3. Train a Random Forest classifier to predict the type of financial statement.***''')
        st.write('''***4. Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.***''')
        
        st.subheader(":orange[5. APPLICATION DEVELOPMENT:]")
        st.write('''***1. Develop a Streamlit application that allows users to upload HTML files.***''')
        st.write('''***2. Extract text from the uploaded files and classify them using the trained model.***''')
        st.write('''***3. Display the classification results and extracted text in a user-friendly format.***''')
        st.write('''***4. Ensure the application is robust and can handle various types of HTML files.***''')


elif Menu == 'Table Classifier':

    uploaded_file = st.file_uploader("Choose an HTML file", type="html")

    if uploaded_file is not None:
        # Read the uploaded file content
        file_content = uploaded_file.read().decode("utf-8")
        
        # Classify the file content
        predicted_class, extracted_text = classify_html_file(file_content)
        
        st.subheader("Prediction")
        st.write(f"## :green[**The predicted class for the file is : {predicted_class}**]")

