# HTML Table Text Extractor and Classifier

Check out the project live [here](https://html-table-text-extractor-and-classifier.onrender.com)!

![image](https://github.com/Srinivasan2821/HTML-Table-Text-Extractor-and-Classifier/assets/154582529/8c530e18-7aa1-4e0a-9283-6b75b8da6e5f)


## Project Overview

This project aims to classify different financial statements (Balance Sheet, Cash Flow, Notes, Income Statement, and Others) from HTML files. Each HTML file contains tables with financial data, and the goal is to accurately classify each document based on its content.

## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Extraction](#data-extraction)
3. [Data Preprocessing](#data-preprocessing)
4. [Handling Class Imbalance](#handling-class-imbalance)
5. [Feature Extraction](#feature-extraction)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Streamlit Application Development](#streamlit-application-development)
9. [Deployment](#deployment)

## Data Collection

The dataset consists of HTML files from five different folders: Balance Sheet, Cash Flow, Notes, Income Statement, and Others. Each folder contains multiple HTML files with financial data in table format.

## Data Extraction

Text is extracted from the HTML files, focusing on the column with the highest word count. BeautifulSoup is used to parse the HTML content, ensuring that the most relevant text data is captured for classification.

## Data Preprocessing

The extracted text is cleaned and preprocessed to prepare it for model training. This includes handling missing values, removing irrelevant information, and standardizing the text format to improve the quality of the data.

## Handling Class Imbalance

Class imbalance is addressed using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to improve model performance and ensure that the classifier does not bias towards the majority class.

## Feature Extraction

TF-IDF Vectorization is used to convert the text data into numerical features suitable for model training. This method captures the importance of words in each document relative to the entire dataset.

## Model Training

A Random Forest classifier is trained to predict the type of financial statement based on the extracted features. This model is chosen for its robustness and ability to handle high-dimensional data.

## Model Evaluation

The model is evaluated using standard metrics such as accuracy, precision, recall, and F1-score to ensure it performs well on unseen data. The evaluation helps in fine-tuning the model and improving its predictive performance.

## Streamlit Application Development

A user-friendly Streamlit application is developed to allow users to upload HTML files and receive predictions on the type of financial statement. The application provides an interface to display the extracted text and the predicted class.

## Deployment

The application is deployed on Render.com, making it accessible for users to classify new HTML files easily. The deployment ensures that the model is available for real-time predictions and can be accessed from any device with an internet connection.

## Conclusion

This project demonstrates the end-to-end process of building a machine learning application, from data collection and preprocessing to model training and deployment. The classifier effectively identifies different types of financial statements from HTML files, providing a valuable tool for financial data analysis.

## Access the Project

You can access the live project [here](https://html-table-text-extractor-and-classifier.onrender.com).

## Repository

The complete project code is available on GitHub. Feel free to explore the code, raise issues, and contribute to the project.

[GitHub Repository](https://github.com/Srinivasan2821/HTML-Table-Text-Extractor-and-Classifier)
