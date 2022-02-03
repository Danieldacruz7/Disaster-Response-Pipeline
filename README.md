# Disaster-Response-Pipeline

-- IN PROGRESS --

Project for creating an Natural Language Processing (NLP) pipeline for recognizing a disaster response.

## Table of Contents

1. [Project Motivation](https://github.com/Danieldacruz7/Disaster-Response-Pipeline#project-motivation)
2. [Installations](https://github.com/Danieldacruz7/Disaster-Response-Pipeline#installations)
3. [File Descriptions](https://github.com/Danieldacruz7/Disaster-Response-Pipeline#file-descriptions)
4. [How To Interact With the Project](https://github.com/Danieldacruz7/Disaster-Response-Pipeline#how-to-interact-with-the-project)

## Project Motivation:
One of the most important challenges in an emergency is the speed at which we respond to a disaster. The quicker the response, the quicker people are able to help in the case of a disaster. Thanks to the growth of social media platforms and the exponential growth of textual data, we are able to identify the presence of an emergency within seconds.

Using natural language processing, we can categorize text to identify the type of help that is required. Although this is an immense undertaking, we can start by categorizing disaster responses into different subcategories.

## Installations:
For this data science project, the following libraries are required:
- JSON
- NLTK
- Flask
- Regex
- Numpy
- Plotly
- Pandas
- Pickle
- Sklearn
- SQLAlchemy

Use "pip install" to download the libraries.

## File Descriptions:
The project consists of three main folders - app, data and model.

1. The app folder contains the HTML files, and the python file for deploying the Flask app.

2. The data folder contains the DisasterResponse database as well as the accompanying CSV files that contain the raw data. The process_data.py cleans the raw CSV files, and creates a database for the disaster response data.

3. The model folder contains the code that will extract information from the database, and fit the data onto a machine learning pipeline and output an NLP model. This model will classify text as different disaster responses.

## How To Interact With the Project:
If you would like to create the NLP model, use git clone to download the project. Then run the project in the sequence of process_data.py, train_classifier.py and finally the run.py file to deploy the application.

Alternatively, you can view the application @. The app is hosted on Heroku on a free subscription plan. (Reducing model size is required for upload. Application is not running yet.)
