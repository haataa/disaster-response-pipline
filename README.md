#### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions:](#Instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


# Installation<a name="Installation"></a>
libraries needed by this project are provided by the Anaconda distribution of Python. The code should run with no 
issues using Python versions 3.*.

# Project Motivation<a name="motivation"></a>
This project build  a web app to classify information on twitter according to its purpose and send the information to the appropriate aid agencies during a disaster event.

A machine learning model is trained to classify information into different categories using [dataset of Figure Eight - Multilingual Disaster Response Messages.](https://www.figure-eight.com/dataset/combined-disaster-response-data/)

# File Descriptions<a name="files"></a>
notebook folder contain ETL pipeline and ML pipeline notebook

data folder contain data and process_data.py file which is an ETL-pipeline for cleaning, transforming and storing the data from CSV-files

models folder contain train_classifier.py which train and build a ML model for classify messages. The output is a pickle file containing the fitted model.

app folder contain html template and run.py file to  run and render web app.

img folder contain screen shot of the web app

# Instructions<a name="Instructions"></a>
Run process_data.py
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

Run the web app
1.Save the app folder in the current working directory.
2.Run the following command in the app directory: python run.py
3.Go to http://0.0.0.0:3001/

# Licensing, Authors, Acknowledgements<a name="licensing"></a>

