# Book-Recommender

The Book Recommender allows the user to enter the title of a book and recieve up to five book recommendations based 
on consumer reviews. If the title entered is not in the dataset, books with similar titles will be suggested.

## Requirements

Our project requires the following modules:
 - numpy
 - pandas
 - matplotlib
 - sklearn.neighbors
 - streamlit
 - scipy.sparse
 - joblib
 - pickle
 - difflib

## Installation

1. Navigate to the desired folder.  
    >cd Book Recommender
2. Install the required modules:  
    >pip install -r requirements.txt
3. Create a new virtual environment in the folder and activate that environment:  
    >python -m venv .venv  
    source .venv/bin/activate
4. Run the application  
    >streamlit run app.py

## ML Models

1. SVM  
    First, we attempted to utilize a support vector machine (known as an SVM). The training complexity of SVM is highly dependent on the size of data set and because we had such a large dataset, the SVM was painstakingly slow. Thus, we decided to utlize a different model. 

2. KNearestNeighbors  
    
3. KNeighbors Classifiers  

<!-- ML models used and what they are doing -->

## Metrics section 

<!-- on how well the model did on the training data and eval data -->