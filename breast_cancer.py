import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from IPython.display import HTML
# st.markdown(
#     """
#     <style>
#     /* Change background color of the whole page */
#     .main {
#         background-color: black;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.markdown("<h1 style='color:#7b241c'>Breast cancer detection </h1>", unsafe_allow_html=True)
data = load_breast_cancer()
breast_cancer_data=pd.DataFrame(np.c_[data['data'],data['target']],columns = np.append(data['feature_names'], ['target']))
st.write(breast_cancer_data)
st.sidebar.header("User Inputs")

def user_inputs():
    mean_texture = st.sidebar.slider("mean texture", 9.710000, 39.2800001, 18.829856)
    mean_smoothness = st.sidebar.slider("mean smoothness", 0.052630, 0.163400, 0.095388)
    mean_symmetry = st.sidebar.slider("mean symmetry", 0.106000, 0.3,0.179094)
    mean_fractal_dimension = st.sidebar.slider("mean_fractal_dimension", 0.049960, 0.09,0.063160)
    radius_error=st.sidebar.slider("radius_error", 0.11150, 5.0, 0.33198)
    texture_error=st.sidebar.slider("texture_error", 0.360200, 12.0, 1.220200)
    perimeter_error=st.sidebar.slider("perimeter_error", 0.757000, 0.03, 2.352856)
    smoothness_error=st.sidebar.slider("smoothness_error", 0.001713, 0.135,0.007129)
    compactness_error=st.sidebar.slider("compactness_error", 0.002252, 0.396, 0.024089)
    concavity_error	=st.sidebar.slider("concavity_error	", 0.0, 0.052, 0.029651)
    concave_points_error=st.sidebar.slider("concave_points_error", 0.0, 0.061, 0.011103)
    symmetry_error=st.sidebar.slider("symmetry_error", 0.00788, 0.0298, 0.020547)
    fractal_dimension_error=st.sidebar.slider("fractal_dimension_error", 0.0, 50.0, 0.003769)
    worst_texture=st.sidebar.slider("worst_texture", 12.0, 0.22, 25.052526)
    worst_smoothness=st.sidebar.slider("worst_smoothness", 0.07, 0.66, 0.130885)
    worst_symmetry=st.sidebar.slider("worst_symmetry", 0.15, 79484.8, 0.285628)
    worst_fractal_dimension	=st.sidebar.slider("worst_fractal_dimension	", 0.055, 0.207, 0.083347)
    breast_cancer_data = {
       'mean texture':mean_texture, 'mean smoothness':mean_smoothness, 'mean symmetry':mean_symmetry,
       'mean fractal dimension':mean_fractal_dimension, 'radius error':radius_error, 'texture error':texture_error,
       'perimeter error':perimeter_error, 'smoothness error':smoothness_error, 'compactness error':compactness_error,
       'concavity error':concavity_error, 'concave points error':concave_points_error, 'symmetry error':symmetry_error,
       'fractal dimension error':fractal_dimension_error, 'worst texture':worst_texture, 'worst smoothness':worst_smoothness,
       'worst symmetry':worst_symmetry, 'worst fractal dimension':worst_fractal_dimension,
    }

    features = pd.DataFrame([breast_cancer_data])
    return features
input_df = user_inputs()
st.sidebar.write("User inputs ",input_df)
st.header(
    """
         Predict  If  They  have  Breast  Cancer  
         """
)	

features=breast_cancer_data[['mean texture', 'mean smoothness', 'mean symmetry',
       'mean fractal dimension', 'radius error', 'texture error',
       'perimeter error', 'smoothness error', 'compactness error',
       'concavity error', 'concave points error', 'symmetry error',
       'fractal dimension error', 'worst texture', 'worst smoothness',
       'worst symmetry', 'worst fractal dimension']]

target=breast_cancer_data["target"]


# st.write(features)
# st.write(target)
Xtrain, Xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=0)

sv=SVC(kernel="linear",C=1)
sv.fit(Xtrain,ytrain)

ypred=sv.predict(Xtest)
st.write("Model Performance", classification_report(ypred, ytest))

if st.button("Predict"):
    prediction = sv.predict(input_df)
    #st.write("User request", prediction)
    
    if prediction[0] == 1:
					st.write("#### The patient is diagnosed with Breast Cancer.")
					st.image("https://i.pinimg.com/originals/46/09/23/460923f017d4f5019f28e6cdbae72dac.gif", use_container_width=True)
					
    else:
					
					st.balloons()
					st.write("#### The patient is not diagnosed with Breast Cancer.")
					st.image("https://i.pinimg.com/originals/26/7b/13/267b13ce9d04ad66d3a2d1e804b376c4.gif",use_container_width=True)
   
					 
           
	
           									
