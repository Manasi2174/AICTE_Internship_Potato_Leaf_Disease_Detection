import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Streamlit Side Bar
st.sidebar.title("Plant Disease System For Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page',['Home','Disease Recognition'])

# Adding Some image in UI
from PIL import Image
img = Image.open('Disease.png')
st.image(img)

# Adding title on home page
if (app_mode=='Home'):
    st.markdown("<h1 style='text-aling:center;'>Plant Disease System For Sustainable Agriculture",unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System For Sustainable Agriculture') 

# Section where we are able to upload our image 
test_image = st.file_uploader('Choose an Image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_column_width=True) 

# Creating Prediction button 
if (st.button('Predict')):
    st.snow()
    st.write('Our Prediction')
    result_index = model_prediction(test_image)
    # Now Displaying Our Results
    class_name = ['Potato__Early_blight','Potato__Late_blight','Potato__healthy']
    st.success('Model is Predicting Its a {}'.format(class_name[result_index]))
