import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
from streamlit_drawable_canvas import st_canvas



# Load pre-trained models
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App
st.title("Digit Recognition App")

st.markdown("Draw a digit (0-9) below:")

# Create a canvas component with white background and black stroke color
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)", 
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas"
)
# Add a Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert drawn image to grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        
        # Display the drawn digit
        # st.image(img, caption="Your drawing", use_column_width=True)

        # Preprocess the image for model prediction
        img_array = np.array(img).reshape(1, -1)
        img_scaled = scaler.transform(img_array)
        img_pca = pca.transform(img_scaled)
        
        # Predict the digit
        prediction = model.predict(img_pca)


        
        # Display the prediction
        st.write(f"Predicted digit: {prediction[0]}")

        st.write("img_array: ",img_array.reshape(28,28),sep='\n')
    else:
        st.write("Please draw a digit before clicking Predict.")