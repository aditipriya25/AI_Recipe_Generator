# Import necessary packages
import streamlit as st
import cv2  # OpenCV for image processing
import easyocr  # EasyOCR
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
import numpy as np
import requests
import json
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import configparser
import re

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Retrieve the OpenAI API key and database credentials from the config file
api_key = config['api']['openai_api_key']
db_username = config['database']['db_username']
db_password = config['database']['db_password']
db_host = config['database']['db_host']
db_port = config['database']['db_port']
db_name = config['database']['db_name']

# Define the PostgreSQL database URL
DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Initialize the database engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Define a base class for declarative models
Base = declarative_base()

# Define the table schema for the User_recipe table
class UserRecipe(Base):
    __tablename__ = 'table_name' #------------------------------------Insert table name accordingly
    
    username = Column(String(50), primary_key=True)  # Using username as the primary key
    ingredients = Column(String(100), nullable=False)  # Detected product name
    recipe_name = Column(String(100), nullable=False)  # Recipe name
    extra_ingredients=Column(Text, nullable=False)
    instructions = Column(Text, nullable=False)  # Instructions
    cooking_time = Column(String(100), nullable=False)  # Cooking time
    nutrition = Column(Text, nullable=False)  # Nutrition information

# Create the User_recipe table in the database
Base.metadata.create_all(engine)

# Function to insert the detected product into the User_recipe table
def insert_detected_product(username, ingredient_name, recipe_name,extra_ingredients, instructions, cooking_time, nutrition):
    if ingredient_name and username:
        new_entry = UserRecipe(
            username=username,
            ingredients=ingredient_name,
            recipe_name=recipe_name,
            extra_ingredients=extra_ingredients,
            instructions=instructions,
            cooking_time=cooking_time,
            nutrition=nutrition
        )
        session.add(new_entry)
        try:
            session.commit()
            st.success(f"Product '{ingredient_name}' inserted into the User_recipe table successfully.")
        except Exception as e:
            session.rollback()
            st.error(f"Error inserting product: {e}")
    else:
        st.warning("No product name detected, skipping insertion.")

# Initialize Streamlit app
st.title("Smart Recipe Generator")
st.write("Upload one or more images of products or fruits/vegetables to get recipe suggestions.")

# Retrieve the model from the config file
pretrained_model = config['model']['pretrained_model']

# Load the pre-trained image classification model
# @st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(pretrained_model)
    return model

model = load_model()

# Load EasyOCR reader
# @st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'])  # Load English model

reader = load_easyocr_reader()

# Multiple image uploader
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Preprocessing function for image enhancement
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    return processed_image

# EasyOCR Function
def ocr_with_easyocr(image_path):
    result = reader.readtext(image_path, detail=0)  # Extract text without details
    return result

# Function to extract product name using OpenAI API
def extract_product_name_from_gpt(ocr_text, api_key):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'user', 'content': (
                "Please identify and extract the product name from the following text and return only the product name:\n"
                f"{ocr_text}"
            )}
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content'].strip()
    else:
        st.error("Error with OpenAI API: " + response.text)
        return None

# Function for image classification
def classify_image(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)  # Convert NumPy array to PIL image
    input_tensor = preprocess(pil_image).unsqueeze(0)  # Preprocess image and add batch dimension
    
    outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs.logits, dim=1).item()
    
    predicted_label = model.config.id2label[predicted_idx]
    return predicted_label

# Function to generate recipe using OpenAI API
def generate_recipe(ingredient_list, api_key):
    ingredient_str = ', '.join(ingredient_list)  # Convert list of ingredients to comma-separated string
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'user', 'content': (
                "Create a recipe using the following ingredients as the main ingredients. The recipe should include: Recipe Name, Ingredients, Instructions, Cooking Time, and Nutritional Information:\n"
                f"{ingredient_str}"
            )}
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content'].strip()
    else:
        st.error("Error with OpenAI API: " + response.text)
        return None

def parse_recipe_response(recipe_text):
    try:
        lines = recipe_text.strip().split('\n')
        
        recipe_name = lines[0].replace('Generated Recipe:', '').strip() if lines else 'Unknown Recipe'
        extra_ingredients = []
        instructions = []
        cooking_time = ''
        nutrition = ''
        
        parsing_extra_ingredients = False
        parsing_instructions = False
        parsing_nutrition = False

        for line in lines[1:]:
            line = line.strip()
            if line.lower().startswith('ingredients:'):
                parsing_extra_ingredients = True
                parsing_instructions = parsing_nutrition = False
                continue
            elif line.lower().startswith('instructions:'):
                parsing_instructions = True
                parsing_extra_ingredients = parsing_nutrition = False
                continue
            elif line.lower().startswith('cooking time:'):
                cooking_time = line[len('Cooking Time:'):].strip()
                parsing_extra_ingredients = parsing_instructions = parsing_nutrition = False
                continue
            elif line.lower().startswith('nutritional information:'):
                parsing_nutrition = True
                parsing_extra_ingredients = parsing_instructions = False
                continue
            
            # Append extra ingredients
            if parsing_extra_ingredients and line:
                extra_ingredients.append(line)
            
            # Append instructions
            elif parsing_instructions and line:
                instructions.append(line)
                
            # Append nutrition information
            elif parsing_nutrition and line:
                nutrition += line + ' '  # Concatenate multiple lines for nutrition

        return recipe_name, ' '.join(extra_ingredients), ' '.join(instructions), cooking_time, nutrition.strip()
    except Exception as e:
        st.error(f"Error parsing recipe response: {e}")
        return None, None, None, None, None


# Main function to handle OCR first, then classification if needed
def process_image(image, api_key):
    preprocessed_image = preprocess_image(image)
    
    easyocr_text = ocr_with_easyocr(preprocessed_image)
    if easyocr_text:
        combined_text = ' '.join(easyocr_text)
        product_name = extract_product_name_from_gpt(combined_text, api_key)
        if product_name:
            return product_name, False  # OCR succeeded, no need for classification

    # If OCR fails or returns no product name, run classification
    predicted_label = classify_image(image, model)
    return predicted_label, True  # OCR failed, using classification

# Button to process the uploaded images
if uploaded_files:
    detected_products = []
    
    username = st.text_input("Enter your username:")
    
    if st.button("Get Recipe"):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            # Convert PIL image to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Process each uploaded image to detect the product
            result, is_classified = process_image(image, api_key)
            detected_products.append(result)

        # If we have detected ingredients, generate a recipe
        if detected_products:
            recipe_text = generate_recipe(detected_products, api_key)
            if recipe_text:
                # Display the generated recipe directly
                st.write("### Generated Recipe:")
                st.write(recipe_text)
                
                # Parse the recipe response
                recipe_name, extra_ingredients, instructions, cooking_time, nutrition = parse_recipe_response(recipe_text)
                
                # Ensure that all necessary variables are defined before insertion
                if recipe_name and extra_ingredients and instructions and cooking_time and nutrition is not None:
                    insert_detected_product(
                        username,
                        ', '.join(detected_products),  # Join detected products
                        recipe_name,
                        ''.join(extra_ingredients),  # extra_ingredients, currently empty
                        ' '.join(instructions),  # Join instructions
                        cooking_time,
                        nutrition
                    )

                else:
                    st.warning("Some recipe components are missing.")
            else:
                st.warning("No recipe generated.")
        else:
            st.warning("No ingredients detected.")

            
# Close the session when done
session.close()
