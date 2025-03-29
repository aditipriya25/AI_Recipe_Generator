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
from sqlalchemy import create_engine, Column, String, Text, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import configparser
import re
import hashlib
from PIL import Image
import base64
import io

st.set_page_config(
    page_title="Smart Recipe Generator", page_icon="🍲", layout="wide",)
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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Define a base class for declarative models
Base = declarative_base()

# Define the table schema for the User_recipe table
class UserRecipe(Base):
    __tablename__ = 'Recipes_Generated_by_user'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)  # Using username as the primary key
    ingredients = Column(String(100), nullable=False)  # Detected product name
    recipe_name = Column(String(100), nullable=False)  # Recipe name
    extra_ingredients=Column(Text, nullable=False)
    instructions = Column(Text, nullable=False)  # Instructions
    cooking_time = Column(String(100), nullable=False)  # Cooking time
    nutrition = Column(Text, nullable=False)  # Nutrition information

class User(Base):
    __tablename__ = "Users_Login"
    profile_picture = Column(LargeBinary, nullable=True)
    username = Column(String, primary_key=True, index=True)
    password = Column(String)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    phone = Column(String(15), nullable=False)

# Create the User_recipe table in the database
Base.metadata.create_all(engine)


# Password helper function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Password validation function
def is_valid_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one digit."
    if not re.search(r"[@$!%*?&]", password):
        return "Password must contain at least one special character (@, $, !, %, *, ?, &)."
    return None

# Check if username exists in the database
def username_exists(username):
    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    return user is not None
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load and encode the background image
image_path = "18.jpg"  # Replace with your local image path
base64_image = get_base64_image(image_path)
st.markdown(
    f"""
<style>
    /* Importing Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Full-page background image */
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay for readability */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }}

    /* Centered form container with shadow and transparency */
    .form-container {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.5);
        max-width: 400px;
        margin: 0 auto;
        font-family: 'Roboto', sans-serif;
        position: relative;
        top: 100px;
        z-index: 1;
        animation: fadeInUp 0.5s ease forwards;
    }}

    /* Fade in up effect */
    @keyframes fadeInUp {{
        from {{ transform: translateY(30px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* Styled heading */
    h2 {{
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 20px;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);
    }}

    /* Input field styling */
    .stTextInput input, .stPassword input {{
        font-size: 1rem;
        border: none;
        border-bottom: 2px solid #FF5722;
        background: transparent;
        color: white;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        padding: 15px;
    }}
    .stTextInput input:focus, .stPassword input:focus {{
        border-bottom: 2px solid #FF5722;
        box-shadow: 0px 4px 10px rgba(255, 87, 34, 0.3);
        outline: none;
    }}

    /* Gradient button with hover */
    .stButton > button {{
        background: linear-gradient(90deg, #FF5722, #FF8A65);
        color: white; /* Default text color */
        font-size: 1.2rem;
        padding: 12px 28px;
        border-radius: 30px;
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        font-family: 'Roboto', sans-serif;
    }}

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {{
        color: white !important; /* Ensure text remains white on hover, focus, and active */
        outline: none; /* Remove outline on focus */
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 300%;
        height: 300%;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: all 0.5s ease;
        transform: translate(-50%, -50%) scale(0);
        z-index: 0;
    }}
    .stButton > button:hover::after {{
        transform: translate(-50%, -50%) scale(1);
    }}
    .stButton > button:hover {{
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px);
    }}
    .stButton > button span {{
        position: relative;
        z-index: 1;
    }}
    .stTabs [role="tablist"] > button {{
        color: white/* Default text color */
        font-size: 1.2rem;
        padding: 15px;
        border-radius: 10px 10px 0 0;
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        font-family: 'Roboto', sans-serif;
    }}
    .stTabs [role="tablist"] > button:hover {{
        background: linear-gradient(90deg, #FF5722, #FF8A65);
    }}
    .stTabs [role="tablist"] > button[aria-selected="true"] {{
        background: linear-gradient(90deg, #FF5722, #FF8A65);
        background-color: white;
        font-weight: bold;
    }}
    
</style>
""",
    unsafe_allow_html=True
)


# Register a new user
def register_user(profile_pic_data, username, password, first_name, last_name, email, phone):
    session = SessionLocal()
    hashed_password = hash_password(password)
    
    # Create a new user with additional details
    new_user = User(
        profile_picture=profile_pic_data,
        username=username,
        password=hashed_password,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone
    )
    
    session.add(new_user)
    session.commit()
    session.close()

def register():
    st.title("Register")
    profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")

    if st.button("Register"):
        if not (first_name and last_name and email and phone and username and password and confirm_password):
            st.warning("Please fill out all fields.")
        elif username_exists(username):
            st.warning("Username already exists!")
        elif "@" not in email or "." not in email:
            st.warning("Please enter a valid email address.")
        else:
            validation_message = is_valid_password(password)
            if validation_message:
                st.warning(validation_message)
            elif password != confirm_password:
                st.warning("Passwords do not match!")
            else:# Convert the profile picture to binary data if uploaded
                profile_pic_data = profile_pic.read() if profile_pic else None
                register_user(profile_pic_data,username, password, first_name, last_name, email, phone)
                st.success("Registration successful! Please log in.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    
# Modify the insert_detected_product function
def insert_detected_product(username, ingredient_name, recipe_name, extra_ingredients, instructions, cooking_time, nutrition):
    if ingredient_name and username:
        session = SessionLocal()
        
        # Check if a record with the same username and ingredients already exists
        existing_entry = session.query(UserRecipe).filter_by(username=username, ingredients=ingredient_name).first()
        
        if existing_entry:
            st.info("Recipe generated successfully.")
        else:
            # Insert the new recipe if no duplicate found
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
                st.success("Recipe generated successfully.")
            except Exception as e:
                session.rollback()
                st.error(f"Error inserting product: {e}")
            finally:
                session.close()
    else:
        st.warning("No product name detected or user not logged in, skipping insertion.")

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        session = SessionLocal()
        user = session.query(User).filter_by(username=username).first()
        session.close()
        
        if user is None:
            st.warning("Username does not exist!")
        elif user.password != hash_password(password):
            st.warning("Incorrect password!")
        else:
            st.session_state['authenticated'] = True
            st.session_state['username'] = username  # Store the logged-in username
            st.success("Login successful!")
            st.rerun()

def display_profile_details(username):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    session.close()

    if user:
        # Display the profile picture
        if user.profile_picture:
            profile_pic = io.BytesIO(user.profile_picture)
            st.sidebar.image(profile_pic, caption="Profile Picture", use_column_width=False, output_format="auto", width=100)
            # Applying the circular style

        # Display the username and other details
        st.sidebar.subheader(f"Username: {user.username}")
        st.sidebar.write(f"Name: {user.first_name} {user.last_name}")
        st.sidebar.write(f"Email: {user.email}")
        st.sidebar.write(f"Phone: {user.phone}")

def get_user_recipes(username):
    session = SessionLocal()
    recipes = session.query(UserRecipe).filter_by(username=username).all()
    session.close()
    return recipes


def display_saved_recipes():
    # Load custom fonts and styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            color: white;
        }
        .saved-recipe-card {
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            width: 300px;
            margin: 10px;
            text-align: center;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<h1 style='text-align: center;'>Saved Recipes</h1>", unsafe_allow_html=True)

    # Fetch saved recipes for the current user
    username = st.session_state.get('username')
    if username:
        saved_recipes = get_user_recipes(username)
        
        if saved_recipes:
            # Dropdown to select a saved recipe
            recipe_names = [recipe.recipe_name for recipe in saved_recipes]
            selected_recipe = st.selectbox("Select a Saved Recipe", recipe_names)

            # Display selected recipe details
            for recipe in saved_recipes:
                if recipe.recipe_name == selected_recipe:
                    # st.markdown(f"<div class='saved-recipe-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{recipe.recipe_name}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Ingredients:</strong> {recipe.ingredients}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Extra Ingredients:</strong> {recipe.extra_ingredients}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Instructions:</strong> {recipe.instructions}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Cooking Time:</strong> {recipe.cooking_time} minutes</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Nutrition:</strong> {recipe.nutrition}</p>", unsafe_allow_html=True)
                    st.button(f"View {recipe.recipe_name}")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No saved recipes found.</p>", unsafe_allow_html=True)
    else:
        st.warning("Please log in to view your saved recipes.")

      
def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.sidebar.title("Menu")
        choice = st.sidebar.radio("Select Option", ["Login", "Register"])
        
        if choice == "Register":
            register()
        else:
            login()
    else:
        st.sidebar.title("Profile Details")
        # Display user profile details
        username = st.session_state.get('username')
        display_profile_details(username)
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state.pop('username', None)  # Clear username from session state
            st.rerun()  # Redirect to login page

            # Initialize Streamlit app
        
        # Main content - Tabs
        tab1, tab2= st.tabs(["Home", "Saved Recipes"])

        with tab1:
            st.markdown("<h2 style='text-align: center; color: white;'>Generate Recipe</h2>", unsafe_allow_html=True)
            st.write("Upload one or more images of products or fruits/vegetables to get recipe suggestions.")

                # Retrieve the model from the config file
            pretrained_model = config['model']['pretrained_model']

                # Load the pre-trained image classification model
            @st.cache_resource
            def load_model():
                model = AutoModelForImageClassification.from_pretrained(pretrained_model)
                return model

            model = load_model()

                # Load EasyOCR reader
            @st.cache_resource
            def load_easyocr_reader():
                return easyocr.Reader(['en'])  # Load English model

            reader = load_easyocr_reader()

                # Multiple image uploader
            uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

                # Preprocessing function for image enhancement
            def preprocess_image(image):
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
            if 'save_clicked' not in st.session_state:
                st.session_state.save_clicked = False
            if 'recipe_generated' not in st.session_state:
                st.session_state.recipe_generated = False

            # Reset save_clicked when generating a new recipe
            if uploaded_files:
                detected_products = []
                    
                    # username = st.text_input("Enter your username:")
                    
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
                                username = st.session_state.get('username')
                                insert_detected_product(
                                    username,
                                    ', '.join(detected_products),  # Join detected products
                                    recipe_name,
                                    ''.join(extra_ingredients),  # extra_ingredients, currently empty
                                    ''.join(instructions),  # Join instructions
                                    cooking_time,
                                    nutrition
                                )
                            else:
                                st.warning("Some recipe components are missing.")
                        else:
                            st.warning("No recipe generated.")
                    else:
                        st.warning("No ingredients detected.")

        with tab2:
            display_saved_recipes()
if __name__ == "__main__":
    main()