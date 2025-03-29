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
import time
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
    __tablename__ = 'User_Recipes_Generated'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)  # Using username as the primary key
    ingredients = Column(String(100), nullable=False)  # Detected product name
    category_name = Column(String, nullable=False)
    recipe_name = Column(String(100), nullable=False)  # Recipe name
    extra_ingredients=Column(Text, nullable=False)
    instructions = Column(Text, nullable=False)  # Instructions
    cooking_time = Column(String(100), nullable=False)  # Cooking time
    nutrition = Column(Text, nullable=False)  # Nutrition information

class User(Base):
    __tablename__ = "Users_Login_Info"
    profile_picture = Column(LargeBinary, nullable=True)
    username = Column(String, primary_key=True, index=True)
    password = Column(String)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    phone = Column(String(15), nullable=False)

# Create the User_recipe table in the database
Base.metadata.create_all(engine)

                # Retrieve the model from the config file
pretrained_model = config['model']['pretrained_model']


def guide():
    st.title("📋 Instructions")

    # Acceptable Section
    with st.container():
        st.subheader("✅ Acceptable")
        st.markdown("These are examples of acceptable inputs:")
        col1, col2, col3, col4 = st.columns(4)

        image_width = 200  # Set fixed width for images # Set fixed height for images

        with col1:
            st.image("potato_1.jpeg", caption="Acceptable 1", width=image_width)
        with col2:
            st.image("mango_30.jpeg", caption="Acceptable 2", width=image_width)
        with col3:
            st.image("cornflakes.jpg", caption="Acceptable 3", width=image_width)
        with col4:
            st.image("pulses.jpeg", caption="Acceptable 4", width=image_width)

        st.markdown("""
        - Image of individual vegetable or fruit.
        - Image of individual group of single vegetable or fruit.
        - Image of packaged item.
        - Image of pulses in packages.
        """)

    # Not Acceptable Section
    with st.container():
        st.subheader("❌ Not Acceptable")
        st.markdown("These are examples of **not acceptable** inputs:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("mix veg.jpeg", caption="Not Acceptable 1", width=image_width)
        with col2:
            st.image("pulses_err.jpeg", caption="Not Acceptable 2", width=image_width)
        with col3:
            st.image("fridge.jpg", caption="Not Acceptable 3", width=image_width)
        with col4:
            st.image("blur2.jpg", caption="Not Acceptable 4", width=image_width)

        st.markdown("""
        - Image of group of vegetable or fruit.
        - Image of raw pulses without package.
        - Image of group of vegetable and fruit in Fridge.
        - Image with low resolution or blurry content.
        """)

    # OK Button
    with st.container():
        st.write("---")
        if st.button("OK"):
            st.success("Acknowledged! You may proceed.")
            st.rerun()

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
        color: Black;
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
    color: black; /* Default text color */
    font-size: 2.8rem; /* Font size */
    font-family: 'Roboto', sans-serif; /* Font style */
    padding: 15px; /* Padding */
    width: 200px; /* Fixed width */
    border-radius: 10px 10px 0 0; /* Rounded corners */
    border: none; /* No border */
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* Add subtle shadow */
    }}
    .stTabs [role="tablist"] > button:hover {{
        background: linear-gradient(90deg, #FF5722, #FF8A65);
        color: white
    }}
    .stTabs [role="tablist"] > button[aria-selected="true"] {{
        background: linear-gradient(90deg, #FF5722, #FF8A65);
        color: white;
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

# Modify the insert_detected_product function
def insert_detected_product(username, ingredient_name, selected_category, recipe_name, extra_ingredients, instructions, cooking_time, nutrition):
    if ingredient_name and username:
        session = SessionLocal()

        # Check if a record with the same username and ingredients already exists
        existing_entry = session.query(UserRecipe).filter_by(username=username, ingredients=ingredient_name).first()

        if existing_entry:
            st.info("This recipe already exists for the user.")
        else:
            # Insert the new recipe if no duplicate found
            new_entry = UserRecipe(
                username=username,
                ingredients=ingredient_name,
                category_name=selected_category,  # Category is included
                recipe_name=recipe_name,
                extra_ingredients=extra_ingredients,
                instructions=instructions,
                cooking_time=cooking_time,
                nutrition=nutrition
            )
            session.add(new_entry)
            try:
                session.commit()
                st.success("Recipe saved successfully.")
            except Exception as e:
                session.rollback()
                st.error(f"Error inserting product: {e}")
            finally:
                session.close()
    else:
        st.warning("No ingredient name detected or user not logged in, skipping insertion.")


def display_profile_details(username):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    session.close()

    if user:
        # Display the profile picture
        if user.profile_picture:
            profile_pic = io.BytesIO(user.profile_picture)
            profile_pic_base64 = base64.b64encode(profile_pic.getvalue()).decode("utf-8")
            
            # CSS for circular image and HTML to display it
            circular_image_html = f"""
                <style>
                    .circular-img {{
                        width: 200px;
                        height: 200px;
                        border-radius: 50%;
                        object-fit: cover;
                    }}
                </style>
                <img src="data:image/png;base64,{profile_pic_base64}" class="circular-img" alt="Profile Picture">
            """
            st.sidebar.markdown(circular_image_html, unsafe_allow_html=True)

        # Display the username and other details
        st.sidebar.markdown(f"<br><p style='font-family: Poppins, sans-serif; font-size: 18px; color: Black;'><strong>Username:</strong> {user.username}</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: Black;'><strong>Name:</strong> {user.first_name} {user.last_name}</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: Black;'><strong>Email:</strong> {user.email}</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: Black;'><strong>Phone:</strong> {user.phone}</p>", unsafe_allow_html=True)

def get_user_recipes(username):
    session = SessionLocal()
    recipes = session.query(UserRecipe).filter_by(username=username).all()
    session.close()
    return recipes

def format_text_with_newlines(text):
    # Replace commas and periods with a newline for better readability
    return text.replace("-", "<br>")

def format_text_with_newstyle(text):
    # Replace commas and periods with a newline for better readability
    return text.replace(",", "<br>")

def format_text(text):
    # Replace commas and periods with a newline for better readability
    return text.replace("1.", "<br>").replace("2.", "<br>").replace("3.", "<br>").replace("4.", "<br>").replace("5.", "<br>").replace("6.", "<br>").replace("7.", "<br>").replace("8.", "<br>").replace("9.", "<br>")

def display_saved_recipes(): 
    st.markdown("<h2 style='text-align: center; color: white;'>Saved Recipe</h2>", unsafe_allow_html=True)

    # Fetch saved recipes for the current user
    username = st.session_state.get('username')
    if username:
        saved_recipes = get_user_recipes(username)
        
        if saved_recipes:
            # Dropdown to select a saved recipe
            recipe_names = [recipe.recipe_name for recipe in saved_recipes]
            selected_recipe = st.selectbox("Select a Saved Recipe", recipe_names)

            # Display selected recipe details
            for idx, recipe in enumerate(saved_recipes):
                if recipe.recipe_name == selected_recipe:
                    # Add a unique key to the button using the index
                    if st.button(f"View {recipe.recipe_name}", key=f"view_{recipe.recipe_name}_{idx}"):
                        st.markdown("<div class='saved-recipe-card'>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='font-family: Poppins, sans-serif; font-size: 24px; color: Black; text-align: center;'>{recipe.recipe_name}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: black;'><strong>Ingredients:</strong><br> {format_text_with_newstyle(recipe.ingredients)}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: black;'><strong>Extra Ingredients:</strong>{format_text_with_newlines(recipe.extra_ingredients)}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: black;'><strong>Instructions:</strong>{format_text(recipe.instructions)}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: black;'><strong>Cooking Time:</strong><br> {recipe.cooking_time} minutes</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-family: Poppins, sans-serif; font-size: 18px; color: black;'><strong>Nutrition:</strong> {format_text_with_newlines(recipe.nutrition)}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No saved recipes found.</p>", unsafe_allow_html=True)
    else:
        st.warning("Please log in to view your saved recipes.")
            
def register():
    st.title("Smart Recipe Generator")
    st.subheader("Register")
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
    
def login():
    st.title("Smart Recipe Generator")
    st.subheader("Login")
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
            
def main():
    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if 'instruction' not in st.session_state:
        st.session_state['instruction'] = True  # Set to True to show instructions after login

    if not st.session_state['authenticated']:
        st.sidebar.title("Menu")
        choice = st.sidebar.radio("Select Option", ["Login", "Register"])
        
        if choice == "Register":
            register()  # Call your register function
        else:
            login()  # Call your login function
    else:
        # Display instructions only once after login
        if st.session_state['instruction']:
            st.session_state['instruction'] = False
            guide()  # Call your instructions function
        
        else:
            # Sidebar for Profile Details
            st.sidebar.title("Profile Details")
            username = st.session_state.get('username')  # Fetch the username from session state
            display_profile_details(username)
            
            if st.sidebar.button("Logout"):
                st.session_state['authenticated'] = False
                st.session_state[instructions]=True
                st.session_state.pop('username', None)  # Clear username from session state
                st.rerun()  # Redirect to login page

                # Initialize Streamlit app
            
            # Main content - Tabs
            tab1, tab2, tab3,tab4 = st.tabs(["Home", "Saved Recipes", "Meal Planner","Search"])

            with tab1:
                st.markdown("<h2 style='text-align: center; color: white;'>Generate Recipe</h2>", unsafe_allow_html=True)
                st.write("Upload one or more images of products or fruits/vegetables to get recipe suggestions.")

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
                        'temperature': 0.8, 
                        'messages': [
                            {'role': 'user', 'content': (
                                "Create a recipe using the following ingredients and food category as the main ingredients. The recipe should include: Recipe Name, Ingredients, Instructions, Cooking Time, and Nutritional Information:\n"
                                f"{ingredient_str}" "and" f"{selected_category}"
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
                    
                if 'save_clicked' not in st.session_state:
                    st.session_state.save_clicked = False
                if 'another_recipe_clicked' not in st.session_state:
                    st.session_state.another_recipe_clicked = False
                if 'recipe_generated' not in st.session_state:
                    st.session_state.recipe_generated = False
                if 'selected_category' not in st.session_state:
                    st.session_state.selected_category = None  # Initialize selected category in session state

                # Reset save_clicked when generating a new recipe
                if uploaded_files:
                    detected_products = []
                    categories = ["Vegetarian", "Non-Vegetarian", "Desserts", "Vegan", "Gluten-Free", "Snacks"]

                    # Create a selectbox for category selection and store it in session state
                    selected_category = st.selectbox("Choose a Food Category", categories)
                    st.session_state.selected_category = selected_category  # Store selected category in session state

                    if st.button("Get Recipe"):
                        st.session_state.recipe_generated = False  # Reset save state for new recipe
                        st.session_state.save_clicked = False  # Reset save state to False when generating a new recipe
                        
                        # Display a loading GIF while processing
                        with st.spinner("Processing..."):
                            loader_placeholder = st.empty()
                            with loader_placeholder.container():
                                st.image("123.gif", width=200)  # Display your local GIF loader
                                time.sleep(1)  # Small delay to ensure loader is visible before processing begins

                            # Backend process
                            for uploaded_file in uploaded_files:
                                image = Image.open(uploaded_file)
                                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                                result, is_classified = process_image(image, api_key)
                                detected_products.append(result)

                            if detected_products:
                                recipe_text = generate_recipe(detected_products, api_key)

                                if recipe_text:
                                    st.write("### Generated Recipe:")
                                    st.write(recipe_text)

                                    # Parse recipe details and store them in session state
                                    try:
                                        recipe_name, extra_ingredients, instructions, cooking_time, nutrition = parse_recipe_response(recipe_text)
                                        st.session_state.update({
                                            'recipe_name': recipe_name,
                                            'extra_ingredients': extra_ingredients,
                                            'instructions': instructions,
                                            'cooking_time': cooking_time,
                                            'nutrition': nutrition,
                                            'detected_products': detected_products,
                                            'recipe_generated': True  # Set flag to true after recipe is generated
                                        })
                                    except Exception as e:
                                        st.error(f"Error parsing recipe: {e}")
                                else:
                                    st.warning("Recipe generation failed. Please try again.")
                            else:
                                st.warning("No ingredients detected.")

                            # Clear the loader after processing is complete
                            loader_placeholder.empty()
                            
                # Only display the Save Recipe button if a recipe is generated
                if st.session_state.get('recipe_generated', False):
                    if st.button("Another Recipe") and not st.session_state.get('another_recipe_clicked', False):
                        st.session_state.another_recipe_clicked = True
                        st.session_state.save_clicked = False

                            # Backend process
                        for uploaded_file in uploaded_files:
                            image = Image.open(uploaded_file)
                            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                            result, is_classified = process_image(image, api_key)
                            detected_products.append(result)

                        if detected_products:
                            recipe_text = generate_recipe(detected_products, api_key)

                            if recipe_text:
                                st.write("### Generated Recipe:")
                                st.write(recipe_text)

                                    # Parse recipe details and store them in session state
                                try:
                                    recipe_name, extra_ingredients, instructions, cooking_time, nutrition = parse_recipe_response(recipe_text)
                                    st.session_state.update({
                                        'recipe_name': recipe_name,
                                        'extra_ingredients': extra_ingredients,
                                        'instructions': instructions,
                                        'cooking_time': cooking_time,
                                        'nutrition': nutrition,
                                        'detected_products': detected_products,
                                        'recipe_generated': True  # Set flag to true after recipe is generated
                                    })
                                except Exception as e:
                                    st.error(f"Error parsing recipe: {e}")
                            else:
                                st.warning("Recipe generation failed. Please try again.")
                        else:
                            st.warning("No ingredients detected.")
                        
                if st.button("Save Recipe") and not st.session_state.get('save_clicked', False):
                    st.session_state.recipe_generated = False
                    st.session_state.another_recipe_clicked = False
                    st.session_state.save_clicked = True
                        
                        # Retrieve saved details from session state
                    category_name = st.session_state.selected_category  # Retrieve selected category from session state
                    recipe_name = st.session_state.get('recipe_name')
                    extra_ingredients = st.session_state.get('extra_ingredients')
                    instructions = st.session_state.get('instructions')
                    cooking_time = st.session_state.get('cooking_time')
                    nutrition = st.session_state.get('nutrition')
                    detected_products = st.session_state.get('detected_products')
                    username = st.session_state.get('username')

                    if username and recipe_name and extra_ingredients and instructions and cooking_time and nutrition:
                        try:
                            # Call the database insert function and include category_name
                            insert_detected_product(
                                username,
                                ', '.join(detected_products),
                                category_name,  # Save the category in the database
                                recipe_name,
                                ''.join(extra_ingredients),
                                ''.join(instructions),
                                cooking_time,
                                nutrition
                            )
                            st.success("Recipe saved successfully.")
                        except Exception as e:
                            st.error(f"Error saving to database: {e}")
                    else:
                        st.warning("Incomplete recipe details or missing username; cannot save.")
                else:
                    st.write("Please generate a recipe first.")
   
            with tab2:
                display_saved_recipes()
                

if __name__ == "__main__":
    main()