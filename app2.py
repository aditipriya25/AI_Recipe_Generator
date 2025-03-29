import streamlit as st
import bcrypt
import os
from datetime import datetime
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor

# Database Configuration
DATABASE = {
    'dbname': 'Recipe_generator',
    'user': 'postgres',
    'password': 'Aditi',
    'host': 'localhost',
    'port': 5432
}

# Establish Database Connection
def create_connection():
    return psycopg2.connect(**DATABASE)

# Initial Configurations
st.set_page_config(
    page_title="Smart Recipe Generator",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory to store profile pictures
PROFILE_PIC_DIR = "profile_pics"
if not os.path.exists(PROFILE_PIC_DIR):
    os.makedirs(PROFILE_PIC_DIR)

# Utility function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Utility function to verify passwords
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# Database functions
def init_db():
    with create_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT,
                password TEXT,
                join_date DATE,
                profile_pic TEXT
            );
            CREATE TABLE IF NOT EXISTS recipes (
                id SERIAL PRIMARY KEY,
                user_email TEXT REFERENCES users(email),
                recipe_name TEXT,
                ingredients TEXT,
                instructions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

# Call init_db to create tables if they do not exist
init_db()

# Store user profile picture
def save_profile_picture(image_file, email):
    image_path = os.path.join(PROFILE_PIC_DIR, f"{email}.png")
    image = Image.open(image_file)
    image.save(image_path)
    return image_path

# Register function with database storage
def register():
    st.title("Register 📝")
    email = st.text_input("Email")
    name = st.text_input("Name")
    password = st.text_input("Password", type="password")
    profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        profile_pic_path = None
        if profile_pic:
            profile_pic_path = save_profile_picture(profile_pic, email)
        
        hashed_pw = hash_password(password)
        join_date = datetime.now().date()

        # Insert user data into database
        with create_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                    INSERT INTO users (email, name, password, join_date, profile_pic) 
                    VALUES (%s, %s, %s, %s, %s)
                    """, (email, name, hashed_pw, join_date, profile_pic_path))
                    conn.commit()
                    st.success("Registration successful! Please login.")
                except psycopg2.IntegrityError:
                    conn.rollback()
                    st.error("User already exists!")

# Login function with database verification
def login():
    st.title("Login 🔐")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        with create_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cur.fetchone()
                
                if user and check_password(password, user['password']):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = email
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid login credentials.")

# Fetch user information from database
def get_user_info(email):
    with create_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cur.fetchone()

# Save recipe to database
def save_recipe(user_email, recipe_name, ingredients, instructions):
    with create_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO recipes (user_email, recipe_name, ingredients, instructions) 
            VALUES (%s, %s, %s, %s)
            """, (user_email, recipe_name, ingredients, instructions))
            conn.commit()
            st.success("Recipe saved successfully!")

# Sidebar for User Profile and Navigation
def sidebar_profile():
    if st.session_state["authenticated"]:
        user_info = get_user_info(st.session_state["username"])
        if user_info["profile_pic"]:
            st.sidebar.image(user_info["profile_pic"], width=100)
        else:
            st.sidebar.image("https://via.placeholder.com/100", width=100)
        st.sidebar.write(f"**{user_info['name']}**")
        st.sidebar.write(f"📧 {user_info['email']}")
        st.sidebar.write(f"📅 Joined: {user_info['join_date']}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Recipes"])
    return page

# Recipe Page
def recipe_page():
    st.title("Recipe Generator 🍲")
    recipe_name = st.text_input("Recipe Name")
    ingredients = st.text_area("Ingredients (comma-separated)")
    instructions = st.text_area("Instructions")
    if st.button("Save Recipe"):
        save_recipe(st.session_state["username"], recipe_name, ingredients, instructions)

# Main App Flow
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    auth_choice = st.sidebar.radio("Choose Action", ["Login", "Register"])
    if auth_choice == "Login":
        login()
    elif auth_choice == "Register":
        register()
else:
    selected_page = sidebar_profile()
    if selected_page == "Home":
        st.title("Home 🏠")
        st.write("Welcome to the **Smart Recipe Generator**!")
    elif selected_page == "Recipes":
        recipe_page()
    
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
