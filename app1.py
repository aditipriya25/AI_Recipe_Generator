import streamlit as st
import bcrypt
import os
from datetime import datetime
from PIL import Image

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

# Utility function to hash passwords with bcrypt
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Utility function to verify passwords
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# Sample User Data Storage (replace with a database for production)
users_db = {
    "user@example.com": {
        "name": "John Doe",
        "password": hash_password("password"),  # Hashed password
        "join_date": "2024-01-01",
        "profile_pic": None  # No profile picture initially
    }
}

# Helper Functions for Authentication and Profile Picture Handling
def save_profile_picture(image_file, email):
    image_path = os.path.join(PROFILE_PIC_DIR, f"{email}.png")
    image = Image.open(image_file)
    image.save(image_path)
    return image_path

def register():
    st.title("Register 📝")
    email = st.text_input("Email")
    name = st.text_input("Name")
    password = st.text_input("Password", type="password")
    profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if email in users_db:
            st.error("User already exists!")
        else:
            profile_pic_path = None
            if profile_pic:
                profile_pic_path = save_profile_picture(profile_pic, email)

            users_db[email] = {
                "name": name,
                "password": hash_password(password),
                "join_date": datetime.now().strftime("%Y-%m-%d"),
                "profile_pic": profile_pic_path
            }
            st.success("Registration successful! Please login.")

def login():
    st.title("Login 🔐")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email in users_db and check_password(password, users_db[email]["password"]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = email
            st.success("Logged in successfully!")
            
        else:
            st.error("Invalid login credentials.")

def get_user_info(email):
    return users_db.get(email, {"name": "Guest", "email": "guest@example.com", "join_date": "N/A", "profile_pic": None})

# Sidebar for User Profile and Navigation
def sidebar_profile():
    if st.session_state["authenticated"]:
        user_info = get_user_info(st.session_state["username"])
        if user_info["profile_pic"]:
            st.sidebar.image(user_info["profile_pic"], width=100)
        else:
            st.sidebar.image("https://via.placeholder.com/100", width=100)  # Placeholder image URL
        st.sidebar.write(f"**{user_info['name']}**")
        st.sidebar.write(f"📧 {user_info['email']}")
        st.sidebar.write(f"📅 Joined: {user_info['join_date']}")

    st.sidebar.title("Navigation")
    st.sidebar.write("🔄 **Switch Pages**:")
    page = st.sidebar.radio("Go to", ["Home", "Recipes"])
    return page

# CSS for Background
def add_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3D%2522recipe%2Bbackground%2522&psig=AOvVaw175rQKZo9abCzFGp3lCye8&ust=1731233177590000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCKiyy7qAz4kDFQAAAAAdAAAAABAE');  /* Replace with your background image URL */
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)

# Main App Flow
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

add_custom_css()  # Apply custom CSS

# Login/Register Choice
if not st.session_state["authenticated"]:
    auth_choice = st.sidebar.radio("Choose Action", ["Login", "Register"])
    if auth_choice == "Login":
        login()
    elif auth_choice == "Register":
        register()
else:
    # Show Profile Sidebar & Page Selection
    selected_page = sidebar_profile()

    # Display Selected Page Content
    if selected_page == "Home":
        st.title("Home 🏠")
        st.write("Welcome to the **Smart Recipe Generator**! Discover new recipes based on your ingredients.")
    elif selected_page == "Recipes":
        st.title("Recipes 🍲")
        st.write("Here’s where you can search for recipes or browse recommended dishes.")

    # Logout Button
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
