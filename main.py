import streamlit as st
from streamlit_option_menu import option_menu
import login, register

st.set_page_config(
       page_title="Welcome",
 )

class MultiApp:
    
    def __init__(self):
        self.apps = []
    def add_app(self,title,function):
        self.apps.append({
            "title":title
            "function":function
        })    
        
    def run():
           
           with st.sidebar:
               app = option_menu(
                   menu_title='Welcome'
                   options=['Login','Register']
                   icons=['person-circle','person-circle']
                   menu_icon='happy-face',
                   default_index=1,
                   styles
               )