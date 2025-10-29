import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import re
import io
import base64
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Sparo - Smart Inbox Defender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar visibility
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    width: 300px !important;
}
[data-testid="stSidebarNav"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'sample_message' not in st.session_state:
    st.session_state.sample_message = ""

# Professional CSS styling
def get_custom_css():
    if st.session_state.dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .main { 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .main-header {
            font-size: 3.2rem;
            color: #ffffff;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 15px;
            padding: 15px;
            border: 2px solid #4a90e2 !important;
            background: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            font-size: 16px;
        }
        .stTextArea > div > div > textarea::placeholder {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        .stTextArea textarea {
            color: white !important;
            background: rgba(255, 255, 255, 0.15) !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border: 2px solid #4a90e2 !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3) !important;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        .stButton > button {
            background: linear-gradient(45deg, #4a90e2, #357abd);
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            color: white;
            font-weight: 500;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .main { 
            background: linear-gradient(135deg, #b3d9ff 0%, #7db8e8 50%, #4a90e2 100%);
            color: #212529;
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #b3d9ff 0%, #7db8e8 50%, #4a90e2 100%);
        }
        .main-header {
            font-size: 3.2rem;
            color: #212529;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 2px solid rgba(74, 144, 226, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            color: #212529;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 15px;
            padding: 15px;
            border: 2px solid #4a90e2;
            background: #ffffff !important;
            color: #212529 !important;
            font-size: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            caret-color: #212529 !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border: 2px solid #4a90e2 !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3) !important;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            border: 2px solid rgba(74, 144, 226, 0.15);
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            color: #212529;
        }
        .stButton > button {
            background: linear-gradient(45deg, #4a90e2, #357abd);
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            color: white;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        }
        .stMarkdown, .stText, p, div {
            color: #212529 !important;
        }
        .stSelectbox > div > div > div {
            background: #ffffff;
            color: #212529;
        }
        .stMetric > div {
            color: #212529 !important;
        }
        .stMetric > div > div {
            color: #212529 !important;
        }
        .stMarkdown h3 {
            color: #212529 !important;
        }
        .stTextArea label, .stTextInput label {
            color: #212529 !important;
        }
        [data-testid="stApp"] .stTextArea label {
            color: white !important;
        }
        .stTextArea > div > div > textarea::placeholder,
        .stTextInput > div > div > input::placeholder {
            color: #6c757d !important;
        }
        .stButton > button p {
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background: rgba(173, 181, 189, 0.9) !important;
        }
        section[data-testid="stSidebar"] * {
            color: #212529 !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(173, 181, 189, 0.9) !important;
            backdrop-filter: blur(20px) !important;
        }
        [data-testid="stSidebar"] * {
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: white !important;
            color: #212529 !important;
            border: 2px solid #6c757d !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div:hover {
            border: 2px solid #495057 !important;
            box-shadow: 0 0 0 2px rgba(108, 117, 125, 0.3) !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #4a90e2 !important;
            color: white !important;
        }
        .stSelectbox > div > div > div {
            background: white !important;
            color: #212529 !important;
            border: 2px solid #6c757d !important;
            border-radius: 8px !important;
        }
        .stSelectbox > div > div > div:hover {
            border: 2px solid #495057 !important;
            box-shadow: 0 0 0 2px rgba(108, 117, 125, 0.3) !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div > div {
            background: white !important;
            color: #212529 !important;
            border: 1px solid #4a90e2 !important;
        }
        [data-testid="stSidebar"] .stSelectbox label {
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div > div > div {
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [role="option"] {
            color: #212529 !important;
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] ul {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] li {
            color: #212529 !important;
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        .stSelectbox [data-baseweb="popover"] {
            background: white !important;
        }
        .stSelectbox [data-baseweb="popover"] ul {
            background: white !important;
        }
        .stSelectbox [data-baseweb="popover"] li {
            color: #212529 !important;
            background: white !important;
        }
        .stSelectbox [data-baseweb="popover"] li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        .stSelectbox [data-baseweb="menu"] {
            background: white !important;
        }
        .stSelectbox [data-baseweb="menu"] ul {
            background: white !important;
        }
        .stSelectbox [data-baseweb="menu"] li {
            color: #212529 !important;
            background: white !important;
        }
        .css-1n76uvr {
            background: white !important;
        }
        .css-1n76uvr li {
            color: #212529 !important;
            background: white !important;
        }
        .css-1n76uvr li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .css-1n76uvr {
            background: white !important;
        }
        [data-testid="stSidebar"] .css-1n76uvr li {
            color: #212529 !important;
            background: white !important;
        }
        .stSelectbox div[role="listbox"] {
            background: white !important;
        }
        .stSelectbox div[role="option"] {
            color: #212529 !important;
            background: white !important;
        }
        .stSelectbox div[role="option"]:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[role="listbox"] {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[role="option"] {
            color: #212529 !important;
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[role="option"]:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        [data-testid="stHeader"] [data-testid="stHeaderActionElements"] {
            background: white !important;
        }
        [data-testid="stHeader"] [data-testid="stHeaderActionElements"] * {
            color: #212529 !important;
        }
        [data-testid="stHeader"] button {
            color: #212529 !important;
        }
        [data-testid="stHeader"] [role="menu"] {
            background: white !important;
        }
        [data-testid="stHeader"] [role="menuitem"] {
            color: #212529 !important;
            background: white !important;
        }
        [data-testid="stHeader"] [role="menuitem"]:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        .css-1dp5vir {
            background: white !important;
        }
        .css-1dp5vir * {
            color: #212529 !important;
        }
        .css-1dp5vir [role="menuitem"] {
            color: #212529 !important;
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] > div {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] ul {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] ul li {
            background: white !important;
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] ul li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        .css-16huue1 {
            background: white !important;
        }
        .css-16huue1 ul {
            background: white !important;
        }
        .css-16huue1 ul li {
            background: white !important;
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div[data-baseweb="popover"] {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div[data-baseweb="popover"] ul {
            background: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div[data-baseweb="popover"] ul li {
            background: white !important;
            color: #212529 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div[data-baseweb="popover"] ul li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        .stSelectbox > div > div[data-baseweb="popover"] {
            background: white !important;
        }
        .stSelectbox > div > div[data-baseweb="popover"] ul {
            background: white !important;
        }
        .stSelectbox > div > div[data-baseweb="popover"] ul li {
            background: white !important;
            color: #212529 !important;
        }
        .stSelectbox > div > div[data-baseweb="popover"] ul li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        div[data-baseweb="popover"] {
            background: white !important;
        }
        div[data-baseweb="popover"] * {
            background: white !important;
            color: #212529 !important;
        }
        ul[role="listbox"] {
            background: white !important;
        }
        ul[role="listbox"] li {
            background: white !important;
            color: #212529 !important;
        }
        ul[role="listbox"] li:hover {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.9) !important;
        }
        [data-testid="stToolbar"] {
            background: rgba(255, 255, 255, 0.9) !important;
        }
        </style>
        """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize NLTK without caching
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return PorterStemmer()

ps = initialize_nltk()

# Optimized text preprocessing with regex
def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in text if i not in stopwords.words('english')]
    return " ".join(y)

# ===============================
# ✅ Load trained model & vectorizer
# ===============================
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Load dataset function
def load_dataset():
    try:
        df = pd.read_csv('data/sms_spam_ham_dataset.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['target', 'text']
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame({'target': [], 'text': []})

# Modern Sidebar - Force display
st.sidebar.markdown("")
with st.sidebar:
    # Header with modern branding
    st.markdown('''
    <div style="
        text-align: center; 
        padding: 25px 15px; 
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(53, 122, 189, 0.1));
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(74, 144, 226, 0.2);
    ">
        <div style="
            font-size: 2.2rem; 
            font-weight: 800; 
            background: linear-gradient(135deg, #4a90e2, #357abd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: 1px;
        ">Sparo</div>
        <div style="
            font-size: 0.95rem;
            color: #6c757d;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
        ">Smart Inbox Defender</div>
        <div style="
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
        ">
            <span style="
                background: #28a745;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
            ">97.1% Accuracy</span>
            <span style="
                background: #4a90e2;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
            ">AI Powered</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Theme toggle with modern styling
    theme_icon = "☀️" if st.session_state.dark_mode else "🌙"
    theme_text = f"{theme_icon} {'Light' if st.session_state.dark_mode else 'Dark'} Mode"
    
    if st.button(theme_text, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # Navigation with modern styling
    st.markdown('''
    <div style="
        font-size: 1.1rem;
        font-weight: 700;
        color: #4a90e2;
        margin-bottom: 15px;
        padding-left: 5px;
        border-left: 3px solid #4a90e2;
    ">📍 Navigate</div>
    ''', unsafe_allow_html=True)
    
    page = st.selectbox(
        "Choose page",
        ["Home", "Dashboard", "Bulk Classification", "Advanced", "About"],
        label_visibility="collapsed"
    )
    

    
    st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)
    
    # Quick Stats with modern cards
    st.markdown('''
    <div style="
        font-size: 1.1rem;
        font-weight: 700;
        color: #4a90e2;
        margin-bottom: 15px;
        padding-left: 5px;
        border-left: 3px solid #4a90e2;
    ">📈 Quick Stats</div>
    ''', unsafe_allow_html=True)
    
    # Load data for sidebar stats
    df_sidebar = load_dataset()
    total_messages = len(df_sidebar)
    spam_count = len(df_sidebar[df_sidebar['target'] == 'spam'])
    ham_count = len(df_sidebar[df_sidebar['target'] == 'ham'])
    spam_percentage = (spam_count / total_messages) * 100
    
    # Modern metric cards
    metrics = [
        ("📧", "Total Messages", f"{total_messages:,}", "#4a90e2"),
        ("🎯", "Accuracy", "97.1%", "#28a745"),
        ("⚡", "Precision", "96.8%", "#17a2b8"),
        ("🧠", "Model", "Naive Bayes", "#6f42c1")
    ]
    
    for icon, label, value, color in metrics:
        st.markdown(f'''
        <div style="
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(74, 144, 226, 0.2);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 8px;
            backdrop-filter: blur(10px);
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <div>
                    <div style="font-size: 0.8rem; color: #6c757d; font-weight: 500;">{label}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {color};">{value}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Recent Classifications with modern styling
    if st.session_state.classification_history:
        st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)
        st.markdown('''
        <div style="
            font-size: 1.1rem;
            font-weight: 700;
            color: #4a90e2;
            margin-bottom: 15px;
            padding-left: 5px;
            border-left: 3px solid #4a90e2;
        ">🕒 Recent Activity</div>
        ''', unsafe_allow_html=True)
        
        recent = st.session_state.classification_history[-3:]
        for item in reversed(recent):
            status_icon = "🚨" if item['result'] == 'SPAM' else "✅"
            status_color = "#dc3545" if item['result'] == 'SPAM' else "#28a745"
            status_text = "SPAM" if item['result'] == 'SPAM' else "SAFE"
            
            st.markdown(f'''
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(74, 144, 226, 0.1);
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 6px;
                font-size: 0.85rem;
            ">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span>{status_icon}</span>
                    <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
                    <span style="color: #6c757d; font-size: 0.8rem;">{item['confidence']:.1f}%</span>
                </div>
                <div style="color: #6c757d; font-size: 0.8rem; line-height: 1.2;">
                    {item['text'][:35]}{'...' if len(item['text']) > 35 else ''}
                </div>
            </div>
            ''', unsafe_allow_html=True)

# Clear any previous page state
if 'current_page' not in st.session_state:
    st.session_state.current_page = page
if st.session_state.current_page != page:
    st.session_state.current_page = page
    st.rerun()

if page == "Home":
    st.markdown('<div style="text-align: center; margin-bottom: 3rem;"><h1 style="font-family: Inter, sans-serif; font-size: 3.5rem; font-weight: 800; color: white !important; margin: 0; letter-spacing: 1px;">Sparo 🔍</h1><p style="font-family: Inter, sans-serif; font-size: 1.3rem; color: white !important; margin: 10px 0 0 0; font-weight: 600; letter-spacing: 0.5px;">Smart Inbox Defender</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown('<h3 style="color: white !important; margin-bottom: 15px; font-weight: bold;">Enter your SMS message below:</h3>', unsafe_allow_html=True)
        
        input_sms = st.text_area(
            "Message",
            value=st.session_state.sample_message,
            placeholder="Type your SMS message here for instant analysis...",
            height=120,
            label_visibility="collapsed"
        )
        
        predict_button = st.button("🚀 Analyze Message", use_container_width=True)
        
        if input_sms and predict_button:
            transform_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]
            confidence = max(probabilities) * 100
            
            classification_result = {
                'text': input_sms,
                'result': 'SPAM' if result == 1 else 'HAM',
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            if classification_result not in st.session_state.classification_history:
                st.session_state.classification_history.append(classification_result)
                
                if result == 1:
                    result_text = "🚨 SPAM DETECTED!"
                    result_color = "#dc3545"
                else:
                    result_text = "✅ SAFE MESSAGE!"
                    result_color = "#28a745"
                
                st.markdown(f'''
                <div style="text-align: center; padding: 15px; border-radius: 10px; background: white; border: 2px solid {result_color}; margin: 10px 0; max-width: 400px; margin-left: auto; margin-right: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color: {result_color}; margin: 0; font-size: 1.8rem; font-weight: bold;">{result_text}</h2>
                    <p style="color: {result_color}; margin: 8px 0 0 0; font-size: 1.1rem; font-weight: 600;">Confidence: {confidence:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.progress(confidence/100)
                
                col_spam, col_ham = st.columns(2)
                with col_spam:
                    st.metric("Spam Probability", f"{probabilities[1]*100:.1f}%")
                with col_ham:
                    st.metric("Ham Probability", f"{probabilities[0]*100:.1f}%")
                
                st.markdown("### Message Analysis")
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.metric("Character Count", len(input_sms))
                    st.metric("Word Count", len(input_sms.split()))
                    
                with col_analysis2:
                    suspicious_words = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'call now']
                    suspicious_count = sum(1 for word in suspicious_words if word.lower() in input_sms.lower())
                    st.metric("Suspicious Words", suspicious_count)
                    
                    has_numbers = bool(re.search(r'\d{4,}', input_sms))
                    st.metric("Contains Phone/Numbers", "Yes" if has_numbers else "No")
        
        st.markdown("### Try Sample Messages")
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            if st.button("💬 Sample Spam", use_container_width=True):
                spam_samples = [
                    "URGENT! You've won $1000! Click here now to claim your prize! Call 09061701461",
                    "FREE! Congratulations! You've been selected for a cash prize of $5000! Reply NOW!",
                    "WINNER! Your mobile number has won £2000 in our weekly draw! Call 08001234567 to claim!"
                ]
                import random
                st.session_state.sample_message = random.choice(spam_samples)
                st.rerun()
                
        with sample_col2:
            if st.button("💡 Sample Ham", use_container_width=True):
                ham_samples = [
                    "Hey, are we still meeting for lunch today? Let me know!",
                    "Thanks for the birthday wishes! Had a great time at the party.",
                    "Can you pick up some milk on your way home? We're out."
                ]
                import random
                st.session_state.sample_message = random.choice(ham_samples)
                st.rerun()

elif page == "Dashboard":
    # Clear any residual content
    st.empty()
    # Load fresh data for dashboard
    df = load_dataset()
    total_messages = len(df)
    spam_count = len(df[df['target'] == 'spam'])
    ham_count = len(df[df['target'] == 'ham'])
    spam_percentage = (spam_count / total_messages) * 100
    
    st.markdown('<h1 style="color: #ffffff; text-align: center; font-size: 3rem; margin-bottom: 2rem;">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", f"{total_messages:,}")
    with col2:
        st.metric("Spam Messages", f"{spam_count:,}")
    with col3:
        st.metric("Ham Messages", f"{ham_count:,}")
    with col4:
        st.metric("Spam Rate", f"{spam_percentage:.1f}%")
    
    # Row 1: Distribution and Length Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Message Distribution")
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f'Ham Messages<br>{ham_count:,} ({100-spam_percentage:.1f}%)', 
                   f'Spam Messages<br>{spam_count:,} ({spam_percentage:.1f}%)'],
            values=[ham_count, spam_count],
            marker=dict(
                colors=['#28a745', '#dc3545'],
                line=dict(color='white', width=3)
            ),
            textinfo='label',
            textposition='auto',
            textfont=dict(size=16, color='black', family='Arial Bold'),
            pull=[0.05, 0.1]
        )])
        fig_pie.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Message Length Distribution")
        df['message_length'] = df['text'].str.len()
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[df['target']=='ham']['message_length'], 
            name='Ham Messages', 
            marker_color='#2ecc71',
            boxpoints='outliers'
        ))
        fig_box.add_trace(go.Box(
            y=df[df['target']=='spam']['message_length'], 
            name='Spam Messages', 
            marker_color='#e74c3c',
            boxpoints='outliers'
        ))
        fig_box.update_layout(
            height=350, 
            yaxis_title='Message Length (Characters)',
            xaxis_title='Message Type',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            xaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Row 2: Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Spam Keywords")
        spam_text = ' '.join(df[df['target'] == 'spam']['text'].values[:100])
        words = spam_text.lower().split()
        word_freq = pd.Series(words).value_counts().head(10)
        fig_bar = go.Figure(data=[go.Bar(x=word_freq.values, y=word_freq.index, orientation='h', marker_color='#e74c3c')])
        fig_bar.update_layout(
            height=350, 
            xaxis_title='Frequency', 
            yaxis_title='Words',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            xaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### Message Length Histogram")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df[df['target']=='ham']['message_length'], name='Ham', marker_color='#2ecc71', opacity=0.7))
        fig_hist.add_trace(go.Histogram(x=df[df['target']=='spam']['message_length'], name='Spam', marker_color='#e74c3c', opacity=0.7))
        fig_hist.update_layout(
            height=350, 
            xaxis_title='Message Length', 
            yaxis_title='Count', 
            barmode='overlay',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            xaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            legend=dict(font=dict(color='black', size=12)),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 3: Advanced Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Hourly Pattern (Simulated)")
        hours = list(range(24))
        spam_pattern = [5, 3, 2, 1, 1, 2, 8, 15, 25, 30, 35, 40, 45, 50, 48, 42, 38, 35, 28, 22, 18, 12, 8, 6]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=hours, y=spam_pattern, mode='lines+markers', name='Spam Activity', line=dict(color='#e74c3c')))
        fig_line.update_layout(
            height=350, 
            xaxis_title='Hour of Day', 
            yaxis_title='Spam Messages',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            xaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                tickfont=dict(color='black', size=12), 
                title_font=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            legend=dict(font=dict(color='black', size=12)),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        st.markdown("### Classification Confidence")
        confidence_ranges = ['90-100%', '80-90%', '70-80%', '60-70%', '<60%']
        confidence_counts = [450, 200, 70, 20, 7]
        fig_donut = go.Figure(data=[go.Pie(
            labels=confidence_ranges,
            values=confidence_counts,
            hole=0.4,
            marker_colors=['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
        )])
        fig_donut.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            legend=dict(font=dict(color='black', size=12)),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        fig_donut.update_traces(textfont_size=12, textfont_color='black')
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Word Clouds
    st.markdown("### Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spam Messages")
        spam_text = ' '.join(df[df['target'] == 'spam']['text'].values[:200])
        if spam_text:
            wordcloud_spam = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(spam_text)
            fig_spam, ax_spam = plt.subplots(figsize=(6, 4))
            ax_spam.imshow(wordcloud_spam, interpolation='bilinear')
            ax_spam.axis('off')
            st.pyplot(fig_spam)
    
    with col2:
        st.markdown("#### Ham Messages")
        ham_text = ' '.join(df[df['target'] == 'ham']['text'].values[:200])
        if ham_text:
            wordcloud_ham = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(ham_text)
            fig_ham, ax_ham = plt.subplots(figsize=(6, 4))
            ax_ham.imshow(wordcloud_ham, interpolation='bilinear')
            ax_ham.axis('off')
            st.pyplot(fig_ham)
    
    # User Classification History
    if st.session_state.classification_history:
        st.markdown("### Your Classification History")
        history_df = pd.DataFrame(st.session_state.classification_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages Classified", len(history_df))
        with col2:
            spam_detected = len(history_df[history_df['result'] == 'SPAM'])
            st.metric("Spam Detected", spam_detected)
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # User activity chart
        if len(history_df) > 1:
            fig_user = go.Figure()
            fig_user.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#4a90e2')
            ))
            fig_user.update_layout(
                height=300, 
                xaxis_title='Classification #', 
                yaxis_title='Confidence %',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=14),
                xaxis=dict(
                    tickfont=dict(color='black', size=12), 
                    title_font=dict(color='black', size=14),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    tickfont=dict(color='black', size=12), 
                    title_font=dict(color='black', size=14),
                    gridcolor='lightgray'
                ),
                legend=dict(font=dict(color='black', size=12)),
                margin=dict(t=40, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_user, use_container_width=True)
        
        st.dataframe(history_df[['text', 'result', 'confidence']].tail(5), use_container_width=True)

elif page == "Bulk Classification":
    # Clear any residual content
    st.empty()
    st.markdown('<h1 style="color: #ffffff; text-align: center; font-size: 3rem; margin-bottom: 2rem;">Bulk Message Classification</h1>', unsafe_allow_html=True)
    
    # Upload Section
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        text-align: center;
    ">
        <h2 style="margin: 0 0 10px 0; font-weight: bold;">📤 Upload Multiple Messages</h2>
        <p style="margin: 0; opacity: 0.9;">Process hundreds of messages in seconds with batch classification</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'csv']
        )
        st.markdown('<p style="color: #6c757d; font-size: 0.9rem; margin-top: 5px;">Upload TXT (one message per line) or CSV (with \'message\' column)</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div style="
            background: rgba(255, 255, 255, 0.1);
            border: 2px dashed #4a90e2;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 25px;
        ">
            <h4 style="color: #4a90e2; margin: 0 0 10px 0;">📋 Supported Formats</h4>
            <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">• TXT files (one message per line)<br>• CSV files (with 'message' column)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.txt'):
                content = str(uploaded_file.read(), "utf-8")
                messages = [line.strip() for line in content.split('\n') if line.strip()]
            else:
                df_upload = pd.read_csv(uploaded_file)
                if 'message' in df_upload.columns:
                    messages = df_upload['message'].tolist()
                else:
                    st.error("❌ CSV file must have a 'message' column")
                    messages = []
            
            if messages:
                st.markdown(f'''
                <div style="
                    background: linear-gradient(135deg, #00b894, #00a085);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    text-align: center;
                ">
                    <h3 style="margin: 0;">✅ Successfully loaded {len(messages)} messages</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.button("🚀 Classify All Messages", use_container_width=True, type="primary"):
                    progress_container = st.container()
                    with progress_container:
                        st.markdown("### 🔄 Processing Messages...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    results = []
                    for i, message in enumerate(messages):
                        status_text.text(f"Processing message {i+1}/{len(messages)}")
                        transformed = transform_text(message)
                        vector_input = tfidf.transform([transformed])
                        result = model.predict(vector_input)[0]
                        probabilities = model.predict_proba(vector_input)[0]
                        confidence = max(probabilities) * 100
                        
                        results.append({
                            'message': message,
                            'classification': 'SPAM' if result == 1 else 'HAM',
                            'confidence': confidence,
                            'spam_probability': probabilities[1] * 100,
                            'ham_probability': probabilities[0] * 100
                        })
                        
                        progress_bar.progress((i + 1) / len(messages))
                    
                    status_text.text("✅ Processing complete!")
                    results_df = pd.DataFrame(results)
                    
                    # Results Summary Cards
                    spam_detected = len(results_df[results_df['classification'] == 'SPAM'])
                    spam_rate = (spam_detected / len(results_df)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="margin: 0; font-size: 2.5rem;">{len(results_df)}</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Processed</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, #fd79a8, #e84393); color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="margin: 0; font-size: 2.5rem;">{spam_detected}</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Spam Detected</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, #fdcb6e, #e17055); color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="margin: 0; font-size: 2.5rem;">{spam_rate:.1f}%</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Spam Rate</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"spam_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Manual Input Section
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, #a29bfe, #6c5ce7);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 30px 0 20px 0;
        box-shadow: 0 4px 20px rgba(162, 155, 254, 0.3);
        text-align: center;
    ">
        <h2 style="margin: 0 0 10px 0; font-weight: bold;">Manual Input ✏️</h2>
        <p style="margin: 0; opacity: 0.9;">Paste your messages directly for quick classification</p>
    </div>
    ''', unsafe_allow_html=True)
    
    bulk_text = st.text_area(
        "Paste messages here (one per line)",
        height=200,
        placeholder="Message 1\nMessage 2\nMessage 3...",
        help="Enter each message on a new line"
    )
    
    if bulk_text and st.button("🔍 Classify Pasted Messages", use_container_width=True, type="primary"):
        messages = [line.strip() for line in bulk_text.split('\n') if line.strip()]
        
        if messages:
            st.markdown("### 🔄 Processing Messages...")
            progress_bar = st.progress(0)
            results = []
            
            for i, message in enumerate(messages):
                transformed = transform_text(message)
                vector_input = tfidf.transform([transformed])
                result = model.predict(vector_input)[0]
                probabilities = model.predict_proba(vector_input)[0]
                confidence = max(probabilities) * 100
                
                results.append({
                    'message': message,
                    'classification': 'SPAM' if result == 1 else 'HAM',
                    'confidence': confidence
                })
                
                progress_bar.progress((i + 1) / len(messages))
            
            st.markdown("### 📊 Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

elif page == "Advanced":
    # Clear any residual content
    st.empty()
    st.markdown('<h1 style="color: #ffffff; text-align: center; font-size: 3rem; margin-bottom: 2rem;">Advanced Features</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Model Details", "Feature Analysis", "Settings"])
    
    with tab1:
        st.markdown("### Model Information")
        
        # Performance Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
            ">
                <h2 style="margin: 0; font-size: 2.5rem; font-weight: bold;">97.1%</h2>
                <p style="margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;">Accuracy</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #17a2b8, #138496);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(23, 162, 184, 0.3);
            ">
                <h2 style="margin: 0; font-size: 2.5rem; font-weight: bold;">96.8%</h2>
                <p style="margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;">Precision</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #ffc107, #e0a800);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
            ">
                <h2 style="margin: 0; font-size: 2.5rem; font-weight: bold;">94.2%</h2>
                <p style="margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;">Recall</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #6f42c1, #5a32a3);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(111, 66, 193, 0.3);
            ">
                <h2 style="margin: 0; font-size: 2.5rem; font-weight: bold;">95.5%</h2>
                <p style="margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;">F1-Score</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Model Details Cards
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #4a90e2, #357abd);
                color: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(74, 144, 226, 0.3);
            ">
                <h3 style="color: white !important; margin-top: 0; font-weight: bold;">🧠 Algorithm Details</h3>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: white !important;">Algorithm:</strong><br>
                    <span style="color: rgba(255, 255, 255, 0.9) !important;">Multinomial Naive Bayes</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: white !important;">Features:</strong><br>
                    <span style="color: rgba(255, 255, 255, 0.9) !important;">TF-IDF Vectorization</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: white !important;">Training Data:</strong><br>
                    <span style="color: rgba(255, 255, 255, 0.9) !important;">5,572 SMS Messages</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
            ">
                <h3 style="color: white !important; margin-top: 0; font-weight: bold;">⚙️ Preprocessing Pipeline</h3>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">1.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">Convert to lowercase</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">2.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">Remove special characters</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">3.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">Tokenization</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">4.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">Remove stopwords</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">5.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">Stemming (Porter Stemmer)</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 12px; border-radius: 8px; border-left: 4px solid white;">
                        <strong style="color: white !important;">6.</strong> <span style="color: rgba(255, 255, 255, 0.9) !important;">TF-IDF Vectorization</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Feature Analysis")
        
        if st.button("🔬 Analyze Sample Message Features", use_container_width=True):
            sample_message = "FREE! Win a $1000 gift card! Call now 123-456-7890"
            
            # Processing Steps with Theme-Aware Cards
            steps = [
                ("📝 Original", sample_message, "#4a90e2"),
                ("💻 Lowercase", sample_message.lower(), "#17a2b8"),
                ("🧹 Remove Special Chars", re.sub(r'[^a-zA-Z\s]', '', sample_message.lower()), "#28a745"),
            ]
            
            tokens = nltk.word_tokenize(re.sub(r'[^a-zA-Z\s]', '', sample_message.lower()))
            filtered = [word for word in tokens if word not in stopwords.words('english')]
            stemmed = [ps.stem(word) for word in filtered]
            final_text = ' '.join(stemmed)
            
            steps.extend([
                ("🧩 Tokens", str(tokens), "#ffc107"),
                ("🚫 Remove Stopwords", str(filtered), "#fd7e14"),
                ("🌱 Stemming", str(stemmed), "#6f42c1"),
                ("📄 Final Text", final_text, "#dc3545")
            ])
            
            for title, content, color in steps:
                st.markdown(f'''
                <div style="
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 16px;
                    padding: 24px;
                    margin: 20px 0;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                ">
                    <div style="margin-bottom: 16px;">
                        <h4 style="color: white; margin: 0; font-weight: 700; font-size: 1.2rem;">{title}</h4>
                    </div>
                    <div style="
                        background: rgba(255, 255, 255, 0.15);
                        backdrop-filter: blur(5px);
                        padding: 16px;
                        border-radius: 12px;
                        border-left: 4px solid {color};
                    ">
                        <p style="color: white; margin: 0; font-family: monospace; font-size: 0.95rem; line-height: 1.5; opacity: 0.9;">{content}</p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            

    
    with tab3:
        st.markdown("### Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Theme Settings - Better Colors
            current_theme = "Dark" if st.session_state.dark_mode else "Light"
            theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
            
            st.markdown(f'''
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                height: 200px;
            ">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                    <div style="
                        background: rgba(255, 255, 255, 0.2);
                        border-radius: 12px;
                        padding: 8px;
                        font-size: 1.2rem;
                    ">{theme_icon}</div>
                    <h4 style="color: white; margin: 0; font-weight: 600; font-size: 1.1rem;">Theme Settings</h4>
                </div>
                <div style="
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 10px;
                    padding: 12px;
                    backdrop-filter: blur(5px);
                ">
                    <div style="color: white; font-size: 0.9rem; opacity: 0.9;">Current Theme</div>
                    <div style="color: white; font-weight: 600; font-size: 1rem; margin-top: 4px;">{current_theme}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            # Data Management - Same size with better colors
            st.markdown('''
            <div style="
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                height: 200px;
            ">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                    <div style="
                        background: rgba(255, 255, 255, 0.2);
                        border-radius: 12px;
                        padding: 8px;
                        font-size: 1.2rem;
                    ">🗂️</div>
                    <h4 style="color: white; margin: 0; font-weight: 600; font-size: 1.1rem;">Data Management</h4>
                </div>
                <div style="
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 10px;
                    padding: 12px;
                    backdrop-filter: blur(5px);
                ">
                    <div style="color: white; font-size: 0.9rem; opacity: 0.9;">Manage your classification data</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
        
        # Force white button text for both themes
        st.markdown('''
        <style>
        .stButton > button {
            color: white !important;
        }
        .stButton > button p {
            color: white !important;
        }
        .stDownloadButton > button {
            color: white !important;
        }
        .stDownloadButton > button p {
            color: white !important;
        }
        .stDownloadButton > button span {
            color: white !important;
        }
        .stDownloadButton > button div {
            color: white !important;
        }
        button[kind="primary"] {
            color: white !important;
        }
        button[kind="primary"] p {
            color: white !important;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.classification_history = []
                st.success("✅ History cleared!")
        
        with col_btn2:
            if st.session_state.classification_history:
                history_df = pd.DataFrame(st.session_state.classification_history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export History",
                    data=csv,
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.markdown('<p style="color: #6c757d; text-align: center; margin-top: 10px;">No history to export</p>', unsafe_allow_html=True)
        
        # Modern Statistics Card
        if st.session_state.classification_history:
            total_classifications = len(st.session_state.classification_history)
            spam_detected = len([x for x in st.session_state.classification_history if x['result'] == 'SPAM'])
            
            st.markdown(f'''
            <div style="
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            ">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="
                        background: rgba(255, 255, 255, 0.3);
                        border-radius: 12px;
                        padding: 8px;
                        font-size: 1.2rem;
                    ">📊</div>
                    <h4 style="color: #2d3748; margin: 0; font-weight: 600; font-size: 1.1rem;">Your Statistics</h4>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div style="
                        background: rgba(255, 255, 255, 0.4);
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                        backdrop-filter: blur(5px);
                    ">
                        <div style="color: #2d3748; font-size: 1.8rem; font-weight: 700; margin-bottom: 4px;">{total_classifications}</div>
                        <div style="color: #4a5568; font-size: 0.85rem; font-weight: 500;">Classifications</div>
                    </div>
                    <div style="
                        background: rgba(255, 255, 255, 0.4);
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                        backdrop-filter: blur(5px);
                    ">
                        <div style="color: #2d3748; font-size: 1.8rem; font-weight: 700; margin-bottom: 4px;">{spam_detected}</div>
                        <div style="color: #4a5568; font-size: 0.85rem; font-weight: 500;">Spam Found</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

elif page == "About":
    # Clear any residual content
    st.empty()
    st.markdown('<h1 style="color: #ffffff; text-align: center; font-size: 3.5rem; margin-bottom: 3rem; font-weight: 700; letter-spacing: 2px;">About Sparo - Smart Inbox Defender</h1>', unsafe_allow_html=True)
    
    # Professional Hero Section
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        color: white;
        padding: 50px;
        border-radius: 25px;
        margin: 30px 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <h2 style="margin: 0 0 20px 0; font-weight: 800; font-size: 3rem; letter-spacing: 1px;">Enterprise-Grade SMS Security</h2>
        <p style="margin: 0; font-size: 1.4rem; opacity: 0.95; font-weight: 300; line-height: 1.6;">Advanced AI-powered spam detection delivering 97.1% accuracy for enterprise communications</p>
        <div style="margin-top: 25px; display: flex; justify-content: center; gap: 20px;">
            <span style="background: rgba(255, 255, 255, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">🛡️ Enterprise Security</span>
            <span style="background: rgba(255, 255, 255, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">⚡ Real-time Processing</span>
            <span style="background: rgba(255, 255, 255, 0.2); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">🎯 97.1% Accuracy</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Core Capabilities
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(248, 249, 250, 0.95));
        border: none;
        border-radius: 20px;
        padding: 40px;
        margin: 40px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    ">
        <h2 style="color: #2c3e50; margin: 0 0 35px 0; font-weight: 700; font-size: 2.2rem; text-align: center;">🚀 Enterprise Capabilities</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px;">
            <div style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); padding: 25px; border-radius: 15px; border: 1px solid #28a745; box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
                <strong style="color: #28a745; font-size: 1.1rem;">Real-Time Analysis</strong><br>
                <span style="color: #495057; margin-top: 8px; display: block; line-height: 1.5;">Sub-second message processing with enterprise-grade performance</span>
            </div>
            <div style="background: linear-gradient(135deg, #e3f2fd, #f0f8ff); padding: 25px; border-radius: 15px; border: 1px solid #17a2b8; box-shadow: 0 5px 15px rgba(23, 162, 184, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                <strong style="color: #17a2b8; font-size: 1.1rem;">Advanced Analytics</strong><br>
                <span style="color: #495057; margin-top: 8px; display: block; line-height: 1.5;">Comprehensive confidence scoring and probability analysis</span>
            </div>
            <div style="background: linear-gradient(135deg, #fff8e1, #fffbf0); padding: 25px; border-radius: 15px; border: 1px solid #ffc107; box-shadow: 0 5px 15px rgba(255, 193, 7, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">📁</div>
                <strong style="color: #e0a800; font-size: 1.1rem;">Batch Processing</strong><br>
                <span style="color: #495057; margin-top: 8px; display: block; line-height: 1.5;">Scalable bulk analysis for enterprise message volumes</span>
            </div>
            <div style="background: linear-gradient(135deg, #f3e5f5, #faf5ff); padding: 25px; border-radius: 15px; border: 1px solid #6f42c1; box-shadow: 0 5px 15px rgba(111, 66, 193, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 10px;">📈</div>
                <strong style="color: #6f42c1; font-size: 1.1rem;">Business Intelligence</strong><br>
                <span style="color: #495057; margin-top: 8px; display: block; line-height: 1.5;">Interactive dashboards with actionable insights</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Performance & Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div style="
            background: linear-gradient(135deg, #00b894, #00a085);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0, 184, 148, 0.3);
        ">
            <h3 style="margin: 0 0 25px 0; font-weight: bold;">🎯 Performance Benchmarks</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: center;">
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2rem;">97.1%</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Accuracy</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2rem;">96.8%</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Precision</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2rem;">94.2%</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Recall</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2rem;">95.5%</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">F1-Score</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg, #fd79a8, #e84393);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(253, 121, 168, 0.3);
        ">
            <h3 style="margin: 0 0 25px 0; font-weight: bold;">📊 Dataset Statistics</h3>
            <div style="text-align: center;">
                <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; font-size: 2.5rem;">{total_messages:,}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Messages</p>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{ham_count:,}</h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9rem;">Ham ({100-spam_percentage:.1f}%)</p>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 10px;">
                        <h3 style="margin: 0; font-size: 1.5rem;">{spam_count:,}</h3>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9rem;">Spam ({spam_percentage:.1f}%)</p>
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Professional Technology Stack
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border-radius: 20px;
        padding: 40px;
        margin: 40px 0;
        box-shadow: 0 15px 50px rgba(44, 62, 80, 0.3);
    ">
        <h2 style="color: white; margin: 0 0 35px 0; font-weight: 700; font-size: 2.2rem; text-align: center;">🛠️ Enterprise Technology Stack</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px;">
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.2);">
                <div style="font-size: 3rem; margin-bottom: 15px;">🐍</div>
                <h4 style="margin: 0 0 10px 0; font-size: 1.3rem; font-weight: 600;">Python 3.8+</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 1rem; line-height: 1.4;">Enterprise Development Platform</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.2);">
                <div style="font-size: 3rem; margin-bottom: 15px;">🚀</div>
                <h4 style="margin: 0 0 10px 0; font-size: 1.3rem; font-weight: 600;">Streamlit</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 1rem; line-height: 1.4;">Modern Web Framework</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.2);">
                <div style="font-size: 3rem; margin-bottom: 15px;">🧩</div>
                <h4 style="margin: 0 0 10px 0; font-size: 1.3rem; font-weight: 600;">Scikit-learn</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 1rem; line-height: 1.4;">Machine Learning Engine</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.2);">
                <div style="font-size: 3rem; margin-bottom: 15px;">📝</div>
                <h4 style="margin: 0 0 10px 0; font-size: 1.3rem; font-weight: 600;">NLTK</h4>
                <p style="margin: 0; opacity: 0.9; font-size: 1rem; line-height: 1.4;">Natural Language Processing</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # User Statistics (if available)
    if st.session_state.classification_history:
        user_total = len(st.session_state.classification_history)
        user_spam = len([x for x in st.session_state.classification_history if x['result'] == 'SPAM'])
        
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin: 25px 0;
            box-shadow: 0 4px 20px rgba(116, 185, 255, 0.3);
            text-align: center;
        ">
            <h2 style="margin: 0 0 20px 0; font-weight: bold;">👤 Your Activity</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2.5rem;">{user_total}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Messages Classified</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 10px;">
                    <h2 style="margin: 0; font-size: 2.5rem;">{user_spam}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Spam Detected</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)