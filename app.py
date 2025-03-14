import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Streamlit App Title and Configuration
st.set_page_config(
    page_title="CryptoPulse: Cryptocurrency Trends",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stSidebar {
        background-color: #2e3b4e;
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stFileUploader {
        background-color: white;
        border-radius: 5px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2e3b4e;
    }
    .stTable {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dataframe th {
        background-color: #4CAF50;
        color: white;
    }
    .dataframe td {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("📊 CryptoPulse: Cryptocurrency Price Trends")
st.markdown("""
    <div style="background-color: #4CAF50; padding: 10px; border-radius: 5px;">
        <h3 style="color: white;">Your Gateway to Cryptocurrency Insights 🚀</h3>
    </div>
    """, unsafe_allow_html=True)

# Function to load and process uploaded data
def load_data(uploaded_files):
    data_dict = {}
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.strip().str.lower()  # Standardize column names
            
            # Convert Date
            data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
            
            # Remove commas and convert numeric columns
            for col in ["price", "open", "high", "low"]:
                data[col] = data[col].astype(str).str.replace(",", "").astype(float)
            
            # Convert Volume (K/M/B to actual numbers)
            def convert_volume(value):
                if isinstance(value, str):
                    if "K" in value:
                        return float(value.replace("K", "")) * 1e3
                    elif "M" in value:
                        return float(value.replace("M", "")) * 1e6
                    elif "B" in value:
                        return float(value.replace("B", "")) * 1e9
                return float(value)
            
            data["vol."] = data["vol."].apply(convert_volume)
            
            # Convert Change %
            data["change %"] = data["change %"].str.replace("%", "").astype(float)
            
            data = data.dropna(subset=["date"])  # Remove invalid dates
            data = data.iloc[::-1]  # Reverse order
            
            # Extract cryptocurrency name from filename
            crypto_name = uploaded_file.name.split('.')[0].capitalize()
            data_dict[crypto_name] = data
    return data_dict

# File Upload for Multiple Cryptocurrencies
st.sidebar.subheader("📂 Upload Cryptocurrency Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV Files (Up to 3)",
    type=["csv"],
    accept_multiple_files=True,
    help="Ensure your CSV files contain columns for date, price, open, high, low, volume, and change percentage."
)

# Load and Display Data
if uploaded_files:
    data_dict = load_data(uploaded_files)
    
    if data_dict:
        st.sidebar.subheader("🔍 Choose the Analysis")
        selected_option = st.sidebar.selectbox(
            "Select Analysis Type",
            [
                "Candlestick Pattern", 
                "Volume And Price Trend",
                "Cryptocurrency Comparison",
                "LSTM Model",
                "ARIMA Model",
                "Integrated Model"
            ],
            help="Select the type of analysis you want to perform."
        )
        
        if selected_option == "Cryptocurrency Comparison":
            st.subheader("📊 Cryptocurrency Comparison")
            col1, col2 = st.columns(2)
            comparison_data = []
            
            for crypto_name, crypto_data in data_dict.items():
                avg_price = crypto_data["price"].mean()
                max_price = crypto_data["price"].max()
                min_price = crypto_data["price"].min()
                vol_avg = crypto_data["vol."].mean()
                change_avg = crypto_data["change %"].mean()
                
                comparison_data.append({
                    "Cryptocurrency": crypto_name,
                    "Avg Closing Price": avg_price,
                    "Max Price": max_price,
                    "Min Price": min_price,
                    "Avg Volume": vol_avg,
                    "Avg % Change": change_avg
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df.style.set_properties(**{
                'background-color': '#f9f9f9',
                'color': '#2e3b4e',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            }))
            
            # Pie Chart for Avg Closing Price Distribution
            with col1:
                fig_pie1 = px.pie(comparison_df, names="Cryptocurrency", values="Avg Closing Price", 
                                title="Market Share by Avg Closing Price 🥧",
                                color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig_pie1)
            
            # Pie Chart for Avg Volume Distribution
            with col2:
                fig_pie2 = px.pie(comparison_df, names="Cryptocurrency", values="Avg Volume", 
                                title="Market Share by Trading Volume 📦",
                                color_discrete_sequence=px.colors.sequential.Plasma)
                st.plotly_chart(fig_pie2)
            
            best_crypto = max(data_dict.keys(), key=lambda x: data_dict[x]["price"].mean())
            st.success(f"📌 Suggested Investment: **{best_crypto}**")
            st.write(f"Based on average closing price and market trends, **{best_crypto}** appears to be the best investment option among the uploaded cryptocurrencies.")

            if len(crypto_data) >= 60:  # Ensure at least 60 rows for decomposition
                period = min(30, len(crypto_data) // 2)  # Adjust period dynamically

            # Decomposing the time series (Trend & Seasonality)
            decomposition = seasonal_decompose(crypto_data["price"], model="additive", period=period)

            # 📈 Trend Graph
            fig_trend = px.line(x=crypto_data["date"], y=decomposition.trend, 
                                title=f"{crypto_name} Trend Analysis 📈",
                                color_discrete_sequence=["#4CAF50"])
            st.plotly_chart(fig_trend)

            # 📊 Seasonality Graph
            fig_seasonality = px.line(x=crypto_data["date"], y=decomposition.seasonal, 
                                    title=f"{crypto_name} Seasonality Analysis 🌊",
                                    color_discrete_sequence=["#FFA500"])
            st.plotly_chart(fig_seasonality)

            # ⚡ Volatility Graph (Rolling Standard Deviation)
            crypto_data["Volatility"] = crypto_data["price"].rolling(window=30).std()

            fig_volatility = px.line(x=crypto_data["date"], y=crypto_data["Volatility"], 
                                    title=f"{crypto_name} Volatility Analysis ⚡",
                                    color_discrete_sequence=["#FF4500"])
            st.plotly_chart(fig_volatility)
        
        else:
            for crypto_name, crypto_data in data_dict.items():
                st.subheader(f"📈 {crypto_name} - {selected_option}")
                
                if selected_option == "Candlestick Pattern":
                    fig = go.Figure(data=[go.Candlestick(
                        x=crypto_data["date"],
                        open=crypto_data["open"],
                        high=crypto_data["high"],
                        low=crypto_data["low"],
                        close=crypto_data["price"],
                    )])
                    fig.update_layout(
                        title=f"{crypto_name} Candlestick Chart 🕯️",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig)
                
                elif selected_option == "Volume And Price Trend":
                    fig = px.bar(crypto_data, x="date", y="vol.", 
                                title=f"{crypto_name} Volume Bar Chart 📊",
                                color_discrete_sequence=["#4CAF50"])
                    st.plotly_chart(fig)
                    fig = px.line(crypto_data, x="date", y="open", 
                                title=f"{crypto_name} Opening Price Trend 📈",
                                color_discrete_sequence=["#FFA500"])
                    st.plotly_chart(fig)
                    fig = px.line(crypto_data, x="date", y="price", 
                                title=f"{crypto_name} Closing Price Trend 📉",
                                color_discrete_sequence=["#FF4500"])
                    st.plotly_chart(fig)
                if selected_option  in ["LSTM Model", "ARIMA Model", "Integrated Model"]:
                    st.subheader(f"📈 {selected_option} Prediction")
                    st.markdown(f"""
                           Cick the link below:
                        - [{selected_option} Notebook](https://colab.research.google.com/drive/18HvI0J2vex6f_Sk6sFs_073LztcTzC4C?usp=sharing)
                    """)
            
            st.markdown("### 📊 Dataset Used for Analysis")
            st.write("You can view and interact with the dataset below:")
        st.subheader("📊 Data Preview")
        for crypto_name, crypto_data in data_dict.items():
            st.write(f"### {crypto_name} Data")
            st.dataframe(crypto_data.head().style.set_properties(**{
                'background-color': '#f9f9f9',
                'color': '#2e3b4e',
                'border-radius': '10px',
                'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            }))

else:
    st.write("Welcome to CryptoPulse! 🚀")
    st.write("To get started, please follow these steps:")
    st.write("1. Use the sidebar on the left to upload your cryptocurrency data files (CSV format).")
    st.write("2. You can upload up to 3 files for different cryptocurrencies.")
    st.write("3. Once uploaded, you'll be able to select various analysis options from the dropdown menu.")
    st.write("4. Explore trends, compare cryptocurrencies, and gain insights from your data!")
    
    st.info("📁 Tip: Prepare your CSV files with columns for date, price, open, high, low, volume, and change percentage.")
    
    # Sample data structure
    st.subheader("Expected CSV Structure:")
    sample_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Price': [50000, 51000, 49000],
        'Open': [49000, 50000, 51000],
        'High': [52000, 53000, 51500],
        'Low': [48000, 49000, 48500],
        'Vol.': ['1.5M', '1.2M', '1.8M'],
        'Change %': ['2%', '-3.92%', '1.5%']
    })
    st.table(sample_data.style.set_properties(**{
        'background-color': '#f9f9f9',
        'color': '#2e3b4e',
        'border-radius': '10px',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
    }))
