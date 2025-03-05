import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Streamlit App Title and Configuration
st.set_page_config(page_title="CryptoPulse: Cryptocurrency Trends", layout="wide")
st.title("📊 CryptoPulse: Cryptocurrency Price Trends")

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
st.sidebar.subheader("Upload Cryptocurrency Data (Up to 3 Files)")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# Load and Display Data
if uploaded_files:
    data_dict = load_data(uploaded_files)
    
    if data_dict:
        st.sidebar.subheader("Choose the Analysis")
        selected_option = st.sidebar.selectbox("Select Analysis Type", [
            "Cryptocurrency Comparison",
            "Candlestick Pattern", 
            "Volume Bar Chart", 
            "Closing Price Trend", 
            "Opening Price Trend", 
            "ARIMA Model Predictions", 
            "Graphs (Trend, Seasonality, Prediction, Volatility)"
        ])
        
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
            st.table(comparison_df)
            
            # Pie Chart for Avg Closing Price Distribution
            with col1:
                fig_pie1 = px.pie(comparison_df, names="Cryptocurrency", values="Avg Closing Price", 
                                title="Market Share by Avg Closing Price")
                st.plotly_chart(fig_pie1)
            
            # Pie Chart for Avg Volume Distribution
            with col2:
                fig_pie2 = px.pie(comparison_df, names="Cryptocurrency", values="Avg Volume", 
                                title="Market Share by Trading Volume")
                st.plotly_chart(fig_pie2)
            
            best_crypto = max(data_dict.keys(), key=lambda x: data_dict[x]["price"].mean())
            st.subheader(f"📌 Suggested Investment: {best_crypto}")
            st.write(f"Based on average closing price and market trends, **{best_crypto}** appears to be the best investment option among the uploaded cryptocurrencies.")
        
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
                    fig.update_layout(title=f"{crypto_name} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig)
                
                elif selected_option == "Volume Bar Chart":
                    fig = px.bar(crypto_data, x="date", y="vol.", title=f"{crypto_name} Volume Bar Chart")
                    st.plotly_chart(fig)
                
                elif selected_option == "Closing Price Trend":
                    fig = px.line(crypto_data, x="date", y="price", title=f"{crypto_name} Closing Price Trend")
                    st.plotly_chart(fig)
                
                elif selected_option == "Opening Price Trend":
                    fig = px.line(crypto_data, x="date", y="open", title=f"{crypto_name} Opening Price Trend")
                    st.plotly_chart(fig)
                
                elif selected_option == "ARIMA Model Predictions":
                    if len(crypto_data) > 50:
                        model = ARIMA(crypto_data["price"], order=(5, 1, 0))
                        arima_result = model.fit()
                        crypto_data["Predicted Price"] = arima_result.predict(start=len(crypto_data)-10, end=len(crypto_data)+10, dynamic=False)
                        
                        # Display Table
                        st.subheader(f"📊 {crypto_name} ARIMA Predictions Table")
                        st.write(crypto_data[["date", "price", "Predicted Price"]].tail(20))
                        
                        # Display Graph
                        fig_prediction = px.line(crypto_data, x="date", y=["price", "Predicted Price"],
                                                title=f"{crypto_name} Price Prediction")
                        st.plotly_chart(fig_prediction)
                    else:
                        st.warning(f"Not enough data to run ARIMA model for {crypto_name}. Minimum 50 data points required.")
                
                elif selected_option == "Graphs (Trend, Seasonality, Prediction, Volatility)":
                    if "price" in crypto_data.columns:
                        # Ensure data is sorted by date
                        crypto_data = crypto_data.sort_values(by="date")

                        # Handle missing values
                        crypto_data["price"] = crypto_data["price"].interpolate()

                        # Check if we have enough data points
                        if len(crypto_data) >= 60:  # Ensure at least 60 rows for decomposition
                            period = min(30, len(crypto_data) // 2)  # Adjust period dynamically

                            # Decomposing the time series (Trend & Seasonality)
                            decomposition = seasonal_decompose(crypto_data["price"], model="additive", period=period)

                            # 📈 Trend Graph
                            fig_trend = px.line(x=crypto_data["date"], y=decomposition.trend, 
                                                title=f"{crypto_name} Trend Analysis")
                            st.plotly_chart(fig_trend)

                            # 📊 Seasonality Graph
                            fig_seasonality = px.line(x=crypto_data["date"], y=decomposition.seasonal, 
                                                    title=f"{crypto_name} Seasonality Analysis")
                            st.plotly_chart(fig_seasonality)

                            # 🔮 ARIMA Model for Prediction
                            model = ARIMA(crypto_data["price"], order=(5, 1, 0))  # (p, d, q) values
                            arima_result = model.fit()
                            crypto_data["Predicted Price"] = arima_result.predict(start=10, end=len(crypto_data), dynamic=False)

                            fig_prediction = px.line(crypto_data, x="date", y=["price", "Predicted Price"], 
                                                    title=f"{crypto_name} Price Prediction")
                            st.plotly_chart(fig_prediction)

                            # ⚡ Volatility Graph (Rolling Standard Deviation)
                            crypto_data["Volatility"] = crypto_data["price"].rolling(window=30).std()

                            fig_volatility = px.line(x=crypto_data["date"], y=crypto_data["Volatility"], 
                                                    title=f"{crypto_name} Volatility Analysis")
                            st.plotly_chart(fig_volatility)

                        else:
                            st.warning(f"Not enough data for seasonal decomposition in {crypto_name}. Minimum 60 data points required.")
                    else:
                        st.error(f"Missing 'price' column in {crypto_name} data.")

        st.subheader("📊 Data Preview")
        for crypto_name, crypto_data in data_dict.items():
            st.write(f"### {crypto_name} Data")
            st.write(crypto_data.head())

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
    st.table(sample_data)

