import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Standard library imports for forecasting
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Data Analysis & Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #ff3333;
    }
    .uploadedFile {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-container {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .forecast-container {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.title("📊 Data Analysis & Forecasting Dashboard")
st.markdown("---")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}


def generate_sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n = len(dates)

    # Generate multiple time series
    data = {
        "date": dates,
        "sales": 1000
        + np.cumsum(np.random.normal(0, 50, n))
        + 50 * np.sin(np.arange(n) * 2 * np.pi / 365),
        "temperature": 20
        + 10 * np.sin(np.arange(n) * 2 * np.pi / 365)
        + np.random.normal(0, 5, n),
        "website_traffic": 5000
        + np.cumsum(np.random.normal(0, 200, n))
        + 200 * np.sin(np.arange(n) * 2 * np.pi / 7),
        "stock_price": 100
        + np.cumsum(np.random.normal(0, 2, n))
        + 5 * np.sin(np.arange(n) * 2 * np.pi / 30),
        "customer_satisfaction": 4.0
        + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 90)
        + np.random.normal(0, 0.2, n),
    }

    return pd.DataFrame(data)


def save_sample_data():
    """Save sample data to CSV"""
    sample_data = generate_sample_data()
    sample_data.to_csv("sample_data.csv", index=False)
    return sample_data


def prepare_lstm_data(data, look_back=30):
    """Prepare data for LSTM model"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back : i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler


def build_lstm_model(look_back):
    """Build LSTM model"""
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def forecast_with_model(data_series, model_name, forecast_periods):
    """Forecast using various standard library models"""
    data_series = data_series.dropna()

    if model_name == "Simple Moving Average":
        # Simple moving average
        window = min(7, len(data_series) // 4)
        ma_forecast = data_series.rolling(window=window).mean().iloc[-1]
        forecast_values = [ma_forecast] * forecast_periods

    elif model_name == "Exponential Smoothing":
        # Statsmodels Exponential Smoothing
        try:
            model = ExponentialSmoothing(
                data_series, seasonal_periods=7, trend="add", seasonal="add"
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(forecast_periods)
            forecast_values = forecast.values
        except:
            # Fallback to simple exponential smoothing
            alpha = 0.3
            forecast_values = []
            last_value = data_series.iloc[-1]
            for _ in range(forecast_periods):
                forecast_values.append(last_value)
                last_value = alpha * data_series.iloc[-1] + (1 - alpha) * last_value

    elif model_name == "Linear Regression":
        # Scikit-learn Linear Regression
        X = np.arange(len(data_series)).reshape(-1, 1)
        y = data_series.values
        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(
            len(data_series), len(data_series) + forecast_periods
        ).reshape(-1, 1)
        forecast_values = model.predict(future_X)

    elif model_name == "ARIMA (Auto-regressive)":
        # Statsmodels ARIMA
        try:
            model = ARIMA(data_series, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecast_values = forecast.values
        except:
            # Fallback to simple trend
            trend = (
                (data_series.iloc[-1] - data_series.iloc[-30]) / 30
                if len(data_series) > 30
                else 0
            )
            forecast_values = [
                data_series.iloc[-1] + trend * i for i in range(1, forecast_periods + 1)
            ]

    elif model_name == "Seasonal Decomposition":
        # Statsmodels Seasonal Decomposition
        try:
            decomposition = seasonal_decompose(
                data_series, period=7, extrapolate_trend="freq"
            )
            trend = decomposition.trend
            seasonal = decomposition.seasonal

            # Extend trend and seasonal components
            last_trend = trend.iloc[-1]
            trend_slope = (
                (trend.iloc[-1] - trend.iloc[-30]) / 30 if len(trend) > 30 else 0
            )

            forecast_values = []
            for i in range(1, forecast_periods + 1):
                future_trend = last_trend + trend_slope * i
                future_seasonal = seasonal.iloc[i % 7] if len(seasonal) >= 7 else 0
                forecast_values.append(future_trend + future_seasonal)
        except:
            # Fallback to simple trend
            trend = (
                (data_series.iloc[-1] - data_series.iloc[-30]) / 30
                if len(data_series) > 30
                else 0
            )
            forecast_values = [
                data_series.iloc[-1] + trend * i for i in range(1, forecast_periods + 1)
            ]

    elif model_name == "Prophet (Facebook)":
        # Simple Prophet-like approach (since Prophet can be heavy)
        try:
            # Use exponential smoothing as proxy for Prophet
            model = ExponentialSmoothing(
                data_series, seasonal_periods=7, trend="add", seasonal="add"
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(forecast_periods)
            forecast_values = forecast.values
        except:
            # Fallback
            trend = (
                (data_series.iloc[-1] - data_series.iloc[-30]) / 30
                if len(data_series) > 30
                else 0
            )
            forecast_values = [
                data_series.iloc[-1] + trend * i for i in range(1, forecast_periods + 1)
            ]

    elif model_name == "LSTM Neural Network":
        # TensorFlow LSTM
        try:
            look_back = min(30, len(data_series) // 4)
            X, y, scaler = prepare_lstm_data(data_series.values, look_back)

            if len(X) > 0:
                model = build_lstm_model(look_back)
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)

                # Prepare future data
                last_sequence = scaler.transform(
                    data_series.values[-look_back:].reshape(-1, 1)
                )
                future_predictions = []

                for _ in range(forecast_periods):
                    next_pred = model.predict(
                        last_sequence.reshape(1, look_back, 1), verbose=0
                    )
                    future_predictions.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1)
                    last_sequence[-1] = next_pred[0, 0]

                forecast_values = scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1)
                ).flatten()
            else:
                # Fallback
                forecast_values = [data_series.iloc[-1]] * forecast_periods
        except:
            # Fallback
            forecast_values = [data_series.iloc[-1]] * forecast_periods

    elif model_name == "Random Forest":
        # Scikit-learn Random Forest
        try:
            # Create features from time series
            df = pd.DataFrame({"value": data_series})
            df["lag1"] = df["value"].shift(1)
            df["lag7"] = df["value"].shift(7)
            df["lag30"] = df["value"].shift(30)
            df["trend"] = range(len(df))
            df = df.dropna()

            if len(df) > 10:
                X = df[["lag1", "lag7", "lag30", "trend"]].values
                y = df["value"].values

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                # Prepare future features
                forecast_values = []
                last_values = data_series.values

                for i in range(forecast_periods):
                    future_features = [
                        last_values[-1],  # lag1
                        (
                            last_values[-7]
                            if len(last_values) >= 7
                            else last_values[-1]
                        ),  # lag7
                        (
                            last_values[-30]
                            if len(last_values) >= 30
                            else last_values[-1]
                        ),  # lag30
                        len(data_series) + i,  # trend
                    ]
                    pred = model.predict([future_features])[0]
                    forecast_values.append(pred)
                    last_values = np.append(last_values, pred)
            else:
                # Fallback
                forecast_values = [data_series.iloc[-1]] * forecast_periods
        except:
            # Fallback
            forecast_values = [data_series.iloc[-1]] * forecast_periods

    else:
        # Default fallback
        trend = (
            (data_series.iloc[-1] - data_series.iloc[-30]) / 30
            if len(data_series) > 30
            else 0
        )
        forecast_values = [
            data_series.iloc[-1] + trend * i for i in range(1, forecast_periods + 1)
        ]

    return forecast_values


# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")

    # Data upload section
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="Upload your data file in CSV format"
    )

    # Or use sample data
    if st.button("📊 Load Sample Data"):
        sample_data = save_sample_data()
        st.session_state.data = sample_data
        st.success("Sample data loaded!")

    if st.session_state.data is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.data)} rows")

    st.markdown("---")

    # Analysis options
    if st.session_state.data is not None:
        st.subheader("📈 Analysis Options")

        # Metric selection
        numeric_columns = st.session_state.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if "date" in st.session_state.data.columns:
            numeric_columns = [col for col in numeric_columns if col != "date"]

        selected_metrics = st.multiselect(
            "Select metrics to analyze:",
            numeric_columns,
            default=(
                numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
            ),
        )

        # Forecasting model selection
        st.subheader("🔮 Forecasting Models")
        forecasting_models = {
            "Simple Moving Average": "sma",
            "Exponential Smoothing": "exp_smooth",
            "Linear Regression": "linear_reg",
            "ARIMA (Auto-regressive)": "arima",
            "Seasonal Decomposition": "seasonal",
            "Prophet (Facebook)": "prophet",
            "LSTM Neural Network": "lstm",
            "Random Forest": "random_forest",
        }

        selected_model = st.selectbox(
            "Choose forecasting model:", list(forecasting_models.keys())
        )

        # Forecast period
        forecast_periods = st.slider(
            "Forecast periods ahead:", min_value=1, max_value=30, value=7
        )

        # Analysis button
        if st.button("🚀 Run Analysis"):
            st.session_state.analysis_results = {
                "metrics": selected_metrics,
                "model": selected_model,
                "forecast_periods": forecast_periods,
            }

# Main content area
if st.session_state.data is not None:
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(st.session_state.data))
    with col2:
        st.metric("Total Columns", len(st.session_state.data.columns))
    with col3:
        st.metric(
            "Date Range",
            f"{st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}",
        )

    st.markdown("---")

    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(st.session_state.data.head(10), use_container_width=True)

    # Analysis results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        metrics = st.session_state.analysis_results["metrics"]
        model = st.session_state.analysis_results["model"]
        forecast_periods = st.session_state.analysis_results["forecast_periods"]

        # Create visualizations
        if metrics:
            # Time series plots
            st.subheader("📈 Time Series Analysis")

            # Create subplots for each metric
            n_metrics = len(metrics)
            fig = make_subplots(
                rows=n_metrics, cols=1, subplot_titles=metrics, vertical_spacing=0.1
            )

            colors = ["#ff4b4b", "#4bff4b", "#4b4bff", "#ffff4b", "#ff4bff"]

            for i, metric in enumerate(metrics):
                if metric in st.session_state.data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state.data["date"],
                            y=st.session_state.data[metric],
                            mode="lines",
                            name=metric,
                            line=dict(color=colors[i % len(colors)]),
                        ),
                        row=i + 1,
                        col=1,
                    )

            fig.update_layout(
                height=300 * n_metrics,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistical summary
            st.subheader("📊 Statistical Summary")
            summary_stats = st.session_state.data[metrics].describe()
            st.dataframe(summary_stats, use_container_width=True)

            # Correlation heatmap
            st.subheader("🔥 Correlation Analysis")
            correlation_matrix = st.session_state.data[metrics].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            # Forecasting section
            st.subheader(f"🔮 Forecasting with {model}")

            # Simple forecasting implementation
            for metric in metrics[:2]:  # Limit to first 2 metrics for demo
                if metric in st.session_state.data.columns:
                    st.write(f"**Forecasting {metric}**")

                    # Simple moving average forecast
                    data_series = st.session_state.data[metric].dropna()

                    if model == "Simple Moving Average":
                        window = 7
                        ma_forecast = data_series.rolling(window=window).mean().iloc[-1]
                        forecast_values = [ma_forecast] * forecast_periods

                    elif model == "Linear Regression":
                        # Simple linear trend
                        x = np.arange(len(data_series))
                        coeffs = np.polyfit(x, data_series, 1)
                        future_x = np.arange(
                            len(data_series), len(data_series) + forecast_periods
                        )
                        forecast_values = np.polyval(coeffs, future_x)

                    else:
                        # Default to simple trend
                        trend = (data_series.iloc[-1] - data_series.iloc[-30]) / 30
                        forecast_values = [
                            data_series.iloc[-1] + trend * i
                            for i in range(1, forecast_periods + 1)
                        ]

                    # Create forecast plot
                    last_date = st.session_state.data["date"].iloc[-1]
                    forecast_dates = [
                        last_date + timedelta(days=i)
                        for i in range(1, forecast_periods + 1)
                    ]

                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state.data["date"],
                            y=st.session_state.data[metric],
                            mode="lines",
                            name="Historical",
                            line=dict(color="#4bff4b"),
                        )
                    )

                    # Forecast data
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=forecast_values,
                            mode="lines+markers",
                            name="Forecast",
                            line=dict(color="#ff4b4b", dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title=f"{metric} Forecast ({model})",
                        xaxis_title="Date",
                        yaxis_title=metric,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Forecast table
                    forecast_df = pd.DataFrame(
                        {"Date": forecast_dates, "Forecast": forecast_values}
                    )
                    st.dataframe(forecast_df, use_container_width=True)

                    st.markdown("---")

        else:
            st.warning("Please select at least one metric to analyze.")

    else:
        st.info(
            "👈 Use the sidebar to select metrics and forecasting model, then click 'Run Analysis'"
        )

else:
    # Welcome message
    st.markdown(
        """
    ## 🎯 Welcome to the Data Analysis & Forecasting Dashboard!
    
    This application allows you to:
    
    - 📁 **Upload your data** in CSV format
    - 📊 **Analyze multiple metrics** with interactive visualizations
    - 🔮 **Forecast future values** using various models
    - 📈 **View statistical summaries** and correlations
    
    ### Getting Started:
    1. Upload a CSV file or click "Load Sample Data" to try with example data
    2. Select the metrics you want to analyze
    3. Choose a forecasting model
    4. Click "Run Analysis" to see results
    
    ### Supported Forecasting Models:
    - **Simple Moving Average**: Good for stable trends
    - **Exponential Smoothing**: Handles trends and seasonality
    - **Linear Regression**: Simple trend projection
    - **ARIMA**: Advanced time series modeling
    - **Seasonal Decomposition**: Separates trend, seasonal, and residual components
    - **Prophet**: Facebook's forecasting tool
    - **LSTM Neural Network**: Deep learning approach
    - **Random Forest**: Ensemble method for complex patterns
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #888;'>
    Built with ❤️ using Streamlit | Data Analysis & Forecasting Dashboard
</div>
""",
    unsafe_allow_html=True,
)
