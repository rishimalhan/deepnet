# Data Analysis & Forecasting Dashboard

A beautiful dark-themed Streamlit application for data analysis and forecasting with multiple model options.

## Features

- 📁 **File Upload**: Upload CSV files or use sample data
- 📊 **Multi-Metric Analysis**: Select and analyze multiple metrics simultaneously
- 🔮 **Multiple Forecasting Models**: Choose from 8 different forecasting approaches
- 📈 **Interactive Visualizations**: Beautiful charts and graphs with Plotly
- 🎨 **Dark Theme**: Modern, eye-friendly dark interface
- 📋 **Statistical Summaries**: Comprehensive data analysis

## Supported Forecasting Models

1. **Simple Moving Average**: Good for stable trends
2. **Exponential Smoothing**: Handles trends and seasonality
3. **Linear Regression**: Simple trend projection
4. **ARIMA**: Advanced time series modeling
5. **Seasonal Decomposition**: Separates trend, seasonal, and residual components
6. **Prophet**: Facebook's forecasting tool
7. **LSTM Neural Network**: Deep learning approach
8. **Random Forest**: Ensemble method for complex patterns

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run demo.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. **Load Data**: Either upload a CSV file or click "Load Sample Data" to try with example data
2. **Select Metrics**: Choose which metrics you want to analyze from the dropdown
3. **Choose Model**: Select a forecasting model from the available options
4. **Set Forecast Period**: Use the slider to set how many periods ahead to forecast
5. **Run Analysis**: Click "Run Analysis" to see the results

## Sample Data

The application includes sample time series data with the following metrics:
- Sales data with seasonal patterns
- Temperature data with yearly cycles
- Website traffic with weekly patterns
- Stock price with monthly trends
- Customer satisfaction scores

## File Structure

```
scripts/
├── demo.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── sample_data.csv     # Generated sample data (created when you click "Load Sample Data")
```

## Customization

You can easily customize the application by:
- Adding new forecasting models in the `forecasting_models` dictionary
- Modifying the CSS styles for different themes
- Adding new visualization types
- Extending the statistical analysis capabilities

## Requirements

- Python 3.8+
- Streamlit 1.28+
- See `requirements.txt` for full dependency list

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that your CSV file has a 'date' column for time series analysis
3. Ensure your data has numeric columns for analysis
4. Try using the sample data first to verify everything works

## License

This project is open source and available under the MIT License. 