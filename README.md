# Car Price Prediction with UI

A premium Streamlit-based machine learning dashboard for car resale price prediction using Multiple Linear Regression.

## Features

✨ **Interactive Predictions** - Enter car details and get instant price predictions  
📊 **Data Visualizations** - Engine vs Price, Mileage vs Price, Correlation Heatmap  
🎨 **Custom Themes** - Choose between Luxury Gold, Dark Mode, and Neon Blue  
⚡ **Responsive Design** - Fully responsive UI with animations  

## Project Structure

```
Car Price Prediction with UI/
├── app.py                          # Main Streamlit app (for local & cloud)
├── model.pkl                       # Trained ML model
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── .gitignore                      # Git ignore rules
└── Price Prediction MLR.ipynb      # Model training notebook
```

## Local Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd "Car Price Prediction with UI"
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Streamlit Cloud Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **New app**
3. Select your repository and branch
4. Set main file path to: `car_price_prediction/app.py`
5. Click **Deploy**

### 3. Required Files (Already Configured)
- ✅ `app.py` - Main entry point (Streamlit Cloud looks for this)
- ✅ `requirements.txt` - All dependencies listed
- ✅ `model.pkl` - Trained model (in same directory as app.py)
- ✅ `.streamlit/config.toml` - Streamlit settings
- ✅ `.gitignore` - Excludes unnecessary files

## Dependencies

- **streamlit** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML algorithms
- **joblib** - Model persistence
- **plotly** - Interactive visualizations
- **seaborn** - Statistical visualizations
- **matplotlib** - Plotting library

## Model Details

**Algorithm:** Multiple Linear Regression  
**Features:** 
- Numeric: vehicle_age, km_driven, mileage, engine, max_power, seats
- Categorical: fuel_type, seller_type, transmission_type (one-hot encoded)

**Target:** Car resale price (₹ Indian Rupees)

## Features Explained

### Home Page
- Dashboard overview with premium design
- Key statistics (avg mileage, avg price, feature count)
- Model explanation

### Predict Page
- Interactive input fields for car details
- Real-time price prediction with animation
- Gauge chart visualization

### Visuals Page
- Scatter plots (Engine vs Price, Mileage vs Price)
- Correlation heatmap
- Trend analysis

### Settings Page
- Theme selector (Luxury Gold, Dark Mode, Neon Blue)
- About information
- Developer credits

## Troubleshooting

### Model Not Found Error
Ensure `model.pkl` is in the same directory as `app.py` and is named exactly `model.pkl`

### Import Errors
Update dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use
Run on different port:
```bash
streamlit run app.py --server.port 8502
```

## Development

To modify the model or add features:
1. Update `Price Prediction MLR.ipynb`
2. Retrain and export model as `model.pkl`
3. Replace the existing `model.pkl` file
4. Push changes to GitHub (Streamlit Cloud auto-redeploys)

## License

This project is open source and available for educational purposes.

## Author

**Chetana Parakh**  
Focus Areas: Machine Learning, UI/UX, Software Engineering

---

**Version:** v2.0 — Ultra Premium Edition  
**Last Updated:** May 2026
