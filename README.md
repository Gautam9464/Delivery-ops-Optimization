# Delivery-ops-Optimization

# 🚀 Global Gourmet: Enterprise Delivery & Optimization Engine

An end-to-end Machine Learning and Operations Research pipeline designed to optimize food delivery logistics, quantify customer sentiment causality, predict user churn, and dynamically optimize surge pricing.

## 📊 Project Architecture

This project is divided into five core operational phases:

### Phase I & II: Data Engineering & Feature Synthesis
* Merged massive datasets (Yelp reviews, TLC trip data, simulated telemetry).
* Executed time-series `merge_asof` joins to accurately map courier pings to order timestamps.
* Engineered spatio-temporal features, integrating real-time NYC weather via the Open-Meteo API and US holiday calendars.

### Phase III: High-Fidelity ETA Engine (XGBoost)
* Built an ETA prediction model utilizing a **Custom Asymmetric L1 Loss Function** to heavily penalize extreme lateness (outliers) while tolerating early deliveries.
* Utilized **SHAP (SHapley Additive exPlanations)** to extract feature importance and visualize the spatio-temporal drivers of delivery delays.

### Phase IV: NLP & Causal Inference
* Applied **BERTopic** and **VADER Sentiment Analysis** to cluster unstructured customer grievances.
* Utilized **Logistic Regression** to quantify the causal relationship between Absolute ETA Errors and 1-star reviews (extracting Odds Ratios to define the "Lateness Cliff").

### Phase V: Churn Prediction & Retention
* Trained a Random Forest classifier to identify at-risk users based on historical lateness and grievance frequency.
* Mathematically tuned the decision threshold using the **F2-Score** to prioritize Recall, maximizing the identification of churning customers over pure precision.

### Phase VI: Multi-Variable Price Discrimination (Surge Pricing)
* Modeled demand elasticity across three user segments (Corporate, Standard, Sensitive).
* Utilized **SciPy SLSQP Optimization** to maximize platform revenue via segmented surge pricing, strictly constrained to ensure user churn propensity never increases by more than 2%.

## 🛠️ Tech Stack
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-learn
* **NLP:** BERTopic, NLTK (VADER)
* **Optimization:** SciPy (`scipy.optimize.minimize`)
* **Explainability & Viz:** SHAP, Matplotlib, Seaborn

## 🚀 Installation & Usage

Clone the repository:
```bash
git clone [https://github.com/YOUR-USERNAME/delivery-ops-optimization.git](https://github.com/YOUR-USERNAME/delivery-ops-optimization.git)
cd delivery-ops-optimization
```
Data:
```bash
https://drive.google.com/drive/folders/1rdIz5dO_yuMxgA9D3O0r5fffS144_6mQ?usp=sharing
```
Create a virtual environment and install dependencies:
```Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Launch Jupyter Notebook to view and run the pipeline:
```Bash
jupyter notebook notebooks/Ingenium.ipynb
```
