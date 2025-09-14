# 📊 Customer Spend Prediction Dashboard  

An **interactive web dashboard** to help businesses analyze customer transaction data and identify **high-value customers** who were predicted to spend more but actually spent less.  

---

## 🚀 Overview  
This project provides a **visual and interactive dashboard** built with Streamlit.  
It enables businesses to:  
- Upload and analyze raw transaction data.  
- Identify customers who underperformed vs. model predictions (🔴 red dots).  
- Segment customers using RFM metrics and ML predictions.  
- Download target customer segments for marketing campaigns.  

---

## 🎯 The Main Dashboard  

### ✨ Scatter Plot (Right Side)  
Each dot = a customer.  

- **Y-axis (Vertical):** Predicted probability of spending in the next 90 days.  
- **X-axis (Horizontal):** Purchase frequency.  
- **Color legend:**  
  - 🔵 **Blue:** Over-performers (spent more than predicted).  
  - 🟡 **Yellow:** Accurate predictions (spent as expected).  
  - 🔴 **Red:** Under-performers (spent less than predicted). ← *Main focus*  

---

## ⚙️ Interactive Controls (Left Side)  

- 📂 **File Uploader** → Upload customer transaction data (`.csv` / `.xlsx`).  
- 🎚 **Min Propensity Slider** → Filter out low-probability customers.  
- 💰 **Shortfall Threshold Slider** → Focus on customers with largest underperformance gaps.  
- 📊 **Segment Size Counter** → Shows how many customers match your filters.  
- ⬇️ **Download Segmentation** → Export selected customers as a CSV for targeted campaigns.  
- 🛠️ **Demo Mode** → Auto-generates sample data if no file is uploaded.  

---

## 📈 Model Performance Check  

- 📉 **Calibration Curve** → Compares predicted vs. actual spend probabilities.  
  - A perfectly calibrated model → Blue line close to dotted diagonal.  
- 🔢 **Brier Score** → Single number summary of prediction accuracy.  
  - Range = 0 → 1.  
  - **Lower = Better.**  

---

## 🛠️ Workflow in a Nutshell  

1. **Upload Data** → Customer transactions file.  
2. **Analyze Visually** → Dashboard cleans, models, and plots.  
3. **Filter & Segment** → Use sliders to focus on underperformers.  
4. **Download & Act** → Get customer list → Run a campaign.  

---

## ▶️ How to Run  

👉 Open in browser: **[http://localhost:8501](http://localhost:8501)**  

---

## 👩‍💻 Author  

Developed by [**Qazi Maria**](https://www.linkedin.com/in/qazi-maria-anees/) ✨  
