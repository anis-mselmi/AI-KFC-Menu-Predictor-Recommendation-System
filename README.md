# AI KFC Menu Predictor & Recommendation System 🍗🤖

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)
![Jupyter](https://img.shields.io/badge/Tool-Jupyter-F37626.svg)

## 📌 Project Objective
This project is a **Mock AI Recommendation System** designed to predict what a customer will order next based on historical order data. It mimics real-world scenarios by generating "messy" data, cleaning it, and then building a **Collaborative Filtering** model to suggest relevant menu items.

The system is designed to simulate a production pipeline where a backend script handles the heavy lifting, supported by research notebooks.

---

## 📂 Project Structure

The project is organized into **1 Consolidated Python Script** (for production/automation) and **3 Research Notebooks** (for analysis).

| File Name | Type | Description |
| :--- | :--- | :--- |
| `kfc_predictor.py` | **Main Script** | The core application. Handles **Data Generation, Cleaning, Model Training, and Prediction**. Run this to use the system. |
| `01_data_cleaning...ipynb` | Notebook | Code-only notebook demonstrating the data generation and cleaning logic. |
| `02_exploratory_data...ipynb` | Notebook | Code-only notebook for visualizing order trends (Seaborn charts). |
| `03_model_building...ipynb` | Notebook | Code-only notebook demonstrating the Cosine Similarity algorithms. |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```
*Note: This project is compatible with Numpy < 2.0.0.*

### 2. Execute the Main System
Run the main script to generate fresh data, train the model, and view predictions:

```bash
python kfc_predictor.py
```

### 3. Expected Output
The script will output the pipeline status and final recommendation examples:
```text
=== AI KFC Menu Predictor System ===
[1/3] Generating and Cleaning Data...
      Data saved to 'cleaned_kfc_data.csv'.
[2/3] Building Model...
      Model saved to 'similarity_matrix.pkl'.

[3/3] Validating Model Functionality...
      Input: Zinger Burger        -> Recommended: ['Mashed Potatoes', 'Biscuits', 'Coleslaw']
      Input: Fries                -> Recommended: ['Twister Wrap', 'Zinger Burger', 'Original Recipe Chicken']
```

---

## 🧠 Model Logic: Collaborative Filtering

The recommendation engine uses **Item-Based Collaborative Filtering**. 

1.  **User-Item Matrix**: We treat each `OrderID` as a session and map which items (Primary + Side) occurred in that session.
2.  **Cosine Similarity**: We calculate the cosine similarity between items based on their co-occurrence patterns.
    
    $$
    \text{similarity}(A,B) = \frac{A \cdot B}{||A|| \times ||B||}
    $$
    
    If "Zinger Burger" and "Fries" frequently appear in the same orders, their vector angle is small, resulting in a high similarity score.

---

## 📊 Features & Workflow

1.  **Data Generation**: Creates a robust mock dataset of 500 orders with intentional anomalies (missing ages, inconsistencies like "zinger burger" vs "Zinger Burger").
2.  **Data Cleaning**: 
    -   Imputes missing categorical data with the Mode.
    -   Imputes missing numerical data (Age) with the Median.
    -   Standardizes capitalization and formatting.
3.  **Visualization**: (In Notebook 02) Analyzes popular items and peak ordering times.
4.  **Prediction API**: Returns the top 3 recommended items for any given input item.

---

## 📝 Authors & Acknowledgments
-   **Lead Machine Learning Engineer**: Anis Mselmi
-   **Tech Stack**: Python, Pandas, Scikit-Learn
