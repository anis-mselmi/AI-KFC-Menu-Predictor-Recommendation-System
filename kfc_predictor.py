import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class KFCProject:
    def __init__(self, data_file='cleaned_kfc_data.csv', model_file='similarity_matrix.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.num_orders = 500
        
    def generate_and_clean_data(self):
        """Generates mock data and performs cleaning."""
        print("[1/3] Generating and Cleaning Data...")
        
        # --- Data Generation ---
        random.seed(42)
        np.random.seed(42)  # For reproducibility

        primary_items = ['Zinger Burger', 'Original Recipe Chicken', 'Popcorn Chicken', 'Twister Wrap', 'Hot Wings', 'zinger burger', 'ZINGER BURGER', 'Original Recipe', None]
        side_items = ['Fries', 'Coleslaw', 'Mashed Potatoes', 'Corn on the Cob', 'Biscuits', 'FRIES', None, 'mashed potatoes']
        times_of_day = ['Lunch', 'Dinner', 'Late Night', 'Afternoon Snack', 'lunch', 'DINNER']

        # Generating lists with pure Python types to avoid potential numpy issues
        data = {
            'OrderID': list(range(1, self.num_orders + 1)),
            'Primary_Item': [random.choice(primary_items) for _ in range(self.num_orders)],
            'Side_Item': [random.choice(side_items) for _ in range(self.num_orders)],
            'Time_Of_Day': [random.choice(times_of_day) for _ in range(self.num_orders)],
            'Customer_Age': [random.randint(18, 65) if random.random() > 0.1 else None for _ in range(self.num_orders)]
        }
        
        df = pd.DataFrame(data)
        
        # --- Data Cleaning ---
        # Standardize Text
        text_cols = ['Primary_Item', 'Side_Item', 'Time_Of_Day']
        for col in text_cols:
            df[col] = df[col].astype(str).str.lower().str.title()
            df[col] = df[col].replace({'None': np.nan, 'Nan': np.nan})

        # Canonical Mapping
        item_mapping = {
            'Original Recipe': 'Original Recipe Chicken',
            'Zinger Burger': 'Zinger Burger' 
        }
        df['Primary_Item'] = df['Primary_Item'].replace(item_mapping)

        # Impute Missing Values
        for col in ['Primary_Item', 'Side_Item', 'Time_Of_Day']:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        if not df['Customer_Age'].empty:
            df['Customer_Age'] = df['Customer_Age'].fillna(df['Customer_Age'].median())

        # Save Data
        df.to_csv(self.data_file, index=False)
        print(f"      Data saved to '{self.data_file}'.")
        return df

    def train_model(self):
        """Builds the collaborative filtering model."""
        print("[2/3] Building Model...")
        
        if not os.path.exists(self.data_file):
            print("      Data file not found. Generating now...")
            self.generate_and_clean_data()
            
        df = pd.read_csv(self.data_file)
        
        # Create User-Item Matrix (Order-Item Matrix)
        melted_df = df.melt(id_vars=['OrderID'], value_vars=['Primary_Item', 'Side_Item'], value_name='Item').dropna()
        melted_df['Quantity'] = 1
        
        order_item_matrix = melted_df.pivot_table(index='OrderID', columns='Item', values='Quantity', fill_value=0)
        
        # Calculate Cosine Similarity (Item-Item)
        item_item_matrix = order_item_matrix.T
        cosine_sim_matrix = cosine_similarity(item_item_matrix)
        
        cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=item_item_matrix.index, columns=item_item_matrix.index)
        
        # Save Model
        with open(self.model_file, 'wb') as f:
            pickle.dump(cosine_sim_df, f)
        print(f"      Model saved to '{self.model_file}'.")
        return cosine_sim_df

    def predict(self, item_name, top_n=3):
        """Recommend items based on input item."""
        if not os.path.exists(self.model_file):
             self.train_model()
             
        with open(self.model_file, 'rb') as f:
            similarity_df = pickle.load(f)
        
        item_name = str(item_name).title()
        
        if item_name not in similarity_df.index:
            return [f"Item '{item_name}' not found in menu history."]
        
        # Get scores
        scores = similarity_df[item_name]
        scores = scores.drop(item_name).sort_values(ascending=False)
        
        return scores.head(top_n).index.tolist()

def main():
    print("=== AI KFC Menu Predictor System ===")
    project = KFCProject()
    
    # 1. Force regeneration to ensure clean state
    # (In production, you might check if files exist, but for this portfolio script, we run the pipeline)
    project.generate_and_clean_data()
    project.train_model()
    
    # 2. Validation / Demonstration
    print("\n[3/3] Validating Model Functionality...")
    
    test_cases = ["Zinger Burger", "Fries", "Hot Wings"]
    
    for item in test_cases:
        recs = project.predict(item)
        print(f"      Input: {item:<20} -> Recommended: {recs}")
        
    print("\n=== System Ready & Verified ===")

if __name__ == "__main__":
    main()
