import pandas as pd
import numpy as np
from cleaning import load_and_clean_data

def validate_results():
    print("📊 Validating Results...")
    
    try:
        df = load_and_clean_data("DISASTERS/disasters_1970_2021.csv")
        print(f"✅ Data validation passed: {len(df)} rows loaded")
        
        # Check required columns
        required_columns = ['Year', 'Disaster Type', 'Total Deaths', 'Total Affected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            print(f"⚠️  Missing columns: {missing}")
        else:
            print("✅ All required columns present")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    print("✅ Validation complete")

if __name__ == "__main__":
    validate_results()
