import os
import glob
import pandas as pd
import numpy as np
import re
import warnings

class BiTDataProcessor:
    """
    Class to process and clean BiT Project CSV files.
    Handles inconsistent columns and Excel-induced date formatting corruption.
    """
    
    EXPECTED_COLUMNS = [
        'C26M1_Ch1', 'C26M2_Ch1', 'C26M3_Ch1', 'C26M4_Ch1'
    ]
    
    # German/Excel month abbreviation mapping
    MONTH_MAP = {
        'Jan': 1, 'Feb': 2, 'Mrz': 3, 'Mar': 3, 'Apr': 4, 
        'Mai': 5, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 
        'Sep': 9, 'Okt': 10, 'Oct': 10, 'Nov': 11, 'Dez': 12, 'Dec': 12
    }

    def __init__(self, input_dir, output_dir):
        """
        Initialize the processor.
        
        Args:
            input_dir (str): Directory containing raw LogJob_*.csv files.
            output_dir (str): Directory to save cleaned CSV files.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def _fix_value(self, val):
        """
        Fix values corrupted by Excel date formatting.
        e.g., '21. Apr' -> 21.4
        
        Args:
           val: The value to fix (string, float, or other).
           
        Returns:
            float: Fixed temperature value.
        """
        # If it's already a number (and not NaN), return it
        if isinstance(val, (int, float)):
            return val
            
        val_str = str(val).strip()
        
        # Check for simple float in string format (e.g. "25.0")
        try:
            return float(val_str.replace(',', '.'))
        except ValueError:
            pass

        # Check for Date pattern: "DD. Mon"
        # Regex capturing group 1: Day (digits)
        # Regex capturing group 2: Month (abbreviation)
        match = re.match(r'^(\d+)\.\s+([A-Za-z]+)', val_str)
        if match:
            day = match.group(1)
            month_str = match.group(2)
            
            # Remove trailing dot if present in month string (e.g. "Mrz.")
            month_str = month_str.rstrip('.')
            
            if month_str in self.MONTH_MAP:
                month_num = self.MONTH_MAP[month_str]
                # Reconstruct the float: "Day.Month"
                try:
                    return float(f"{day}.{month_num}")
                except ValueError:
                    pass
        
        # If we reach here, we couldn't parse it. Return NaN or the original if debugging
        # But for cleaning, NaN is safer than keeping garbage
        return np.nan

    def process_file(self, file_path):
        """
        Process a single CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
        """
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            # Read CSV - assuming semicolon delimiter based on context
            df = pd.read_csv(file_path, sep=';')
            
            # 1. Standardize Columns
            # Ensure Date column is preserved/detected
            date_col = None
            for col in df.columns:
                if 'Date' in col or 'Time' in col:
                    date_col = col
                    break
            
            if date_col is None:
                print(f"  Warning: No Date/Time column found in {filename}. Creating dummy index.")
                df['Date'] = range(len(df))
                date_col = 'Date'
            
            # Setup new dataframe with Date and expected columns
            new_df = pd.DataFrame()
            new_df['Date'] = df[date_col]
            
            existing_cols = df.columns.tolist()
            
            for target_col in self.EXPECTED_COLUMNS:
                if target_col in existing_cols:
                    new_df[target_col] = df[target_col]
                else:
                    print(f"  Warning: Column {target_col} missing in {filename}. Filling with NaN.")
                    new_df[target_col] = np.nan
            
            # 2. Fix corrupted values
            for col in self.EXPECTED_COLUMNS:
                # Apply fix_value to each element in the sensor columns
                new_df[col] = new_df[col].apply(self._fix_value)
                
            # 3. Save cleaned file
            output_path = os.path.join(self.output_dir, filename)
            # Use standard CSV format (comma separated) for cleaned data
            new_df.to_csv(output_path, index=False)
            print(f"  Saved cleaned file to {output_path}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    def run(self):
        """
        Run processing for all matching files in input directory.
        """
        search_pattern = os.path.join(self.input_dir, "LogJob_*.csv")
        files = sorted(glob.glob(search_pattern))
        
        if not files:
            print(f"No LogJob_*.csv files found in {self.input_dir}")
            return
            
        print(f"Found {len(files)} files to process.")
        
        for file_path in files:
            self.process_file(file_path)
        print("Processing complete.")

if __name__ == "__main__":
    # Example usage
    INPUT_DIR = "/mnt/data2/video_regression/data/new_data/BiT_Projekt/"
    OUTPUT_DIR = "/mnt/data2/video_regression/data/new_data/cleaned/"
    
    processor = BiTDataProcessor(INPUT_DIR, OUTPUT_DIR)
    processor.run()
