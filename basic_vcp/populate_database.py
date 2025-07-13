"""
Script to populate SQLite database with NASDAQ stock data
Replaces existing database with new data from NASDAQ_STOCKS directory
"""

import sqlite3
import pandas as pd
import os
import glob
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabasePopulator:
    def __init__(self, db_file="stock_data.db", data_dir="C:/Users/chris/Desktop/repos/scratch/data/NASDAQ_STOCKS"):
        self.db_file = db_file
        self.data_dir = Path(data_dir)
        self.table_name = "stock_prices"
        
    def create_database(self):
        """Create new database and table structure"""
        logger.info(f"Creating new database: {self.db_file}")
        
        # Remove existing database if it exists
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
            logger.info("Removed existing database")
        
        # Create new database and table
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Create table with required fields
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(ticker, date)
            )
            """
            cursor.execute(create_table_sql)
            
            # Create indexes for better performance
            cursor.execute(f"CREATE INDEX idx_ticker_date ON {self.table_name} (ticker, date)")
            cursor.execute(f"CREATE INDEX idx_ticker ON {self.table_name} (ticker)")
            cursor.execute(f"CREATE INDEX idx_date ON {self.table_name} (date)")
            
            conn.commit()
            logger.info("Database and table created successfully")
    
    def parse_date(self, date_str):
        """Convert date string from YYYYMMDD to YYYY-MM-DD format"""
        try:
            # Parse YYYYMMDD format
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{year}-{month}-{day}"
        except:
            return None
    
    def process_file(self, file_path):
        """Process a single data file and return DataFrame"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path, header=None, 
                           names=['ticker', 'period', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'openint'])
            
            # Extract ticker symbol (remove .US suffix)
            df['ticker'] = df['ticker'].str.replace('.US', '')
            
            # Convert date format
            df['date'] = df['date'].astype(str).apply(self.parse_date)
            
            # Filter out invalid dates
            df = df[df['date'].notna()]
            
            # Select only required columns
            df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert volume to integer, handling any non-numeric values
            volume_series = pd.to_numeric(df['volume'], errors='coerce')
            df['volume'] = volume_series.fillna(0).astype(int)
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid prices
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def populate_database(self):
        """Populate database with all data files"""
        logger.info("Starting database population...")
        
        # Create database
        self.create_database()
        
        # Get all .txt files in the data directory
        data_files = list(self.data_dir.glob("*.txt"))
        logger.info(f"Found {len(data_files)} data files")
        
        # Process files in smaller batches to avoid SQLite variable limit
        batch_size = 10  # Reduced batch size
        total_records = 0
        successful_files = 0
        
        with sqlite3.connect(self.db_file) as conn:
            for i in range(0, len(data_files), batch_size):
                batch_files = data_files[i:i+batch_size]
                
                for file_path in batch_files:
                    logger.info(f"Processing {file_path.name}...")
                    
                    df = self.process_file(file_path)
                    if df is not None and not df.empty:
                        # Insert each file individually to avoid SQLite variable limit
                        df.to_sql(self.table_name, conn, if_exists='append', index=False, chunksize=500)
                        successful_files += 1
                        total_records += len(df)
                        logger.info(f"  Added {len(df)} records from {file_path.name}")
                    else:
                        logger.warning(f"  No valid data in {file_path.name}")
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(data_files) + batch_size - 1)//batch_size}")
        
        logger.info(f"Database population complete!")
        logger.info(f"Successfully processed {successful_files} files")
        logger.info(f"Total records inserted: {total_records}")
        
        # Print database statistics
        self.print_database_stats()
    
    def print_database_stats(self):
        """Print database statistics"""
        with sqlite3.connect(self.db_file) as conn:
            # Get total records
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_records = cursor.fetchone()[0]
            
            # Get unique tickers
            cursor.execute(f"SELECT COUNT(DISTINCT ticker) FROM {self.table_name}")
            unique_tickers = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute(f"SELECT MIN(date), MAX(date) FROM {self.table_name}")
            min_date, max_date = cursor.fetchone()
            
            # Get sample tickers
            cursor.execute(f"SELECT DISTINCT ticker FROM {self.table_name} LIMIT 10")
            sample_tickers = [row[0] for row in cursor.fetchall()]
            
            logger.info("Database Statistics:")
            logger.info(f"  Total records: {total_records:,}")
            logger.info(f"  Unique tickers: {unique_tickers}")
            logger.info(f"  Date range: {min_date} to {max_date}")
            logger.info(f"  Sample tickers: {', '.join(sample_tickers)}")

def main():
    """Main function to populate database"""
    logger.info("Starting NASDAQ stock data database population")
    
    populator = DatabasePopulator()
    populator.populate_database()
    
    logger.info("Database population completed successfully!")

if __name__ == "__main__":
    main() 