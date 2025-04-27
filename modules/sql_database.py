# modules/sql_database.py
import os
import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# ============= CONFIGURATION SETTINGS =============
SQL_PROCESSING_CONFIG = {
    # Database Configuration
    'DB_PATH': "Files/Sql_Files/mysql_db_converted.db",
    'OUTPUT_DIR': "data/processed/sql",
    
    # Logging Configuration
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s: %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    
    # Processing Options
    'MAX_TABLES_TO_PROCESS': 50,
    'INCLUDE_COLUMN_DETAILS': True,
    'SAVE_FULL_SCHEMA': True,
    
    # Error Handling
    'IGNORE_SYSTEM_TABLES': True,
    'SKIP_EMPTY_TABLES': True
}

# Configure logging based on settings
logging.basicConfig(
    level=SQL_PROCESSING_CONFIG['LOG_LEVEL'], 
    format=SQL_PROCESSING_CONFIG['LOG_FORMAT'],
    datefmt=SQL_PROCESSING_CONFIG['LOG_DATE_FORMAT']
)

class SQLDatabaseProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SQL database processor
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        load_dotenv()  # Load environment variables
        
        # Merge provided config with default config
        self.config = SQL_PROCESSING_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.db_path = self.config['DB_PATH']
        self.output_dir = self.config['OUTPUT_DIR']
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if database exists
        if not os.path.exists(self.db_path):
            logging.warning(f"Database not found at {self.db_path}")
        
    def get_database_schema(self) -> Optional[Dict[str, Any]]:
        """
        Extract database schema information
        
        Returns:
            Dictionary containing database schema or None if extraction fails
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {'tables': {}}
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info['tables'][table] = {
                    'columns': [
                        {
                            'name': column[1], 
                            'type': column[2],
                            'primary_key': column[5] == 1
                        } for column in columns
                    ]
                }
            
            conn.close()
            return schema_info
        
        except Exception as e:
            logging.error(f"Error extracting database schema: {e}")
            return None
    
    def save_schema(self, schema_info: Dict[str, Any]) -> None:
        """
        Save database schema to JSON file
        
        Args:
            schema_info: Database schema dictionary
        """
        try:
            schema_file = os.path.join(self.output_dir, 'sql_schema.json')
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema_info, f, indent=2)
            
            logging.info(f"Saved SQL schema to {schema_file}")
        except Exception as e:
            logging.error(f"Error saving schema: {e}")

def process_sql_database(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to process SQL database
    
    Args:
        config: Optional configuration for SQL database processing
    
    Returns:
        Dictionary with processing results
    """
    processor = SQLDatabaseProcessor(config)
    
    # Extract database schema
    schema_info = processor.get_database_schema()
    
    if schema_info:
        # Save schema
        processor.save_schema(schema_info)
        
        return {
            'total_tables': len(schema_info['tables']),
            'schema': schema_info,
            'success': True
        }
    else:
        return {
            'total_tables': 0,
            'schema': None,
            'success': False
        }

# Optional: main block for standalone testing
if __name__ == "__main__":
    result = process_sql_database()
    
    print(f"Total Tables: {result['total_tables']}")
    if result['success']:
        print("Schema processed successfully")