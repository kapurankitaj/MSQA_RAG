import yaml
import os

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config_path = os.path.join('config', 'config.yaml')
    config = load_config(config_path)
    
    print("RAG System Configuration Loaded Successfully!")
    print("Data Sources:", config['data_sources'])

if __name__ == '__main__':
    main()