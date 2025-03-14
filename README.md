# AWS to Weaviate Console

## Project Overview

The AWS to Weaviate Console is a powerful data transfer and management tool designed to seamlessly bridge AWS data sources with Weaviate vector databases. This project provides an intuitive interface for extracting, transforming, and loading data from various AWS services into Weaviate, enabling advanced vector search and AI-powered data analysis.

## Key Features

- **Seamless AWS Integration**: Connect to multiple AWS data sources
- **Weaviate Vector Database Transfer**: Effortless data migration and indexing
- **Interactive Data Management**: User-friendly Streamlit interface
- **Configurable Data Pipelines**: Flexible data transformation options

## Prerequisites

- Python 3.8+
- AWS Account
- Weaviate Instance
- Virtual Environment

## Installation

### 1. Clone the Repository

```bash
git clone [repository-url]
cd [project-directory]
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

1. AWS Credentials

   - Configure AWS CLI: `aws configure`
   - Or set environment variables:
     ```
     export AWS_ACCESS_KEY_ID='your-access-key'
     export AWS_SECRET_ACCESS_KEY='your-secret-key'
     ```

2. Weaviate Connection
   - Update connection parameters in the configuration file
   - Ensure Weaviate instance is accessible

## Running the Application

```bash
streamlit run main.py
```

## Project Structure

```
project-root/
│
├── main.py             # Main application entry point
├── requirements.txt    # Project dependencies
├── venv/               # Virtual environment
│
├── src/                # Source code directory
│   ├── aws_handler.py  # AWS data extraction logic
│   ├── weaviate_handler.py  # Weaviate data import logic
│   └── data_transformer.py  # Data transformation utilities
│
├── config/             # Configuration files
│   └── settings.yaml   # Project settings
│
└── docs/               # Documentation
```

## Development

### Running Tests

```bash
python -m pytest
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

- Ensure AWS credentials are correctly configured
- Check Weaviate connection settings
- Verify Python and dependency versions

## License

[Specify your license here]

## Contact

[Your contact information or project maintainer details]
