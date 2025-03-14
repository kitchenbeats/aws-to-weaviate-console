#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA Consultant AI - Premium UI Interface
----------------------------------------
A beautiful, professional UI for the FDA Consultant AI pipeline.

This is the main entry point for the application.
"""

from datetime import datetime
import atexit
import streamlit as st

# Import utility modules
from utils.config_manager import ConfigManager
from utils.error_handling import handle_application_error, setup_error_logging

# Import service modules
from services.aws_manager import AWSManager
from services.weaviate_manager import WeaviateManager
from services.pipeline_runner import EnhancedPipelineRunner
from services.cloud_monitor import CloudProcessMonitor

# Import UI modules
from ui.dashboard import render_dashboard
from ui.config_section import render_config_section
from ui.execution_section import render_execution_section
from ui.extensions_section import render_extensions_section

# Configure page layout
st.set_page_config(
    page_title="AWS to Weaviate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look and feel
def load_custom_css():
    """Load custom CSS styling for premium UI."""
    st.markdown("""
    <style>
        /* Main color scheme */
        :root {
            --primary: #1E88E5;
            --primary-dark: #1565C0;
            --secondary: #26A69A;
            --background: #FAFAFA;
            --surface: #FFFFFF;
            --error: #D32F2F;
            --text-primary: #212121;
            --text-secondary: #757575;
            --border: #EEEEEE;
        }

        /* Background and text colors */
        .stApp {
            background-color: var(--background);
            color: var(--text-primary);
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        h1 {
            font-size: 2.25rem;
            letter-spacing: -0.5px;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            font-size: 1.8rem;
            margin-top: 1.5rem;
            color: var(--primary-dark);
        }
        
        h3 {
            font-size: 1.5rem;
            margin-top: 1rem;
            color: var(--primary-dark);
        }

        /* Cards styling */
        .stCardContainer {
            background-color: var(--surface);
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary);
        }
        
        /* Input widgets styling */
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="textarea"] textarea {
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Primary button */
        .primary-btn {
            background-color: var(--primary);
            color: white;
        }
        
        /* Section containers */
        .section-container {
            background-color: var(--surface);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active {
            background-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2);
        }
        
        .status-warning {
            background-color: #FF9800;
            box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.2);
        }
        
        .status-error {
            background-color: #F44336;
            box-shadow: 0 0 0 3px rgba(244, 67, 54, 0.2);
        }
        
        .status-inactive {
            background-color: #9E9E9E;
            box-shadow: 0 0 0 3px rgba(158, 158, 158, 0.2);
        }
        
        /* Dashboard metrics */
        .metric-container {
            background-color: var(--surface);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: var(--primary);
        }
        
        /* Code blocks */
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
        }
        
        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 6px 6px 0 0;
            gap: 1rem;
            padding: 10px 16px;
            color: var(--text-primary);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: white !important;
        }
        
        .output-container {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 1rem;
            overflow-x: auto;
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)


def load_env_config():
    """Load configuration from environment variables."""
    import os
    from dotenv import load_dotenv
    
    # Load .env file if it exists
    load_dotenv()
    
    # Create a default config from environment variables
    config = {}
    
    # Weaviate settings
    if os.getenv("WEAVIATE_URL"):
        config["weaviate_url"] = os.getenv("WEAVIATE_URL")
        if os.getenv("WEAVIATE_API_KEY"):
            config["weaviate_api_key"] = os.getenv("WEAVIATE_API_KEY")
    
    # AWS Region
    if os.getenv("AWS_REGION"):
        config["region"] = os.getenv("AWS_REGION")
    
    # S3 bucket configuration
    if os.getenv("INPUT_S3_BUCKET"):
        config["input_bucket"] = os.getenv("INPUT_S3_BUCKET")
    if os.getenv("OUTPUT_S3_BUCKET"):
        config["output_bucket"] = os.getenv("OUTPUT_S3_BUCKET")
    if os.getenv("INPUT_S3_PREFIX"):
        config["input_s3_prefix"] = os.getenv("INPUT_S3_PREFIX")
    if os.getenv("OUTPUT_S3_PREFIX"):
        config["output_s3_prefix"] = os.getenv("OUTPUT_S3_PREFIX")
    
    # SageMaker configuration
    if os.getenv("SAGEMAKER_MODEL_NAME"):
        config["model_name"] = os.getenv("SAGEMAKER_MODEL_NAME")
    
    return config


def cleanup_application():
    """Clean up application resources before shutdown."""
    try:
        # Access global managers through session state for cleanup
        if 'pipeline_runner' in st.session_state:
            pipeline_runner = st.session_state['pipeline_runner']
            if hasattr(pipeline_runner, 'stop') and getattr(pipeline_runner, 'is_running', False):
                pipeline_runner.stop()
        
        if 'aws_manager' in st.session_state:
            aws_manager = st.session_state['aws_manager']
            if hasattr(aws_manager, 's3_client') and aws_manager.s3_client:
                try:
                    aws_manager.s3_client._endpoint.http_session.close()
                except:
                    pass
        
        if 'weaviate_manager' in st.session_state:
            weaviate_manager = st.session_state['weaviate_manager']
            if hasattr(weaviate_manager, 'close'):
                weaviate_manager.close()
        
        print("Application cleanup completed")
    except Exception as e:
        print(f"Error during application cleanup: {e}")


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.title("FDA Consultant AI")
        st.markdown("#### The Ultimate PubMed Data Processing & Embedding Pipeline")
    
    with col2:
        # Dynamic status indicator
        status_class = "status-active"
        status_text = "Ready"
        
        # Check if pipeline is running
        if 'pipeline_runner' in st.session_state:
            pipeline_runner = st.session_state['pipeline_runner']
            if getattr(pipeline_runner, 'is_running', False):
                status_class = "status-active"
                status_text = "Running"
        
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 1rem;">
                <span class="status-indicator {status_class}"></span> <span style="font-weight: 600;">{status_text}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )


def initialize_session_state():
    """Initialize session state with default values."""
    # Initialize base session state
    if 'page_initialized' not in st.session_state:
        st.session_state.page_initialized = True
        st.session_state['current_config'] = {}
        st.session_state['current_prefix'] = ""
    
    # Initialize managers if not already present
    if 'aws_manager' not in st.session_state:
        st.session_state['aws_manager'] = AWSManager()
    
    if 'weaviate_manager' not in st.session_state:
        st.session_state['weaviate_manager'] = WeaviateManager()
    
    if 'pipeline_runner' not in st.session_state:
        st.session_state['pipeline_runner'] = EnhancedPipelineRunner()
    
    if 'cloud_monitor' not in st.session_state:
        st.session_state['cloud_monitor'] = CloudProcessMonitor()
    
    # Initialize error tracking
    if 'error_count' not in st.session_state:
        st.session_state['error_count'] = 0
    
    # Initialize last navigation time
    if 'last_navigation' not in st.session_state:
        st.session_state['last_navigation'] = datetime.now()


def render_sidebar(aws_manager, weaviate_manager):
    """Render sidebar with navigation and connection status."""
    st.sidebar.title("Navigation")
    
    # Navigation options
    page = st.sidebar.radio("Go to", ["Dashboard", "Configuration", "Execution", "Extensions"])
    
    # AWS connection status
    if aws_manager.initialized:
        st.sidebar.markdown(
            """
            <div style="padding: 0.5rem; background-color: #e8f5e9; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #4CAF50; font-weight: bold;">✓</span> AWS Connected
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="padding: 0.5rem; background-color: #ffebee; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #F44336; font-weight: bold;">✗</span> AWS Not Connected
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Weaviate connection status
    if weaviate_manager.initialized:
        st.sidebar.markdown(
            """
            <div style="padding: 0.5rem; background-color: #e8f5e9; border-radius: 5px; margin-top: 1rem;">
                <span style="color: #4CAF50; font-weight: bold;">✓</span> Weaviate Connected
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Version information
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Version Information")
    st.sidebar.markdown("FDA Consultant AI - Premium Edition v1.2.0")
    st.sidebar.markdown("© 2025 - All Rights Reserved")
    
    return page


def render_footer():
    """Render application footer."""
    st.markdown("""
    <hr style="margin-top: 2rem;">
    <div style="text-align: center; padding: 1rem 0; color: var(--text-secondary);">
        FDA Consultant AI - Premium Edition &copy; 2025
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    try:
        # Load custom CSS for premium UI
        load_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Set up error logging
        setup_error_logging()
        
        # Register cleanup function to run at exit
        atexit.register(cleanup_application)
        
        # Load environment variables
        env_config = load_env_config()
        
        # Initialize or retrieve managers from session state
        config_manager = ConfigManager()
        aws_manager = st.session_state['aws_manager']
        weaviate_manager = st.session_state['weaviate_manager']
        pipeline_runner = st.session_state['pipeline_runner']
        
        # Load default config from saved profiles
        saved_config = config_manager.get_default_profile() or {}
        
        # Merge environment config with saved config (env takes precedence)
        default_config = {**saved_config, **env_config}
        
        # Render header
        render_header()
        
        # Render sidebar and get selected page
        page = render_sidebar(aws_manager, weaviate_manager)
        
        # Initialize AWS connection
        try:
            region = default_config.get("region", "us-east-1")
            aws_manager.initialize(region)
        except Exception as e:
            st.sidebar.error(f"AWS connection error: {e}")
        
        # Initialize Weaviate connection if URL is provided
        try:
            if default_config.get("weaviate_url"):
                weaviate_manager.initialize(
                    default_config.get("weaviate_url"),
                    default_config.get("weaviate_api_key")
                )
        except Exception as e:
            st.sidebar.error(f"Weaviate connection error: {e}")
        
        # Main content based on selected page
        if page == "Configuration":
            config = render_config_section(config_manager, aws_manager, weaviate_manager)
            # Store config in session state for other tabs to use
            st.session_state['current_config'] = config
        elif page == "Dashboard":
            # Get the current configuration from session state or default profile
            dashboard_config = st.session_state.get('current_config', default_config)
            render_dashboard(aws_manager, weaviate_manager, dashboard_config)
        elif page == "Execution":
            # Get the current configuration from session state or default profile
            execution_config = st.session_state.get('current_config', default_config)
            render_execution_section(pipeline_runner, aws_manager, config_manager, execution_config)
        elif page == "Extensions":
            # Use the current config from session state or default
            current_config = st.session_state.get('current_config', default_config)
            render_extensions_section(aws_manager, weaviate_manager, config_manager, current_config)
        
        # Render footer
        render_footer()
        
        # Update last navigation time
        st.session_state['last_navigation'] = datetime.now()
        
    except Exception as e:
        # Handle any unhandled exceptions
        handle_application_error(e)


if __name__ == "__main__":
    main()