#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA Consultant AI - Premium UI Interface
----------------------------------------
A beautiful, professional UI for the FDA Consultant AI pipeline.

Features:
- Clean, modern interface with premium aesthetics
- Save/load configuration profiles
- Real-time progress monitoring
- Detailed analytics and visualizations
- One-click deployment
"""

import os
import json
import time
import yaml
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional
import atexit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3

# Configure page layout
st.set_page_config(
    page_title="FDA Consultant AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look and feel
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
    
    /* Profile selector dropdown custom styling */
    div[aria-controls="react-select-2-listbox"] {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--primary);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border-radius: 10px;
        border: 2px dashed var(--primary);
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: var(--border);
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

# Utility Functions
class ConfigManager:
    """Handles saving and loading of configuration profiles."""
    
    def __init__(self, config_dir: str = "~/.fda_ai"):
        self.config_dir = os.path.expanduser(config_dir)
        os.makedirs(self.config_dir, exist_ok=True)
        self.profiles_file = os.path.join(self.config_dir, "profiles.json")
        self.load_profiles()
    
    def load_profiles(self) -> Dict:
        """Load all saved configuration profiles."""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"Error loading profiles: {str(e)}")
                return {"profiles": [], "default": None}
        return {"profiles": [], "default": None}
    
    def save_profile(self, name: str, config: Dict, set_default: bool = False) -> None:
        """Save a configuration profile."""
        try:
            profiles_data = self.load_profiles()
            
            # Check if profile exists and update it
            profile_exists = False
            for i, profile in enumerate(profiles_data["profiles"]):
                if profile["name"] == name:
                    profiles_data["profiles"][i]["config"] = config
                    profiles_data["profiles"][i]["last_updated"] = datetime.now().isoformat()
                    profile_exists = True
                    break
            
            # Add new profile if it doesn't exist
            if not profile_exists:
                profiles_data["profiles"].append({
                    "name": name,
                    "config": config,
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                })
            
            # Set as default if requested
            if set_default:
                profiles_data["default"] = name
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save profile: {str(e)}")
    
    def get_profile(self, name: str) -> Optional[Dict]:
        """Get a specific profile by name."""
        try:
            profiles_data = self.load_profiles()
            for profile in profiles_data["profiles"]:
                if profile["name"] == name:
                    return profile["config"]
        except Exception as e:
            st.error(f"Error retrieving profile: {str(e)}")
        return None
    
    def get_profile_names(self) -> List[str]:
        """Get a list of all profile names."""
        try:
            profiles_data = self.load_profiles()
            return [profile["name"] for profile in profiles_data["profiles"]]
        except Exception as e:
            st.error(f"Error retrieving profile names: {str(e)}")
            return []
    
    def get_default_profile(self) -> Optional[Dict]:
        """Get the default profile."""
        try:
            profiles_data = self.load_profiles()
            default_name = profiles_data.get("default")
            if default_name:
                return self.get_profile(default_name)
        except Exception as e:
            st.error(f"Error retrieving default profile: {str(e)}")
        return None
    
    def delete_profile(self, name: str) -> None:
        """Delete a profile by name."""
        try:
            profiles_data = self.load_profiles()
            profiles_data["profiles"] = [p for p in profiles_data["profiles"] if p["name"] != name]
            
            # Update default if deleted
            if profiles_data["default"] == name:
                profiles_data["default"] = None if not profiles_data["profiles"] else profiles_data["profiles"][0]["name"]
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
        except Exception as e:
            st.error(f"Failed to delete profile: {str(e)}")

def cleanup_application():
    """Clean up application resources before shutdown."""
    # Access global instances if needed
    if 'pipeline_runner' in globals():
        try:
            global pipeline_runner
            if hasattr(pipeline_runner, 'stop') and pipeline_runner.is_running:
                pipeline_runner.stop()
            
            # Close any open clients
            if hasattr(pipeline_runner, 'aws_client') and pipeline_runner.aws_client:
                try:
                    pipeline_runner.aws_client.close()
                except:
                    pass
        except Exception as e:
            print(f"Error during pipeline cleanup: {e}")
    
    # Clean up AWS manager
    if 'aws_manager' in globals():
        try:
            global aws_manager
            if hasattr(aws_manager, 's3_client') and aws_manager.s3_client:
                try:
                    aws_manager.s3_client._endpoint.http_session.close()
                except:
                    pass
            if hasattr(aws_manager, 'sagemaker_client') and aws_manager.sagemaker_client:
                try:
                    aws_manager.sagemaker_client._endpoint.http_session.close()
                except:
                    pass
        except Exception as e:
            print(f"Error during AWS cleanup: {e}")
    
    # Clean up Weaviate manager
    if 'weaviate_manager' in globals():
        try:
            global weaviate_manager
            if hasattr(weaviate_manager, 'close'):
                weaviate_manager.close()
            elif hasattr(weaviate_manager, 'client') and weaviate_manager.client:
                try:
                    weaviate_manager.client.close()
                except:
                    pass
        except Exception as e:
            print(f"Error during Weaviate cleanup: {e}")
    
    print("Application cleanup completed")

# Register the cleanup function to run at exit
atexit.register(cleanup_application)

class PipelineStatus:
    """Tracks detailed status information for the pipeline with improved resource tracking."""
    
    def __init__(self):
        self.status = "idle"  # idle, running, paused, completed, failed, stopped
        self.current_stage = ""  # e.g., "preparing", "uploading", "processing", "embedding", etc.
        self.progress = 0.0  # 0.0 to 1.0
        self.stage_progress = 0.0  # 0.0 to 1.0 for current stage
        self.start_time = None
        self.end_time = None
        self.active_instances = []  # List of active SageMaker instances
        self.processed_files = 0
        self.total_files = 0
        self.errors = []
        self.warnings = []
        self.metrics = {}  # Additional metrics
        self.last_update = None
        self.lock = threading.Lock()
    
    def update(self, **kwargs):
        """Update status fields."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.last_update = datetime.now()
    
    def parse_output(self, line: str):
        """Parse output line for status information."""
        with self.lock:
            # Update last activity timestamp
            self.last_update = datetime.now()
            
            # Basic stage detection
            if "Starting FDA Consultant AI pipeline" in line:
                self.status = "running"
                self.current_stage = "initialization"
            elif "Listing CSV files in bucket" in line:
                self.current_stage = "scanning_input"
            elif "Creating transform job:" in line:
                self.current_stage = "creating_transform_job"
                # Extract job name to track
                job_name = line.split("Creating transform job: ")[-1].strip()
                if job_name and job_name not in [i.get("job_name") for i in self.active_instances]:
                    self.active_instances.append({
                        "job_name": job_name,
                        "status": "creating",
                        "instance_type": self._extract_instance_type(line),
                        "start_time": datetime.now()
                    })
            elif "Job" in line and "using spot instances" in line:
                job_name = line.split("Job ")[1].split(" using")[0].strip()
                # Update instance info
                for instance in self.active_instances:
                    if instance["job_name"] == job_name:
                        instance["spot"] = True
                        # Try to extract instance type if we didn't get it before
                        if instance["instance_type"] == "unknown":
                            instance["instance_type"] = self._extract_instance_type(line)
            elif "Transform job" in line and "completed successfully" in line:
                job_name = line.split("Transform job ")[1].split(" completed")[0].strip()
                # Update instance status
                for instance in self.active_instances:
                    if instance["job_name"] == job_name:
                        instance["status"] = "completed"
                        instance["end_time"] = datetime.now()
            elif "Transform job" in line and "failed" in line:
                job_name = line.split("Transform job ")[1].split(" failed")[0].strip()
                # Update instance status
                for instance in self.active_instances:
                    if instance["job_name"] == job_name:
                        instance["status"] = "failed"
                        instance["end_time"] = datetime.now()
            elif "Created combined output with" in line:
                # Extract record count
                try:
                    records = int(line.split("with ")[1].split(" records")[0])
                    self.metrics["processed_records"] = self.metrics.get("processed_records", 0) + records
                except:
                    pass
            elif "Successfully processed file" in line:
                self.processed_files += 1
                if self.total_files > 0:
                    self.progress = min(1.0, self.processed_files / self.total_files)
            elif "Found" in line and "CSV files in S3" in line:
                try:
                    self.total_files = int(line.split("Found ")[1].split(" CSV")[0])
                except:
                    pass
            elif "Pipeline completed:" in line:
                self.status = "completed"
                self.end_time = datetime.now()
                # Try to extract completion stats
                try:
                    success_count = int(line.split("completed: ")[1].split("/")[0])
                    total_count = int(line.split("/")[1].split(" files")[0])
                    self.processed_files = success_count
                    self.total_files = total_count
                    self.progress = success_count / total_count if total_count > 0 else 1.0
                except:
                    pass
            elif "ERROR" in line or "Error" in line:
                if line not in self.errors:
                    self.errors.append(line)
            elif "WARNING" in line or "Warning" in line:
                if line not in self.warnings:
                    self.warnings.append(line)
    
    def _extract_instance_type(self, line):
        """Extract instance type from log line if present."""
        instance_types = [
            "ml.t3.medium", "ml.t3.large", "ml.t3.xlarge", "ml.t3.2xlarge",
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge", "ml.m5.12xlarge",
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge", "ml.c5.9xlarge", "ml.c5.18xlarge",
            "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge", "ml.g4dn.8xlarge", "ml.g4dn.12xlarge", "ml.g4dn.16xlarge",
            "ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge"
        ]
        
        for instance_type in instance_types:
            if instance_type in line:
                return instance_type
        
        return "unknown"
    
    def get_active_jobs(self):
        """Get currently active SageMaker jobs."""
        with self.lock:
            return [i for i in self.active_instances if i.get("status") != "completed" and i.get("status") != "failed"]
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        with self.lock:
            if not self.start_time:
                return 0
            end = self.end_time if self.end_time else datetime.now()
            return (end - self.start_time).total_seconds()
    
    def get_status_summary(self):
        """Get a summary of the current status."""
        with self.lock:
            return {
                "status": self.status,
                "current_stage": self.current_stage,
                "progress": self.progress,
                "elapsed_time": self.get_elapsed_time(),
                "processed_files": self.processed_files,
                "total_files": self.total_files,
                "active_jobs": len(self.get_active_jobs()),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "resources": len(self.active_instances)
            }
    
    def clear_status(self):
        """Reset status tracking."""
        with self.lock:
            self.status = "idle"
            self.current_stage = ""
            self.progress = 0.0
            self.stage_progress = 0.0
            # Don't clear active_instances as we want to keep history
            self.processed_files = 0
            self.total_files = 0
            self.errors = []
            self.warnings = []
            self.metrics = {}
            self.last_update = datetime.now()


class EnhancedPipelineRunner:
    """Enhanced version of PipelineRunner with better status tracking and control."""
    
    def __init__(self, pipeline_script: str = "fda-consultant-pipeline.py"):
        self.pipeline_script = pipeline_script
        self.process = None
        self.output = []
        self.is_running = False
        self.output_lock = threading.Lock()
        self.status = PipelineStatus()
        self.allow_pause = False  # Set to True if the pipeline supports pausing
        self.is_paused = False
        self.aws_client = None
        
        # Prevent thread issues by tracking initialization
        self._initialized = True
    
    def initialize_aws_client(self, region: str):
        """Initialize AWS client for additional control operations."""
        try:
            import boto3
            self.aws_client = boto3.client('sagemaker', region_name=region)
            return True
        except Exception as e:
            print(f"Failed to initialize AWS client: {str(e)}")
            return False
    
    def build_command(self, config: Dict) -> List[str]:
        """Build the command to run the pipeline with corrected argument handling."""
        command = ["python", "-u", self.pipeline_script]
        
        # Special argument mappings - match script expectations
        arg_mappings = {
            "max_concurrent_transforms": "max-concurrent",
            "input_s3_prefix": "input-prefix",
            "output_s3_prefix": "output-prefix"
        }
        
        # Add all arguments from config
        for key, value in config.items():
            if key.startswith("_"):  # Skip internal keys
                continue
                
            # Get the correct argument name using mappings
            if key in arg_mappings:
                arg_name = f"--{arg_mappings[key]}"
            else:
                arg_name = f"--{key.replace('_', '-')}"
            
            # Handle different types of values
            if isinstance(value, bool):
                if value:
                    command.append(arg_name)
            elif value is not None:
                command.append(arg_name)
                command.append(str(value))
        
        return command
    
    def run(self, config: Dict) -> None:
        """Run the pipeline with the given configuration."""
        if self.is_running:
            print("Pipeline is already running")
            return
        
        # Reset status
        self.status = PipelineStatus()
        self.status.start_time = datetime.now()
        self.status.status = "running"
        
        # Initialize AWS client if needed
        if config.get("region") and not self.aws_client:
            self.initialize_aws_client(config.get("region"))
        
        try:
            command = self.build_command(config)
            self.is_running = True
            
            # Reset output with thread safety
            with self.output_lock:
                self.output = []
                self.output.append(f"Starting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.output.append(f"Command: {' '.join(command)}")
            
            # Start the process and capture output
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start output capture thread
            output_thread = threading.Thread(target=self._capture_output)
            output_thread.daemon = True
            output_thread.start()
            
            return True
        except Exception as e:
            self.is_running = False
            self.status.status = "failed"
            
            # Log the error with thread safety
            with self.output_lock:
                self.output.append(f"ERROR: Failed to start pipeline: {str(e)}")
            
            print(f"Failed to start pipeline: {str(e)}")
            return False
    
    def _capture_output(self) -> None:
        """Capture output from the pipeline process."""
        try:
            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Thread-safe output update
                with self.output_lock:
                    self.output.append(line)
                
                # Parse the line for status information
                self.status.parse_output(line)
            
            self.process.stdout.close()
            return_code = self.process.wait()
            
            # Handle process completion
            with self.output_lock:
                self.output.append(f"Process exited with code {return_code}")
                
                if return_code != 0:
                    self.output.append(f"ERROR: Pipeline process failed with code {return_code}")
                    self.status.status = "failed"
                else:
                    self.output.append("Pipeline process completed successfully")
                    self.status.status = "completed"
            
            self.status.end_time = datetime.now()
            
        except Exception as e:
            # Log the exception with thread safety
            with self.output_lock:
                self.output.append(f"ERROR: Exception in output capture: {str(e)}")
            
            self.status.status = "failed"
            print(f"Error capturing pipeline output: {str(e)}")
        finally:
            self.is_running = False
            self.is_paused = False
    
    def get_output(self) -> List[str]:
        """Get the current output from the pipeline."""
        with self.output_lock:
            return self.output.copy()
    
    def get_errors(self) -> List[str]:
        """Get any errors that occurred during pipeline execution."""
        return [line for line in self.get_output() if "ERROR" in line or "Error" in line]
    
    def get_warnings(self) -> List[str]:
        """Get any warnings that occurred during pipeline execution."""
        return [line for line in self.get_output() if "WARNING" in line or "Warning" in line]
    
    def get_status_summary(self) -> Dict:
        """Get a summary of the current status."""
        return self.status.get_status_summary()
    
    def stop(self) -> bool:
        """Stop the running pipeline and clean up resources."""
        if not self.is_running:
            return True
            
        if self.process:
            try:
                # Add a log entry first
                with self.output_lock:
                    self.output.append(f"Stopping pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Stop the process
                self.process.terminate()
                
                # Give it a moment to terminate gracefully
                for _ in range(5):  # wait up to 5 seconds
                    if self.process.poll() is not None:
                        break
                    time.sleep(1)
                
                # Force kill if still running
                if self.process.poll() is None:
                    with self.output_lock:
                        self.output.append("Process did not terminate gracefully, forcing kill")
                    self.process.kill()
                
                # Clean up any active SageMaker jobs if AWS client is available
                if self.aws_client:
                    active_jobs = self.status.get_active_jobs()
                    for job in active_jobs:
                        try:
                            job_name = job.get("job_name")
                            if job_name:
                                with self.output_lock:
                                    self.output.append(f"Stopping SageMaker job: {job_name}")
                                self.aws_client.stop_transform_job(TransformJobName=job_name)
                                job["status"] = "stopped"
                        except Exception as e:
                            with self.output_lock:
                                self.output.append(f"ERROR: Failed to stop job {job.get('job_name')}: {str(e)}")
                
                self.is_running = False
                self.status.status = "stopped"
                
                with self.output_lock:
                    self.output.append("Pipeline stopped successfully")
                
                return True
            except Exception as e:
                with self.output_lock:
                    self.output.append(f"ERROR: Failed to stop pipeline: {str(e)}")
                print(f"Error stopping pipeline: {str(e)}")
                return False
        return True
    
    def get_elapsed_time(self) -> float:
        """Get the elapsed time since the pipeline started."""
        return self.status.get_elapsed_time()
    
    def get_active_instances(self) -> List[Dict]:
        """Get information about active SageMaker instances."""
        return self.status.active_instances
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        self.stop()


class AWSManager:
    """Handles AWS operations for the FDA Consultant AI."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.s3_client = None
        self.sagemaker_client = None
        self.initialized = False
    
    def initialize(self, region_name: Optional[str] = None) -> bool:
        """Initialize AWS clients."""
        if region_name:
            self.region_name = region_name
            
        try:
            self.s3_client = boto3.client('s3', region_name=self.region_name)
            self.sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize AWS clients: {e}")
            self.initialized = False
            return False
    
    def list_s3_buckets(self) -> List[str]:
        """List all S3 buckets."""
        if not self.initialized:
            self.initialize()
            
        if not self.initialized:
            return []
            
        try:
            response = self.s3_client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        except Exception as e:
            st.error(f"Failed to list S3 buckets: {e}")
            return []
    
    def list_s3_prefixes(self, bucket: str, prefix: str = "", delimiter: str = "/") -> List[str]:
        """List all prefixes (folders) in an S3 bucket with a given prefix."""
        if not self.initialized:
            self.initialize()
            
        if not self.initialized:
            return []
            
        try:
            prefixes = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
                if 'CommonPrefixes' in page:
                    for common_prefix in page['CommonPrefixes']:
                        prefixes.append(common_prefix['Prefix'])
            
            return prefixes
        except Exception as e:
            st.error(f"Failed to list S3 prefixes in bucket {bucket}: {e}")
            return []
    
    def list_sagemaker_models(self) -> List[str]:
        """List all SageMaker models."""
        if not self.initialized:
            self.initialize()
            
        if not self.initialized:
            return []
            
        try:
            response = self.sagemaker_client.list_models()
            return [model['ModelName'] for model in response['Models']]
        except Exception as e:
            st.error(f"Failed to list SageMaker models: {e}")
            return []
    
    def get_s3_file_count(self, bucket: str, prefix: str = "", suffix: str = "") -> int:
        """Count files in an S3 bucket with a given prefix and suffix."""
        # Validate bucket name before proceeding
        if not bucket or bucket.strip() == "":
            return 0
                
        if not self.initialized:
            self.initialize()
                
        if not self.initialized:
            return 0
                
        try:
            count = 0
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if suffix and not obj['Key'].endswith(suffix):
                            continue
                        count += 1
            return count
        except Exception as e:
            st.error(f"Failed to count files in S3 bucket: {e}")
            return 0
    
    def get_sagemaker_instance_types(self) -> List[str]:
        """Get a list of common SageMaker instance types."""
        return [
            "ml.t3.medium",
            "ml.t3.large",
            "ml.t3.xlarge",
            "ml.t3.2xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.c5.large",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.12xlarge",
            "ml.g4dn.16xlarge",
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge"
        ]


class WeaviateManager:
    """Handles Weaviate operations for the FDA Consultant AI using client v4."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.initialized = False
    
    def initialize(self, url: Optional[str] = None, api_key: Optional[str] = None) -> bool:
        """Initialize Weaviate client."""
        if url:
            self.url = url
        if api_key:
            self.api_key = api_key
            
        if not self.url:
            return False
            
        try:
            import weaviate
            from weaviate.classes.init import Auth
            
            # Remove protocol prefix if present for proper connection
            clean_url = self.url
            for prefix in ["http://", "https://"]:
                if clean_url.startswith(prefix):
                    clean_url = clean_url[len(prefix):]
                    break
            
            # Connect based on URL type
            if ".weaviate.network" in clean_url or ".weaviate.cloud" in clean_url:
                # Connect to Weaviate Cloud
                if self.api_key:
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=clean_url,
                        auth_credentials=Auth.api_key(self.api_key)
                    )
                else:
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=clean_url
                    )
            else:
                # Connect to self-hosted Weaviate
                if self.api_key:
                    self.client = weaviate.connect_to_local(
                        host=clean_url,
                        auth_credentials=Auth.api_key(self.api_key)
                    )
                else:
                    self.client = weaviate.connect_to_local(
                        host=clean_url
                    )
            
            # Test connection - just try to list collections instead of getting cluster status
            self.client.collections.list_all()
            
            self.initialized = True
            return True
            
        except ImportError:
            st.error("Weaviate client not installed. Run 'pip install weaviate-client'.")
            self.initialized = False
            return False
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {e}")
            self.initialized = False
            return False
    
    def __del__(self):
        """Ensure client is closed on garbage collection."""
        self.close()
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors during closing
            self.client = None
            self.initialized = False
    
    def get_schema(self) -> Dict:
        """Get the Weaviate schema."""
        if not self.initialized and not self.initialize():
            return {}
            
        try:
            # Get all collections
            collections = self.client.collections.list_all()
            
            # Format as schema similar to v3 API
            schema = {
                "classes": [
                    {
                        "class": collection.name,
                        "description": getattr(collection, "description", ""),
                        "properties": [
                            {
                                "name": prop.name,
                                "dataType": [prop.data_type.value],
                                "description": getattr(prop, "description", "")
                            }
                            for prop in collection.properties
                        ]
                    }
                    for collection in collections
                ]
            }
            
            return schema
        except Exception as e:
            st.error(f"Failed to get Weaviate schema: {e}")
            return {}
    
    def get_class_names(self) -> List[str]:
        """Get a list of all class names (collections) in Weaviate."""
        if not self.initialized and not self.initialize():
            return []
            
        try:
            collections = self.client.collections.list_all()
            return [collection.name for collection in collections]
        except Exception as e:
            st.error(f"Failed to get collection names: {e}")
            return []
    
    def get_class_count(self, class_name: str) -> int:
        """Get the count of objects in a collection."""
        if not self.initialized and not self.initialize():
            return 0
            
        try:
            # Check if the collection exists
            try:
                collection = self.client.collections.get(class_name)
            except ValueError:
                # Collection doesn't exist
                return 0
                
            # Get the count using aggregate
            try:
                aggregate_result = collection.aggregate.over_all(total_count=True)
                return aggregate_result.total_count
            except Exception as e:
                st.error(f"Error getting collection count: {e}")
                return 0
                
        except Exception as e:
            st.error(f"Failed to get count for collection {class_name}: {e}")
            return 0

# Main App Components
def render_header():
    """Render the application header."""
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.title("FDA Consultant AI")
        st.markdown("#### The Ultimate PubMed Data Processing & Embedding Pipeline")
    
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 1rem;">
                <span class="status-indicator status-active"></span> <span style="font-weight: 600;">Ready</span>
            </div>
            """, 
            unsafe_allow_html=True
        )


def render_config_section(config_manager, aws_manager, weaviate_manager):
    """Render the configuration section."""
    st.markdown("## Configuration")
    
    # Configuration management
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown("### Profile Settings")
        
        profiles = config_manager.get_profile_names()
        default_profile = config_manager.load_profiles().get("default")
        
        selected_profile = st.selectbox(
            "Configuration Profile",
            ["New Profile"] + profiles,
            index=0 if not profiles else profiles.index(default_profile) + 1 if default_profile in profiles else 0
        )
        
        if selected_profile == "New Profile":
            profile_name = st.text_input("Profile Name", value="Default Profile")
            config = {}
        else:
            profile_name = selected_profile
            config = config_manager.get_profile(profile_name) or {}
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            set_default = st.checkbox("Set as Default", value=selected_profile == default_profile)
        
        with col1b:
            if selected_profile != "New Profile":
                delete_profile = st.button("Delete Profile")
                if delete_profile:
                    config_manager.delete_profile(profile_name)
                    st.rerun()
    
    with col2:
        st.markdown("### Import/Export")
        
        # Import config from file
        uploaded_file = st.file_uploader("Import Configuration", type=["json", "yaml", "yml"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith((".yaml", ".yml")):
                    imported_config = yaml.safe_load(uploaded_file)
                else:
                    imported_config = json.load(uploaded_file)
                
                if isinstance(imported_config, dict):
                    config = imported_config
                    st.success("Configuration imported successfully!")
                else:
                    st.error("Invalid configuration format")
            except Exception as e:
                st.error(f"Failed to import configuration: {e}")
    
    with col3:
        st.markdown("### Quick Setup")
        
        # Quick setup templates
        template = st.selectbox(
            "Template",
            [
                "Select a template...",
                "Standard Processing (8 hours)",
                "Fast Processing (4 hours)",
                "Optimal Performance (7 hours)",
                "Budget Friendly (12 hours)"
            ]
        )
        
        if template != "Select a template...":
            # Define templates
            templates = {
                "Standard Processing (8 hours)": {
                    "instance_type": "ml.g4dn.xlarge",
                    "instance_count": 15,
                    "use_spot_instances": True,
                    "enable_parallel_upload": False,
                    "max_workers": 4
                },
                "Fast Processing (4 hours)": {
                    "instance_type": "ml.g4dn.12xlarge",
                    "instance_count": 40,
                    "use_spot_instances": False,
                    "enable_parallel_upload": True,
                    "chunk_size": 10,
                    "max_workers": 12,
                    "weaviate_upload_workers": 20,
                },
                "Optimal Performance (7 hours)": {
                    "instance_type": "ml.g4dn.12xlarge",
                    "instance_count": 30,
                    "use_spot_instances": True,
                    "enable_parallel_upload": True,
                    "chunk_size": 20,
                    "max_workers": 8,
                    "weaviate_upload_workers": 16,
                },
                "Budget Friendly (12 hours)": {
                    "instance_type": "ml.g4dn.xlarge",
                    "instance_count": 10,
                    "use_spot_instances": True,
                    "enable_parallel_upload": False,
                    "max_workers": 4
                }
            }
            
            config.update(templates.get(template, {}))
            st.success(f"Applied {template} template!")
    
    # Configuration tabs
    tabs = st.tabs(["AWS & S3 Settings", "SageMaker Settings", "Processing Options", "Weaviate Integration"])
    
    # Tab 1: AWS & S3 Settings
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AWS Configuration")
            
            region_name = st.text_input("AWS Region", value=config.get("region", "us-east-1"))
            config["region"] = region_name
            
            # Update AWS manager with region
            aws_manager.initialize(region_name)
            
            # List buckets
            buckets = aws_manager.list_s3_buckets()
            if buckets:
                input_bucket = st.selectbox(
                    "Input S3 Bucket",
                    buckets,
                    index=buckets.index(config.get("input_bucket", "")) if config.get("input_bucket", "") in buckets else 0
                )
                config["input_bucket"] = input_bucket
                
                output_bucket = st.selectbox(
                    "Output S3 Bucket",
                    buckets,
                    index=buckets.index(config.get("output_bucket", "")) if config.get("output_bucket", "") in buckets else 0
                )
                config["output_bucket"] = output_bucket
            else:
                input_bucket = st.text_input("Input S3 Bucket", value=config.get("input_bucket", ""))
                config["input_bucket"] = input_bucket
                
                output_bucket = st.text_input("Output S3 Bucket", value=config.get("output_bucket", ""))
                config["output_bucket"] = output_bucket
        
        with col2:
            # S3 paths section
            st.markdown("#### S3 Path Configuration")
            
            # Input S3 prefix
            input_s3_prefix = st.text_input(
                "Input S3 Prefix", 
                value=config.get("input_s3_prefix", ""),
                help="Folder path within your input bucket where CSV files are located (e.g., 'data/pubmed/')"
            )
            if input_s3_prefix:
                config["input_s3_prefix"] = input_s3_prefix
            elif "input_s3_prefix" in config:
                del config["input_s3_prefix"]
            
            # Output S3 prefix
            output_s3_prefix = st.text_input(
                "Output S3 Prefix", 
                value=config.get("output_s3_prefix", ""),
                help="Folder path within your output bucket where to save results (e.g., 'embeddings/pubmed/')"
            )
            if output_s3_prefix:
                config["output_s3_prefix"] = output_s3_prefix
            elif "output_s3_prefix" in config:
                del config["output_s3_prefix"]
            
            # Temp directory for pipeline processing
            temp_dir = st.text_input(
                "Temporary Directory (optional)", 
                value=config.get("temp_dir", ""),
                help="Directory for temporary processing files. If not specified, a system temporary directory will be used."
            )
            if temp_dir:
                config["temp_dir"] = temp_dir
            elif "temp_dir" in config:
                del config["temp_dir"]
            
            # Checkpointing
            checkpoint_enabled = st.checkbox(
                "Enable Checkpointing", 
                value=config.get("checkpoint_enabled", True),
                help="Enable checkpointing for resilience against interruptions"
            )
            config["checkpoint_enabled"] = checkpoint_enabled
        
        # S3 Browser - Available if input bucket is selected
        if input_bucket:
            st.markdown("#### S3 Browser")
            
            # Current path being browsed, default to input prefix or root
            current_prefix = st.session_state.get('current_prefix', config.get("input_s3_prefix", ""))
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("📂 Root"):
                    current_prefix = ""
                    st.session_state['current_prefix'] = current_prefix
            
            with col2:
                if current_prefix and st.button("⬆️ Up one level"):
                    # Remove the last folder and the trailing slash
                    parts = current_prefix.rstrip('/').split('/')
                    current_prefix = '/'.join(parts[:-1])
                    if current_prefix:
                        current_prefix += '/'
                    st.session_state['current_prefix'] = current_prefix
            
            with col3:
                st.write(f"Current path: `s3://{input_bucket}/{current_prefix}`")
            
            # List prefixes and files
            try:
                # Get folders
                folders = aws_manager.list_s3_prefixes(input_bucket, current_prefix)
                
                # Display folders
                if folders:
                    st.markdown("##### Folders:")
                    folder_cols = st.columns(3)
                    for i, folder in enumerate(folders):
                        # Extract just the folder name from the full prefix
                        folder_name = folder.replace(current_prefix, '').rstrip('/')
                        with folder_cols[i % 3]:
                            if st.button(f"📁 {folder_name}", key=f"folder_{i}"):
                                st.session_state['current_prefix'] = folder
                                st.rerun()
                
                # Show option to select current path for input or output
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use as Input Prefix"):
                        config["input_s3_prefix"] = current_prefix
                        st.success(f"Set input prefix to: {current_prefix}")
                
                with col2:
                    if st.button("Use as Output Prefix"):
                        config["output_s3_prefix"] = current_prefix
                        st.success(f"Set output prefix to: {current_prefix}")
                
                # Show file count
                file_count = aws_manager.get_s3_file_count(input_bucket, current_prefix, ".csv")
                if file_count > 0:
                    st.info(f"Found {file_count} CSV files in this location")
                    
            except Exception as e:
                st.error(f"Error browsing S3: {e}")
    
    # Tab 2: SageMaker Settings
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### SageMaker Model")
            
            # List models
            models = aws_manager.list_sagemaker_models()
            if models:
                model_name = st.selectbox(
                    "Model Name",
                    models,
                    index=models.index(config.get("model_name", "")) if config.get("model_name", "") in models else 0
                )
            else:
                model_name = st.text_input("Model Name", value=config.get("model_name", ""))
            config["model_name"] = model_name
            
            # Spot instances
            use_spot_instances = st.checkbox(
                "Use Spot Instances (70-90% Cost Savings)", 
                value=config.get("use_spot_instances", True),
                help="Use less expensive spot instances for cost optimization"
            )
            config["use_spot_instances"] = use_spot_instances
            
            if use_spot_instances:
                max_spot_retry = st.number_input(
                    "Maximum Spot Retry Attempts",
                    min_value=1,
                    max_value=10,
                    value=config.get("max_spot_retry", 5),
                    help="Number of retries if spot instances are interrupted"
                )
                config["max_spot_retry"] = max_spot_retry
        
        with col2:
            st.markdown("#### Compute Resources")
            
            # Instance type and count
            instance_types = aws_manager.get_sagemaker_instance_types()
            if instance_types:
                instance_type = st.selectbox(
                    "Instance Type",
                    instance_types,
                    index=instance_types.index(config.get("instance_type", "ml.g4dn.xlarge")) if config.get("instance_type", "ml.g4dn.xlarge") in instance_types else 0
                )
            else:
                instance_type = st.text_input("Instance Type", value=config.get("instance_type", "ml.g4dn.xlarge"))
            config["instance_type"] = instance_type
            
            instance_count = st.number_input(
                "Instance Count",
                min_value=1,
                max_value=100,
                value=config.get("instance_count", 1)
            )
            config["instance_count"] = instance_count
            
            max_concurrent = st.number_input(
                "Maximum Concurrent Transform Jobs",
                min_value=1,
                max_value=100,
                value=config.get("max_concurrent_transforms", 10)
            )
            config["max_concurrent_transforms"] = max_concurrent
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=1000,
                value=config.get("batch_size", 100)
            )
            config["batch_size"] = batch_size
    
    # Tab 3: Processing Options
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Parallel Processing")
            
            max_workers = st.number_input(
                "Maximum Worker Threads",
                min_value=1,
                max_value=32,
                value=config.get("max_workers", 4)
            )
            config["max_workers"] = max_workers
            
            enable_parallel_upload = st.checkbox(
                "Enable Parallel Upload (Process & Upload Simultaneously)",
                value=config.get("enable_parallel_upload", False)
            )
            config["enable_parallel_upload"] = enable_parallel_upload
            
            if enable_parallel_upload:
                chunk_size = st.number_input(
                    "Chunk Size (Files to Process Before Upload Starts)",
                    min_value=1,
                    max_value=100,
                    value=config.get("chunk_size", 20)
                )
                config["chunk_size"] = chunk_size

            # Maximum files to process - handle None value for max_files
            max_files_value = config.get("max_files", 0)
            # Convert None to 0 for the UI
            if max_files_value is None:
                max_files_value = 0
                
            max_files = st.number_input(
                "Maximum Files to Process (0 for all)",
                min_value=0,
                value=max_files_value,
                help="Set to 0 to process all files"
            )

            # Store as None if 0, otherwise store the actual value
            config["max_files"] = None if max_files == 0 else max_files
            
            # Log level
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("log_level", "INFO"))
            )
            config["log_level"] = log_level
        
        with col2:
            # Performance analysis and recommendations
            st.markdown("#### Performance Analysis")
            
            # Calculate estimated processing time
            instance_perf = {
                "ml.g4dn.xlarge": 1,
                "ml.g4dn.2xlarge": 1.8,
                "ml.g4dn.4xlarge": 3.5,
                "ml.g4dn.8xlarge": 7,
                "ml.g4dn.12xlarge": 10,
                "ml.g4dn.16xlarge": 14
            }
            
            instance_factor = instance_perf.get(config.get("instance_type", "ml.g4dn.xlarge"), 1)
            count_factor = config.get("instance_count", 1)
            parallel_factor = 1.2 if config.get("enable_parallel_upload", False) else 1
            
            # Base time for processing all files (37M articles)
            base_hours = 48  # Base time with 1 ml.g4dn.xlarge
            
            estimated_hours = base_hours / (instance_factor * count_factor * parallel_factor)
            estimated_cost = (count_factor * (4.95 if not config.get("use_spot_instances", True) else 1.5) * estimated_hours)
            
            st.markdown(f"""
            <div class="metric-container" style="margin-bottom: 1rem;">
                <div class="metric-label">Estimated Processing Time</div>
                <div class="metric-value">{estimated_hours:.1f} hours</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Estimated AWS Cost</div>
                <div class="metric-value">${estimated_cost:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            if estimated_hours > 10:
                st.warning("Consider increasing instance count or using a more powerful instance type to reduce processing time.")
            elif estimated_cost > 500:
                st.warning("Consider using spot instances or reducing instance count to save costs.")
            
            # Scale estimation based on file count
            if input_bucket:
                file_count = aws_manager.get_s3_file_count(input_bucket, config.get("input_s3_prefix", ""), ".csv")
                if file_count > 0:
                    scale_factor = min(file_count / 100, 1.0)  # Assume 100 files is full scale
                    adjusted_hours = estimated_hours * scale_factor
                    adjusted_cost = estimated_cost * scale_factor
                    
                    st.markdown(f"""
                    <div class="metric-container" style="margin-top: 1rem;">
                        <div class="metric-label">Adjusted for {file_count} CSV Files</div>
                        <div class="metric-value">{adjusted_hours:.1f} h / ${adjusted_cost:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 4: Weaviate Integration
    with tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Weaviate Connection")
            
            weaviate_url = st.text_input("Weaviate URL", value=config.get("weaviate_url", ""))
            config["weaviate_url"] = weaviate_url if weaviate_url else None
            
            weaviate_api_key = st.text_input("Weaviate API Key (optional)", value=config.get("weaviate_api_key", ""), type="password")
            config["weaviate_api_key"] = weaviate_api_key if weaviate_api_key else None
            
            # Test connection
            if weaviate_url:
                if st.button("Test Connection"):
                    with st.spinner("Testing connection to Weaviate..."):
                        if weaviate_manager.initialize(weaviate_url, weaviate_api_key):
                            st.success("Successfully connected to Weaviate!")
                        else:
                            st.error("Failed to connect to Weaviate")
        
        with col2:
            st.markdown("#### Weaviate Upload Settings")
            
            weaviate_class_name = st.text_input("Class Name", value=config.get("weaviate_class_name", "Article"))
            config["weaviate_class_name"] = weaviate_class_name
            
            weaviate_batch_size = st.number_input(
                "Batch Size for Uploads",
                min_value=1,
                max_value=1000,
                value=config.get("weaviate_batch_size", 200)
            )
            config["weaviate_batch_size"] = weaviate_batch_size
            
            weaviate_upload_workers = st.number_input(
                "Upload Worker Threads",
                min_value=1,
                max_value=32,
                value=config.get("weaviate_upload_workers", 8)
            )
            config["weaviate_upload_workers"] = weaviate_upload_workers
    
    # Save configuration
    st.markdown("### Save Configuration")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Save Profile"):
            config_manager.save_profile(profile_name, config, set_default=set_default)
            st.success(f"Profile '{profile_name}' saved successfully!")
    
    with col2:
        # Export config
        if st.download_button(
            "Export Configuration",
            data=json.dumps(config, indent=2),
            file_name=f"{profile_name.lower().replace(' ', '_')}_config.json",
            mime="application/json"
        ):
            st.info("Configuration exported successfully!")
     
    # Auto-save configuration to session state for use in other tabs
    if 'current_config' not in st.session_state:
        st.session_state['current_config'] = {}

    # Update the session state with the current config
    st.session_state['current_config'] = config

    # Auto-save configuration to default profile if it has required fields
    required_fields = ["input_bucket", "output_bucket", "model_name"]
    if all(config.get(field) for field in required_fields):
        if st.session_state.get('auto_save_enabled', True):  # Add toggle in sidebar if desired
            config_manager.save_profile("Auto-saved Profile", config, set_default=True)
            if not st.session_state.get('auto_save_notified', False):
                st.success("Configuration auto-saved for use in other tabs")
                st.session_state['auto_save_notified'] = True

    return config       
        

def render_dashboard(aws_manager, weaviate_manager, config):
    """Render the dashboard with metrics and status."""
    # Ensure AWS manager is initialized with the correct region
    if not aws_manager.initialized and config.get("region"):
        aws_manager.initialize(config.get("region"))
        
    # Ensure Weaviate manager is initialized if URL is provided
    if not weaviate_manager.initialized and config.get("weaviate_url"):
        weaviate_manager.initialize(
            config.get("weaviate_url"),
            config.get("weaviate_api_key")
        )
    
    st.markdown("## Dashboard")
    
    # Get bucket info with validation
    input_bucket = config.get("input_bucket", "")
    output_bucket = config.get("output_bucket", "")
    input_prefix = config.get("input_s3_prefix", "")
    output_prefix = config.get("output_s3_prefix", "")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_files_count = 0
        if input_bucket:  # Only try to count if bucket is not empty
            input_files_count = aws_manager.get_s3_file_count(input_bucket, input_prefix, ".csv")
            
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Input Files</div>
            <div class="metric-value">
                {input_files_count if input_bucket else "N/A"}
            </div>
            <div style="font-size: 0.8rem;">CSV files in input bucket</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        output_files_count = 0
        final_prefix = f"{output_prefix}final/" if output_prefix else "final/"
        if output_bucket:  # Only try to count if bucket is not empty
            output_files_count = aws_manager.get_s3_file_count(output_bucket, final_prefix)
            
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Processed Files</div>
            <div class="metric-value">
                {output_files_count if output_bucket else "N/A"}
            </div>
            <div style="font-size: 0.8rem;">Files with embeddings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Initialize Weaviate if URL is provided
        weaviate_count = 0
        weaviate_class = config.get("weaviate_class_name", "Article")
        if weaviate_manager.initialized:
            weaviate_count = weaviate_manager.get_class_count(weaviate_class)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Weaviate Objects</div>
            <div class="metric-value">{weaviate_count if weaviate_manager.initialized else "N/A"}</div>
            <div style="font-size: 0.8rem;">Articles in Weaviate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Estimate total articles - only if we have a valid count
        csv_count = input_files_count if input_bucket else 0
        total_articles = csv_count * 30000  # Assuming 30k articles per CSV
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Articles</div>
            <div class="metric-value">{total_articles:,}</div>
            <div style="font-size: 0.8rem;">Estimated across all files</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status section
    st.markdown("### System Status")
    
    # AWS status
    aws_status = "Connected" if aws_manager.initialized else "Not Connected"
    aws_color = "status-active" if aws_manager.initialized else "status-error"
    
    # Weaviate status
    weaviate_status = "Connected" if weaviate_manager.initialized else "Not Connected"
    weaviate_color = "status-active" if weaviate_manager.initialized else "status-warning"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">AWS Connection</div>
            <div>
                <span class="status-indicator {aws_color}"></span>
                <span>{aws_status}</span>
            </div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">Region: {config.get("region", "N/A")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Weaviate Connection</div>
            <div>
                <span class="status-indicator {weaviate_color}"></span>
                <span>{weaviate_status}</span>
            </div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">URL: {config.get("weaviate_url", "Not configured")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # S3 path overview
    st.markdown("### S3 Paths Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Input S3 Path</div>
            <div style="font-family: monospace; word-break: break-all;">
                s3://{input_bucket}/{config.get("input_s3_prefix", "")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Output S3 Path</div>
            <div style="font-family: monospace; word-break: break-all;">
                s3://{output_bucket}/{config.get("output_s3_prefix", "")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance visualization
    st.markdown("### Performance Metrics")
    
    # Create dummy performance data or load real data if available
    try:
        # In a real implementation, this would load actual data from S3 or a database
        dates = pd.date_range(start="2023-01-01", periods=30)
        cumulative_files = np.random.randint(0, aws_manager.get_s3_file_count(output_bucket, config.get("output_s3_prefix", ""), "") or 100, size=30)
        cumulative_files.sort()
        
        # Create the plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=dates, y=cumulative_files, name="Processed Files", line=dict(color="#1E88E5")),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(
                x=dates, 
                y=np.diff(np.concatenate([[0], cumulative_files])), 
                name="Daily Processing",
                marker_color="#26A69A"
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Processing Progress Over Time",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Cumulative Files Processed", secondary_y=False)
        fig.update_yaxes(title_text="Files Processed Per Day", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating performance chart: {e}")
        st.info("Performance metrics will be available after pipeline execution.")


class ExecutionStateManager:
    """
    Manages state for the execution page, providing consistent access and updates
    to pipeline status, logs, and configuration.
    
    This class centralizes state management to prevent race conditions and ensure
    consistency across reruns.
    """
    
    def __init__(self, session_state):
        """
        Initialize the state manager with the Streamlit session state.
        
        Parameters:
        -----------
        session_state : SessionState
            The Streamlit session_state object
        """
        self.session_state = session_state
        
        # Initialize state if needed
        self._ensure_state_initialized()
    
    def _ensure_state_initialized(self):
        """Ensure all required state variables are initialized."""
        # Pipeline status
        if 'pipeline_status' not in self.session_state:
            self.session_state['pipeline_status'] = {
                'timestamp': 0,
                'data': {
                    'status': 'idle',
                    'current_stage': '',
                    'progress': 0.0,
                    'elapsed_time': 0,
                    'processed_files': 0,
                    'total_files': 0,
                    'active_jobs': 0,
                    'errors': 0,
                    'warnings': 0
                }
            }
        
        # Log parsing
        if 'parsed_logs' not in self.session_state:
            self.session_state['parsed_logs'] = {'errors': [], 'warnings': []}
        
        if 'log_last_len' not in self.session_state:
            self.session_state['log_last_len'] = 0
        
        # Pipeline running state
        if 'pipeline_started' not in self.session_state:
            self.session_state['pipeline_started'] = False
        
        # UI state
        if 'confirm_stop_requested' not in self.session_state:
            self.session_state['confirm_stop_requested'] = False
    
    def update_status(self, pipeline_runner, force_update=False):
        """
        Update the pipeline status efficiently, only if enough time has passed
        or if force_update is True.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
        force_update : bool, optional
            Whether to force an update regardless of timing, by default False
            
        Returns:
        --------
        dict
            The current status summary
        """
        current_time = time.time()
        should_update = force_update or (current_time - self.session_state['pipeline_status']['timestamp'] > 2)
        
        if should_update:
            try:
                if hasattr(pipeline_runner, 'get_status_summary'):
                    self.session_state['pipeline_status']['data'] = pipeline_runner.get_status_summary()
                else:
                    # Fallback for compatibility
                    self.session_state['pipeline_status']['data'] = {
                        'status': 'running' if pipeline_runner.is_running else 'idle',
                        'current_stage': '',
                        'progress': 0.0,
                        'elapsed_time': pipeline_runner.get_elapsed_time() if hasattr(pipeline_runner, 'get_elapsed_time') else 0,
                        'processed_files': 0,
                        'total_files': 0,
                        'active_jobs': 0,
                        'errors': 0,
                        'warnings': 0
                    }
                self.session_state['pipeline_status']['timestamp'] = current_time
            except Exception as e:
                print(f"Error updating status: {str(e)}")
                # Don't update timestamp to allow retry
        
        return self.session_state['pipeline_status']['data']
    
    def parse_logs(self, pipeline_runner):
        """
        Parse logs from the pipeline runner efficiently, only if logs have changed.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
            
        Returns:
        --------
        tuple
            Tuple containing (output_logs, errors, warnings)
        """
        try:
            # Get output with error handling
            if hasattr(pipeline_runner, 'get_output'):
                output = pipeline_runner.get_output()
            else:
                output = getattr(pipeline_runner, 'output', [])
            
            # Only parse if logs have changed
            if self.session_state.get('log_last_len', 0) != len(output):
                errors = []
                warnings = []
                
                for line in output:
                    if isinstance(line, str):
                        if "ERROR" in line or "Error" in line:
                            errors.append(line)
                        elif "WARNING" in line or "Warning" in line:
                            warnings.append(line)
                
                self.session_state['parsed_logs'] = {'errors': errors, 'warnings': warnings}
                self.session_state['log_last_len'] = len(output)
            
            return (
                output,
                self.session_state['parsed_logs']['errors'],
                self.session_state['parsed_logs']['warnings']
            )
            
        except Exception as e:
            print(f"Error parsing logs: {str(e)}")
            return ([], [], [])
    
    def clear_logs(self, pipeline_runner):
        """
        Clear logs in the pipeline runner and reset parsed logs.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
            
        Returns:
        --------
        bool
            Whether the operation was successful
        """
        try:
            # Safely clear logs
            if hasattr(pipeline_runner, 'output_lock'):
                with pipeline_runner.output_lock:
                    pipeline_runner.output = []
            else:
                if hasattr(pipeline_runner, 'output'):
                    pipeline_runner.output = []
            
            # Clear cached data
            if 'parsed_logs' in self.session_state:
                self.session_state['parsed_logs'] = {'errors': [], 'warnings': []}
            if 'log_last_len' in self.session_state:
                self.session_state['log_last_len'] = 0
            
            return True
        except Exception as e:
            print(f"Error clearing logs: {str(e)}")
            return False
    
    def is_pipeline_running(self, pipeline_runner):
        """
        Check if the pipeline is running, using both the pipeline runner and session state.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
            
        Returns:
        --------
        bool
            Whether the pipeline is running
        """
        runner_status = hasattr(pipeline_runner, 'is_running') and pipeline_runner.is_running
        session_status = self.session_state.get("pipeline_started", False)
        
        return runner_status or session_status
    
    def start_pipeline(self, pipeline_runner, config):
        """
        Start the pipeline in a background thread.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
        config : dict
            The pipeline configuration
            
        Returns:
        --------
        bool
            Whether the pipeline was started successfully
        """
        try:
            # Store pipeline runner in session state for background thread access
            self.session_state['pipeline_runner'] = pipeline_runner
            
            # Start the background thread
            pipeline_thread = threading.Thread(
                target=background_run_pipeline, 
                args=(config,), 
                daemon=True
            )
            pipeline_thread.start()
            
            # Update state
            self.session_state["pipeline_started"] = True
            
            # Set initial status
            self.session_state['pipeline_status']['data'] = {
                'status': 'running',
                'current_stage': 'initialization',
                'progress': 0.0,
                'elapsed_time': 0,
                'processed_files': 0,
                'total_files': 0,
                'active_jobs': 0,
                'errors': 0,
                'warnings': 0
            }
            self.session_state['pipeline_status']['timestamp'] = time.time()
            
            return True
        except Exception as e:
            print(f"Error starting pipeline: {str(e)}")
            self.session_state["pipeline_started"] = False
            return False
    
    def stop_pipeline(self, pipeline_runner):
        """
        Stop the running pipeline.
        
        Parameters:
        -----------
        pipeline_runner : PipelineRunner
            The pipeline runner instance
            
        Returns:
        --------
        bool
            Whether the pipeline was stopped successfully
        """
        try:
            # Stop the pipeline
            result = pipeline_runner.stop()
            
            # Update state
            self.session_state["pipeline_started"] = False
            self.session_state['confirm_stop_requested'] = False
            
            # Update status
            if result and 'pipeline_status' in self.session_state:
                self.session_state['pipeline_status']['data']['status'] = 'stopped'
                self.session_state['pipeline_status']['timestamp'] = time.time()
            
            return result
        except Exception as e:
            print(f"Error stopping pipeline: {str(e)}")
            return False
    
    def reset_execution_state(self):
        """
        Reset the execution state completely. Useful for recovery.
        
        Returns:
        --------
        bool
            Whether the state was reset successfully
        """
        try:
            # Reset critical state variables
            if 'pipeline_status' in self.session_state:
                self.session_state['pipeline_status'] = {
                    'timestamp': 0,
                    'data': {
                        'status': 'idle',
                        'current_stage': '',
                        'progress': 0.0,
                        'elapsed_time': 0,
                        'processed_files': 0,
                        'total_files': 0,
                        'active_jobs': 0,
                        'errors': 0,
                        'warnings': 0
                    }
                }
            
            if 'pipeline_started' in self.session_state:
                self.session_state['pipeline_started'] = False
            
            if 'confirm_stop_requested' in self.session_state:
                self.session_state['confirm_stop_requested'] = False
            
            if 'pipeline_error' in self.session_state:
                del self.session_state['pipeline_error']
            
            if 'pipeline_error_details' in self.session_state:
                del self.session_state['pipeline_error_details']
            
            return True
        except Exception as e:
            print(f"Error resetting execution state: {str(e)}")
            return False

def background_run_pipeline(config):
    """
    Runs the pipeline in a background thread with enhanced error handling and status updates.
    Any exception is stored in session state, and the pipeline_started flag is reset on exit.
    
    Parameters:
    -----------
    config : dict
        The pipeline configuration dictionary
    """
    import traceback
    import time
    from datetime import datetime
    
    # Access the pipeline runner from session state
    if 'pipeline_runner' not in st.session_state:
        st.session_state["pipeline_error"] = "Pipeline runner not initialized"
        st.session_state["pipeline_started"] = False
        return
    
    pipeline_runner = st.session_state['pipeline_runner']
    
    # Track start time
    start_time = time.time()
    
    try:
        # Update status to show we're starting
        if 'pipeline_status' in st.session_state:
            st.session_state['pipeline_status']['data'] = {
                'status': 'running',
                'current_stage': 'initialization',
                'progress': 0.0,
                'elapsed_time': 0,
                'processed_files': 0,
                'total_files': 0,
                'active_jobs': 0,
                'errors': 0,
                'warnings': 0
            }
            st.session_state['pipeline_status']['timestamp'] = time.time()
        
        # Validate config before running
        required_fields = ["input_bucket", "output_bucket", "model_name"]
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        # Start the pipeline
        pipeline_runner.run(config)
        
    except Exception as e:
        # Capture the full error details
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_details = {
            "timestamp": error_time,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "config": {k: v for k, v in config.items() if k not in ["weaviate_api_key"]}  # Exclude sensitive data
        }
        
        st.session_state["pipeline_error"] = str(e)
        st.session_state["pipeline_error_details"] = error_details
        
        # Update status to reflect the error
        if 'pipeline_status' in st.session_state:
            st.session_state['pipeline_status']['data']['status'] = 'failed'
            st.session_state['pipeline_status']['data']['errors'] += 1
            st.session_state['pipeline_status']['timestamp'] = time.time()
        
        # Log the error
        print(f"Pipeline Error: {str(e)}")
        print(traceback.format_exc())
        
    finally:
        # Always reset the pipeline_started flag
        st.session_state["pipeline_started"] = False
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Add to session state for reporting
        st.session_state["last_execution_time"] = execution_time
        
        # Log completion
        print(f"Background pipeline thread completed after {execution_time:.2f} seconds")
        
def render_enhanced_execution_section(pipeline_runner, config_manager):
    """
    Enhanced execution section with added resource monitoring and cancel/pause capabilities.
    """
    st.markdown("## Pipeline Execution")
    
    # ------- CONFIG LOADING -------
    if 'current_config' in st.session_state and st.session_state['current_config']:
        config = st.session_state['current_config']
    elif hasattr(config_manager, 'get_default_profile'):
        config = config_manager.get_default_profile() or {}
    else:
        config = {}
    
    # ------- STATE MANAGEMENT -------
    is_running = getattr(pipeline_runner, 'is_running', False)
    
    # ------- UI LAYOUT -------
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown("### Control Panel")
        
        # Status card with better styling
        status = getattr(pipeline_runner.status, 'status', "idle") if hasattr(pipeline_runner, 'status') else "idle"
        status_color = {
            "idle": "blue",
            "running": "green",
            "paused": "orange",
            "completed": "blue",
            "failed": "red",
            "stopped": "red"
        }.get(status, "blue")
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; background-color: white; 
                    border-left: 5px solid {status_color}; margin-bottom: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-weight: bold; font-size: 16px; text-transform: capitalize;">
                {status}
            </div>
            <div style="color: #666; font-size: 14px;">
                {pipeline_runner.status.current_stage.replace('_', ' ').title() if hasattr(pipeline_runner, 'status') and hasattr(pipeline_runner.status, 'current_stage') and pipeline_runner.status.current_stage else "No active stage"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress display
        try:
            if hasattr(pipeline_runner, 'status'):
                progress = pipeline_runner.status.progress
                st.progress(progress)
                
                # Basic metrics with better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    elapsed_time = pipeline_runner.get_elapsed_time() if hasattr(pipeline_runner, 'get_elapsed_time') else 0
                    hours, remainder = divmod(int(elapsed_time), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    elapsed_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    st.metric("Elapsed Time", elapsed_formatted)
                
                with col2:
                    if hasattr(pipeline_runner.status, 'processed_files') and hasattr(pipeline_runner.status, 'total_files'):
                        st.metric("Files Processed", f"{pipeline_runner.status.processed_files}/{pipeline_runner.status.total_files}")
                    else:
                        st.metric("Files Processed", "0/0")
            else:
                st.progress(0)
        except Exception as e:
            st.warning(f"Could not display progress: {str(e)}")
            st.progress(0)
        
        # ------- RESOURCE MONITORING SECTION -------
        st.markdown("### Active Resources")
        
        try:
            # Get active instances/jobs
            active_instances = []
            
            if hasattr(pipeline_runner, 'get_active_instances'):
                active_instances = pipeline_runner.get_active_instances()
            elif hasattr(pipeline_runner, 'status') and hasattr(pipeline_runner.status, 'active_instances'):
                active_instances = pipeline_runner.status.active_instances
            
            if active_instances:
                # Display each resource in a card
                for idx, instance in enumerate(active_instances):
                    job_name = instance.get("job_name", "Unknown")
                    job_status = instance.get("status", "Unknown")
                    instance_type = instance.get("instance_type", "Unknown")
                    spot = instance.get("spot", False)
                    
                    # Determine status color
                    status_color = {
                        "creating": "#FFC107",  # amber
                        "running": "#4CAF50",   # green
                        "completed": "#2196F3", # blue
                        "failed": "#F44336",    # red
                        "stopped": "#795548"    # brown
                    }.get(job_status.lower(), "#9E9E9E")  # default gray
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: white; 
                                margin-bottom: 10px; border-left: 4px solid {status_color};
                                box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="font-weight: bold; font-size: 14px; margin-bottom: 5px; 
                                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
                             title="{job_name}">
                            {job_name[:20] + '...' if len(job_name) > 20 else job_name}
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 12px;">
                            <span>Status: <span style="color: {status_color}; font-weight: bold;">{job_status}</span></span>
                            <span>Type: {instance_type}</span>
                            <span>{spot and '🔄 Spot' or '💲 On-Demand'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show total resources
                st.info(f"Total active resources: {len(active_instances)}")
                
                # Add button to refresh resource status
                if st.button("🔄 Refresh Resources"):
                    st.rerun()
            else:
                st.markdown("""
                <div style="padding: 15px; border-radius: 8px; background-color: #f5f5f5; 
                            text-align: center; color: #666;">
                    <div style="font-size: 30px; margin-bottom: 10px;">🔍</div>
                    <div>No active resources found</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying resources: {str(e)}")
            st.info("Resource monitoring will be available when the pipeline is running.")
        
        # ------- CONTROL BUTTONS -------
        st.markdown("### Pipeline Controls")
        
        # Check for required configuration
        required_fields = ["input_bucket", "output_bucket", "model_name"]
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            st.error(f"Missing configuration: {', '.join(missing_fields)}")
            st.button("▶️ Start Pipeline", disabled=True)
            st.info("Please complete configuration on the Configuration tab")
        else:
            # Control buttons in a row for better layout
            if is_running:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stop button
                    if st.button("🛑 Stop Pipeline", use_container_width=True):
                        try:
                            with st.spinner("Stopping pipeline..."):
                                pipeline_runner.stop()
                            st.success("Pipeline stopped")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error stopping pipeline: {e}")
                
                with col2:
                    # Pause button (if supported)
                    if hasattr(pipeline_runner, 'pause') and getattr(pipeline_runner, 'allow_pause', False):
                        if pipeline_runner.is_paused:
                            if st.button("▶️ Resume Pipeline", use_container_width=True):
                                try:
                                    pipeline_runner.resume()
                                    st.success("Pipeline resumed")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error resuming pipeline: {e}")
                        else:
                            if st.button("⏸️ Pause Pipeline", use_container_width=True):
                                try:
                                    pipeline_runner.pause()
                                    st.success("Pipeline paused")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error pausing pipeline: {e}")
                    else:
                        st.button("⏸️ Pause Pipeline", disabled=True, use_container_width=True)
                        st.info("Pause functionality is not available for this pipeline", icon="ℹ️")
            else:
                # Start button
                if st.button("▶️ Start Pipeline", use_container_width=True):
                    try:
                        # Run the pipeline
                        result = pipeline_runner.run(config)
                        if result:
                            st.success("Pipeline started!")
                        else:
                            st.error("Failed to start pipeline")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start pipeline: {e}")
        
        # Configuration summary
        with st.expander("Pipeline Configuration"):
            st.write("Input:", f"s3://{config.get('input_bucket', '')}/{config.get('input_s3_prefix', '')}")
            st.write("Output:", f"s3://{config.get('output_bucket', '')}/{config.get('output_s3_prefix', '')}")
            st.write("Model:", config.get('model_name', ''))
            st.write("Instance:", f"{config.get('instance_type', '')} (x{config.get('instance_count', 1)})")
            st.write("Spot Instances:", "Enabled" if config.get('use_spot_instances', False) else "Disabled")
    
    with right_col:
        # Logs section with tabs
        log_tabs = st.tabs(["Output Log", "Errors", "Warnings"])
        
        # Output Log tab
        with log_tabs[0]:
            try:
                # Get logs with error handling
                if hasattr(pipeline_runner, 'get_output'):
                    output = pipeline_runner.get_output()
                else:
                    output = getattr(pipeline_runner, 'output', [])
                
                if output:
                    # Create scrollable log view
                    log_container = st.container()
                    
                    # Show log content - just the last 100 lines to avoid performance issues
                    with log_container:
                        st.code("\n".join(output[-100:]), language=None)
                    
                    # Download and clear buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Full Log",
                            "\n".join(output),
                            file_name=f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🗑️ Clear Logs", use_container_width=True):
                            try:
                                if hasattr(pipeline_runner, 'output_lock'):
                                    with pipeline_runner.output_lock:
                                        pipeline_runner.output = []
                                else:
                                    pipeline_runner.output = []
                                st.success("Logs cleared")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error clearing logs: {e}")
                else:
                    st.info("No logs available. Start the pipeline to see output.")
            except Exception as e:
                st.error(f"Error displaying logs: {e}")
        
        # Errors tab
        with log_tabs[1]:
            try:
                error_logs = [line for line in pipeline_runner.get_output() if "ERROR" in line or "Error" in line]
                
                if error_logs:
                    st.code("\n".join(error_logs), language=None)
                    st.error(f"Found {len(error_logs)} errors")
                else:
                    st.success("No errors detected in the logs")
            except Exception as e:
                st.error(f"Error processing error logs: {e}")
        
        # Warnings tab
        with log_tabs[2]:
            try:
                warning_logs = [line for line in pipeline_runner.get_output() if "WARNING" in line or "Warning" in line]
                
                if warning_logs:
                    st.code("\n".join(warning_logs), language=None)
                    st.warning(f"Found {len(warning_logs)} warnings")
                else:
                    st.success("No warnings detected in the logs")
            except Exception as e:
                st.error(f"Error processing warning logs: {e}")

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

class ExtensionError(Exception):
    """Custom exception for extension-related errors."""
    
    def __init__(self, message, error_code=None, component=None, details=None):
        self.message = message
        self.error_code = error_code
        self.component = component
        self.details = details
        super().__init__(self.message)


def handle_extension_error(func):
    """
    Decorator for handling errors in extension components.
    Catches exceptions and displays user-friendly error messages.
    """
    import functools
    import traceback
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ExtensionError as e:
            # Handle custom extension errors
            error_component = e.component or "Extension"
            st.error(f"**{error_component} Error**: {e.message}")
            
            if e.error_code:
                st.info(f"Error Code: {e.error_code}")
                
            if e.details:
                with st.expander("Error Details"):
                    st.write(e.details)
                    
            st.info("Please check your configuration and try again.")
            return None
        except Exception as e:
            # Handle unexpected errors
            st.error(f"**Unexpected Error**: {str(e)}")
            
            # Show detailed error information in an expander
            with st.expander("Technical Details"):
                st.code(traceback.format_exc(), language="python")
                
            st.info("This error has been logged. Please try again or contact support if the issue persists.")
            return None
    
    return wrapper

def render_extensions_section(aws_manager, weaviate_manager, config_manager, config):
    """Render the extensions section with plugin management, custom integrations, and more."""
    st.markdown("## Extensions")
    
    # Create tabs for different extension categories
    extensions_tabs = st.tabs(["Plugins", "Custom Transformations", "External Integrations", "Advanced Settings"])
    
    # Tab 1: Plugins Management
    with extensions_tabs[0]:
        st.markdown("### Plugin Management")
        
        # Plugin directory display
        plugin_col1, plugin_col2 = st.columns([2, 1])
        
        with plugin_col1:
            # Available plugins
            st.markdown("""
            <div class="section-container">
                <h4>Available Plugins</h4>
                <p>Extend FDA Consultant AI functionality with these verified plugins</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mock plugin list with premium styling
            plugins = [
                {
                    "name": "Advanced Analytics",
                    "version": "1.2.0",
                    "description": "Enhanced analytics and visualization capabilities",
                    "author": "FDA Consultant Team",
                    "status": "installed"
                },
                {
                    "name": "Custom Embeddings",
                    "version": "0.9.5",
                    "description": "Support for custom embedding models and configurations",
                    "author": "BioNLP Labs",
                    "status": "available"
                },
                # Additional plugins...
            ]
            
            # Render each plugin card
            for i, plugin in enumerate(plugins):
                status_color = "#4CAF50" if plugin["status"] == "installed" else "#9E9E9E"
                status_text = "Installed" if plugin["status"] == "installed" else "Available"
                
                st.markdown(f"""
                <div style="padding: 1rem; background-color: white; border-radius: 10px; 
                           margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                           border-left: 4px solid {status_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; font-size: 1.1rem; color: var(--text-primary);">
                                {plugin["name"]}
                            </div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                                v{plugin["version"]} | By {plugin["author"]}
                            </div>
                        </div>
                        <div>
                            <span style="display: inline-block; padding: 0.25rem 0.75rem; 
                                   border-radius: 20px; font-size: 0.8rem; color: white;
                                   background-color: {status_color};">
                                {status_text}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; color: var(--text-secondary);">
                        {plugin["description"]}
                    </div>
                    <div style="margin-top: 0.75rem;">
                        <button style="background-color: {'#F5F5F5' if plugin['status'] == 'installed' else 'var(--primary)'};
                                       color: {'var(--text-secondary)' if plugin['status'] == 'installed' else 'white'};
                                       border: none; padding: 0.25rem 1rem; border-radius: 4px;
                                       cursor: pointer; font-weight: 500; font-size: 0.9rem;">
                            {'Uninstall' if plugin["status"] == "installed" else 'Install'}
                        </button>
                        <button style="background-color: transparent; color: var(--primary);
                                       border: 1px solid var(--primary); padding: 0.25rem 1rem; 
                                       border-radius: 4px; margin-left: 0.5rem;
                                       cursor: pointer; font-weight: 500; font-size: 0.9rem;">
                            View Details
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with plugin_col2:
            st.markdown("""
            <div class="section-container">
                <h4>Plugin Actions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Upload new plugin
            st.file_uploader("Upload Plugin Package (.zip)", type=["zip"], 
                            help="Upload a custom plugin package to extend FDA Consultant AI functionality")
            
            # Plugin marketplace link and other UI elements
            # (more code here...)
    
    # Tab 2: Custom Transformations 
    with extensions_tabs[1]:
        st.markdown("### Custom Data Transformations")
        # (implementation for custom transformations tab)
        
    # Tab 3: External Integrations
    with extensions_tabs[2]:
        st.markdown("### External Service Integrations")
        # (implementation for external integrations tab)
    
    # Tab 4: Advanced Settings
    with extensions_tabs[3]:
        st.markdown("### Advanced Settings")
        # (implementation for advanced settings tab)
        
def main():
    """Main application with simplified execution handling."""
    # Initialize basic session state
    if 'page_initialized' not in st.session_state:
        st.session_state.page_initialized = True
        st.session_state['current_config'] = {}
        st.session_state['current_prefix'] = ""
    
    # Load environment variables
    env_config = load_env_config()
    
    # Initialize managers - streamlined initialization
    config_manager = ConfigManager()
    aws_manager = AWSManager()
    weaviate_manager = WeaviateManager()
    
    # Initialize pipeline runner just once and store in a session state variable
    if 'pipeline_runner' not in st.session_state:
        st.session_state['pipeline_runner'] = EnhancedPipelineRunner()
    pipeline_runner = st.session_state['pipeline_runner']
    
    # Load default config from saved profiles
    saved_config = config_manager.get_default_profile() or {}
    
    # Merge environment config with saved config (env takes precedence)
    default_config = {**saved_config, **env_config}
    
    # Render header
    render_header()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Configuration", "Execution", "Extensions"])
    
    # Initialize AWS connection - simplified error handling
    try:
        region = default_config.get("region", "us-east-1")
        aws_manager.initialize(region)
    except Exception as e:
        st.sidebar.error(f"AWS connection error: {e}")
    
    # Initialize Weaviate connection if URL is provided - simplified
    try:
        if default_config.get("weaviate_url"):
            weaviate_manager.initialize(
                default_config.get("weaviate_url"),
                default_config.get("weaviate_api_key")
            )
    except Exception as e:
        st.sidebar.error(f"Weaviate connection error: {e}")
    
    # Sidebar AWS connection status
    if aws_manager.initialized:
        st.sidebar.success("AWS Connected")
    else:
        st.sidebar.error("AWS Not Connected")
    
    # Sidebar Weaviate connection status - simplified
    if weaviate_manager.initialized:
        st.sidebar.success("Weaviate Connected")
    
    # Main content based on selected page
    if page == "Configuration":
        config = render_config_section(config_manager, aws_manager, weaviate_manager)
        # Store config in session state for other tabs to use
        st.session_state['current_config'] = config
    elif page == "Dashboard":
        st.markdown("## FDA Consultant AI Dashboard")
        
        # Get the current configuration from session state or default profile
        dashboard_config = st.session_state.get('current_config', default_config)
        render_dashboard(aws_manager, weaviate_manager, dashboard_config)
    elif page == "Execution":
        # Streamlined execution page
        render_enhanced_execution_section(pipeline_runner, config_manager)
    elif page == "Extensions":
        # Use the current config from session state or default
        current_config = st.session_state.get('current_config', default_config)
        try:
            # Render the extensions section
            render_extensions_section(aws_manager, weaviate_manager, config_manager, current_config)
        except Exception as e:
            st.error(f"Error loading Extensions tab: {str(e)}")
            st.exception(e)
            st.info("Try refreshing the page or check the console for more details.")
    
    # Footer
    st.markdown("""
    <hr style="margin-top: 2rem;">
    <div style="text-align: center; padding: 1rem 0; color: var(--text-secondary);">
        FDA Consultant AI - Premium Edition &copy; 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)