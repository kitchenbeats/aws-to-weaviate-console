import threading
import subprocess
import time
from datetime import datetime
from typing import Dict, List
from services.pipeline_status import PipelineStatus

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