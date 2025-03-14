import threading
from datetime import datetime

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