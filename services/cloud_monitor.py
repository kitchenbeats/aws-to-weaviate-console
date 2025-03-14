from datetime import datetime

class CloudProcessMonitor:
    """
    Monitors cloud-based processes by connecting to AWS services
    to retrieve the current state of running jobs.
    
    This class doesn't maintain processes (the cloud does that),
    it just provides visibility into their current state.
    """
    
    def __init__(self, region=None):
        """Initialize with optional AWS region."""
        self.region = region
        self.sagemaker_client = None
        self.s3_client = None
        self.initialized = False
        self.process_history = {}  # Stores info about processes we've started
    
    def initialize(self, region=None):
        """Initialize AWS clients."""
        if region:
            self.region = region
            
        try:
            import boto3
            
            self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
            self.s3_client = boto3.client('s3', region_name=self.region)
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize AWS clients: {e}")
            return False
    
    def register_process(self, process_id, job_type, job_name, config):
        """
        Register a cloud process that was started through the UI.
        This doesn't affect the actual process, just records that we started it.
        
        Args:
            process_id: Unique identifier for this process
            job_type: Type of job ('transform', 'training', etc.)
            job_name: AWS job name
            config: Configuration used to start the process
            
        Returns:
            Boolean indicating success
        """
        self.process_history[process_id] = {
            'job_type': job_type,
            'job_name': job_name,
            'config': config,
            'start_time': datetime.now(),
            'last_checked': datetime.now(),
        }
        return True
    
    def get_transform_job_status(self, job_name):
        """
        Get the current status of a SageMaker transform job.
        
        Args:
            job_name: The name of the transform job
            
        Returns:
            Dict with job status information
        """
        if not self.initialized:
            if not self.initialize():
                return {'status': 'unknown', 'error': 'AWS client not initialized'}
        
        try:
            response = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            
            # Extract relevant information
            status = response['TransformJobStatus']
            creation_time = response['CreationTime']
            model_name = response['ModelName']
            instance_type = response['TransformResources']['InstanceType']
            instance_count = response['TransformResources']['InstanceCount']
            
            # Get additional details based on status
            details = {}
            if status == 'Completed':
                details['end_time'] = response['TransformEndTime']
            elif status == 'Failed':
                if 'FailureReason' in response:
                    details['failure_reason'] = response['FailureReason']
            
            return {
                'status': status.lower(),
                'creation_time': creation_time,
                'model_name': model_name,
                'instance_type': instance_type,
                'instance_count': instance_count,
                'details': details
            }
        except Exception as e:
            print(f"Error getting transform job status for {job_name}: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def list_active_transform_jobs(self):
        """
        List all currently active SageMaker transform jobs.
        
        Returns:
            List of active transform jobs
        """
        if not self.initialized:
            if not self.initialize():
                return []
        
        try:
            # Get jobs with status "InProgress"
            response = self.sagemaker_client.list_transform_jobs(
                StatusEquals='InProgress'
            )
            
            jobs = []
            for job in response['TransformJobSummaries']:
                jobs.append({
                    'job_name': job['TransformJobName'],
                    'status': 'running',
                    'creation_time': job['CreationTime'],
                    'model_name': job['ModelName'],
                    'last_modified': job.get('LastModifiedTime', job['CreationTime'])
                })
            
            return jobs
        except Exception as e:
            print(f"Error listing active transform jobs: {e}")
            return []
    
    def update_all_process_statuses(self):
        """
        Update the status of all processes we have registered.
        This connects to AWS to get the current state of each job.
        
        Returns:
            Dict mapping process_id to current status
        """
        statuses = {}
        
        for process_id, process_info in self.process_history.items():
            job_type = process_info['job_type']
            job_name = process_info['job_name']
            
            if job_type == 'transform':
                status = self.get_transform_job_status(job_name)
                statuses[process_id] = status
            else:
                # Handle other job types as needed
                statuses[process_id] = {'status': 'unknown', 'error': f'Unsupported job type: {job_type}'}
            
            # Update last checked time
            self.process_history[process_id]['last_checked'] = datetime.now()
        
        return statuses
    
    def stop_transform_job(self, job_name):
        """
        Stop a running SageMaker transform job.
        
        Args:
            job_name: The name of the transform job to stop
            
        Returns:
            Boolean indicating success
        """
        if not self.initialized:
            if not self.initialize():
                return False
        
        try:
            self.sagemaker_client.stop_transform_job(
                TransformJobName=job_name
            )
            return True
        except Exception as e:
            print(f"Error stopping transform job {job_name}: {e}")
            return False