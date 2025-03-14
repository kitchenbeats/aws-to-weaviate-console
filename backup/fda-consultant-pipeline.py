#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA Consultant AI - Optimized PubMed Processing Pipeline
--------------------------------------------------------
This script orchestrates the entire process of:
1. Loading PubMed articles from S3
2. Creating and running SageMaker batch transform jobs for embeddings using spot instances
3. Combining the embeddings with original article metadata
4. Uploading to Weaviate in parallel with continued processing
5. Managing checkpoints for resilience against interruptions

Author: FDA AI Team
"""

import os
import json
import time
import logging
import argparse
import tempfile
import threading
import queue
import uuid
from typing import Dict, List, Union, Optional, Tuple, Any, Set
import concurrent.futures
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from botocore.exceptions import ClientError, BotoCoreError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('fda-ai-pipeline')

# Set up a file handler for logging
file_handler = logging.FileHandler(f'fda_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
logger.addHandler(file_handler)


class FDAConsultantPipeline:
    """
    Optimized pipeline class for processing PubMed data, generating embeddings
    using SageMaker batch transform with spot instances, and uploading to Weaviate in parallel.
    """
    
    def __init__(
        self,
        input_bucket: str,
        output_bucket: str,
        model_name: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        max_concurrent_transforms: int = 10,
        batch_size: int = 100,
        region_name: str = "us-east-1",
        temp_dir: Optional[str] = None,
        checkpoint_enabled: bool = True,
        use_spot_instances: bool = False,
        max_spot_retry: int = 5,
        enable_parallel_upload: bool = False,
        chunk_size: int = 20,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None,
        weaviate_class_name: str = "PubMedArticle",
        weaviate_batch_size: int = 200,
        weaviate_upload_workers: int = 8,
        input_prefix: str = "",
        output_prefix: str = ""
    ):
        """
        Initialize the optimized FDA Consultant AI Pipeline.
        
        Args:
            input_bucket: S3 bucket name containing CSV files with PubMed articles
            output_bucket: S3 bucket name for storing the embeddings and final output
            model_name: Name of the SageMaker model to use for embeddings
            instance_type: SageMaker instance type for batch transform
            instance_count: Number of instances to use for batch transform
            max_concurrent_transforms: Maximum number of concurrent transform jobs
            batch_size: Number of records per mini-batch
            region_name: AWS region name
            temp_dir: Directory for temporary files (default: system temp dir)
            checkpoint_enabled: Enable checkpointing for resilience
            use_spot_instances: Use spot instances for cost savings (70-90% cheaper)
            max_spot_retry: Maximum number of retries for spot instance interruptions
            enable_parallel_upload: Start uploading to Weaviate while processing continues
            chunk_size: Number of files to process before starting parallel upload
            weaviate_url: URL of the Weaviate instance (required for parallel upload)
            weaviate_api_key: API key for Weaviate instance
            weaviate_class_name: Name of the Weaviate class to use
            weaviate_batch_size: Batch size for Weaviate uploads
            weaviate_upload_workers: Number of concurrent workers for Weaviate uploads
        """
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.model_name = model_name
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.max_concurrent_transforms = max_concurrent_transforms
        self.batch_size = batch_size
        self.region_name = region_name
        self.checkpoint_enabled = checkpoint_enabled
        self.use_spot_instances = use_spot_instances
        self.max_spot_retry = max_spot_retry
        self.enable_parallel_upload = enable_parallel_upload
        self.chunk_size = chunk_size
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.weaviate_class_name = weaviate_class_name
        self.weaviate_batch_size = weaviate_batch_size
        self.weaviate_upload_workers = weaviate_upload_workers
        
        # Set S3 path prefixes
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        
        # Log S3 path usage
        if self.input_prefix:
            logger.info(f"Using input S3 prefix: {self.input_prefix}")
        if self.output_prefix:
            logger.info(f"Using output S3 prefix: {self.output_prefix}")
        
        # Set up temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="fda_ai_"))
            
        # Set up AWS clients
        try:
            self.s3_client = boto3.client('s3', region_name=self.region_name)
            self.s3_resource = boto3.resource('s3', region_name=self.region_name)
            self.sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
            logger.info(f"AWS clients initialized for region: {self.region_name}")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise

        # Initialize Weaviate client if needed
        self.weaviate_client = None
        if self.enable_parallel_upload and self.weaviate_url:
            self._initialize_weaviate_client()

        self.article_files = []  # List to store CSV file paths
        self.transform_jobs = []  # List to track SageMaker transform jobs
        self.processed_files = set()  # Keep track of processed files for checkpointing
        self.files_for_upload = queue.Queue()  # Queue for files ready to upload to Weaviate
        self.upload_thread = None  # Thread for parallel uploading to Weaviate
        self.upload_stop_event = threading.Event()  # Event to signal upload thread to stop
        
        # Create checkpoint directory
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if it exists
        self.checkpoint_path = self.checkpoint_dir / "pipeline_checkpoint.json"
        if self.checkpoint_enabled and os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
        
        logger.info(f"Optimized FDA Consultant AI Pipeline initialized")
        logger.info(f"Input bucket: {self.input_bucket}")
        logger.info(f"Output bucket: {self.output_bucket}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Instance type: {self.instance_type}, Count: {self.instance_count}")
        logger.info(f"Using spot instances: {self.use_spot_instances}")
        logger.info(f"Parallel upload enabled: {self.enable_parallel_upload}")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def _initialize_weaviate_client(self):
        """Initialize the Weaviate client for parallel uploading."""
        try:
            import weaviate
            from weaviate.exceptions import WeaviateBaseError
            
            auth_config = None
            if self.weaviate_api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.weaviate_api_key)
                
            self.weaviate_client = weaviate.Client(
                url=self.weaviate_url,
                auth_client_secret=auth_config,
                timeout_config=(5, 60)  # (connect_timeout, read_timeout)
            )
            
            # Verify connection
            self.weaviate_client.cluster.get_nodes_status()
            logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            
            # Create schema if not exists
            self._create_weaviate_schema()
            
        except ImportError:
            logger.error("Weaviate client library not installed. Run 'pip install weaviate-client'.")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise
    
    def _create_weaviate_schema(self):
        """Create the Weaviate schema for PubMed articles if it doesn't exist."""
        try:
            # Check if the class already exists
            schema = self.weaviate_client.schema.get()
            existing_classes = [c['class'] for c in schema['classes']] if 'classes' in schema else []
            
            if self.weaviate_class_name in existing_classes:
                logger.info(f"Class {self.weaviate_class_name} already exists in Weaviate schema")
                return
                
            # Define the schema for PubMed articles
            class_obj = {
                "class": self.weaviate_class_name,
                "description": "A medical or scientific article from PubMed with vector embeddings",
                "vectorizer": "none",  # We'll provide our own vectors
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "skip": False,
                    "ef": 256,
                    "efConstruction": 256,
                    "maxConnections": 64,
                    "dynamicEfMin": 100,
                    "dynamicEfMax": 500
                },
                "properties": [
                    {
                        "name": "pmid",
                        "description": "PubMed ID",
                        "dataType": ["string"],
                        "indexInverted": True
                    },
                    {
                        "name": "title",
                        "description": "Article title",
                        "dataType": ["text"],
                        "indexInverted": True,
                        "tokenization": "word"
                    },
                    {
                        "name": "abstract",
                        "description": "Article abstract",
                        "dataType": ["text"],
                        "indexInverted": True,
                        "tokenization": "word"
                    },
                    {
                        "name": "authors",
                        "description": "Article authors",
                        "dataType": ["string[]"],
                        "indexInverted": True
                    },
                    {
                        "name": "journal",
                        "description": "Journal name",
                        "dataType": ["string"],
                        "indexInverted": True
                    },
                    {
                        "name": "publicationDate",
                        "description": "Publication date",
                        "dataType": ["date"],
                        "indexInverted": True
                    },
                    {
                        "name": "meshTerms",
                        "description": "MeSH terms",
                        "dataType": ["string[]"],
                        "indexInverted": True
                    },
                    {
                        "name": "keywords",
                        "description": "Keywords",
                        "dataType": ["string[]"],
                        "indexInverted": True
                    },
                    {
                        "name": "doi",
                        "description": "Digital Object Identifier",
                        "dataType": ["string"],
                        "indexInverted": True
                    },
                    {
                        "name": "url",
                        "description": "URL to the article",
                        "dataType": ["string"],
                        "indexInverted": True
                    },
                    {
                        "name": "source",
                        "description": "Source of the article data",
                        "dataType": ["string"],
                        "indexInverted": True
                    }
                ]
            }
            
            # Create the class
            self.weaviate_client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate schema for class: {self.weaviate_class_name}")
            
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {str(e)}")
            raise
    
    def _save_checkpoint(self):
        """Save the current state of the pipeline for resumability."""
        if not self.checkpoint_enabled:
            return
            
        checkpoint_data = {
            'processed_files': list(self.processed_files),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
            
        logger.debug(f"Saved checkpoint with {len(self.processed_files)} processed files")
    
    def _load_checkpoint(self):
        """Load the pipeline state from a checkpoint."""
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.processed_files = set(checkpoint_data.get('processed_files', []))
            
            logger.info(f"Loaded checkpoint with {len(self.processed_files)} previously processed files")
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting fresh.")
            self.processed_files = set()
    
    def list_input_files(self) -> List[str]:
        """
        List all CSV files in the input S3 bucket with the specified prefix.
        
        Returns:
            List of CSV file keys in the S3 bucket
        """
        logger.info(f"Listing CSV files in bucket: {self.input_bucket}, prefix: {self.input_prefix}")
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            file_keys = []
            
            # Use pagination to handle large number of files
            for page in paginator.paginate(
                Bucket=self.input_bucket, 
                Prefix=self.input_prefix
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.csv'):
                            file_keys.append(key)
            
            logger.info(f"Found {len(file_keys)} CSV files in S3")
            self.article_files = file_keys
            return file_keys
        
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error listing files in S3 bucket {self.input_bucket}: {str(e)}")
            raise
    
    def download_file(self, file_key: str, local_dir: Union[str, Path]) -> Path:
        """
        Download a file from S3 to a local directory.
        
        Args:
            file_key: S3 object key
            local_dir: Local directory to download the file to
            
        Returns:
            Path to the downloaded file
        """
        local_path = Path(local_dir) / Path(file_key).name
        
        try:
            logger.debug(f"Downloading {file_key} to {local_path}")
            self.s3_client.download_file(self.input_bucket, file_key, str(local_path))
            return local_path
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error downloading file {file_key}: {str(e)}")
            raise
    
    def upload_file(self, local_path: Union[str, Path], s3_key: str, bucket: Optional[str] = None) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Path to the local file
            s3_key: S3 object key to upload the file to
            bucket: S3 bucket (defaults to self.output_bucket)
            
        Returns:
            S3 URI of the uploaded file
        """
        if bucket is None:
            bucket = self.output_bucket
            
        # If output prefix is set, prepend it to the key (unless the key already includes it)
        if self.output_prefix and not s3_key.startswith(self.output_prefix):
            # Ensure prefix ends with / if it doesn't already
            prefix = self.output_prefix if self.output_prefix.endswith('/') else f"{self.output_prefix}/"
            s3_key = f"{prefix}{s3_key}"
            
        try:
            logger.debug(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            self.s3_client.upload_file(str(local_path), bucket, s3_key)
            return f"s3://{bucket}/{s3_key}"
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error uploading file {local_path} to {s3_key}: {str(e)}")
            raise
    
    def prepare_input_data(self, file_path: Path) -> Tuple[Path, int]:
        """
        Prepare CSV data for embedding by converting it to the format expected
        by the SageMaker model.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (path to the prepared data file, number of records)
        """
        logger.info(f"Preparing input data from {file_path}")
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check if the DataFrame is empty
            if df.empty:
                logger.warning(f"CSV file {file_path} is empty")
                return None, 0
            
            # Extract the text to embed - typically title and abstract for PubMed
            # Adjust these fields as needed based on your CSV structure
            title_col = next((col for col in df.columns if 'title' in col.lower()), None)
            abstract_col = next((col for col in df.columns if 'abstract' in col.lower()), None)
            
            if not title_col or not abstract_col:
                logger.warning(f"Could not find title or abstract columns in {file_path}")
                # Make a best guess of what to use
                text_cols = [col for col in df.columns if df[col].dtype == 'object'][:2]
                if len(text_cols) >= 2:
                    title_col, abstract_col = text_cols[:2]
                elif len(text_cols) == 1:
                    title_col = text_cols[0]
                    abstract_col = title_col
                else:
                    logger.error(f"No suitable text columns found in {file_path}")
                    return None, 0
                
                logger.info(f"Using columns: {title_col} (title) and {abstract_col} (abstract)")
            
            # Combine title and abstract
            df['text_to_embed'] = df[title_col].fillna('') + " " + df[abstract_col].fillna('')
            
            # Prepare the input format for the model
            # This format depends on your model's input requirements
            # For the pubmedbert model, we'll use a simple JSON lines format
            output_path = self.temp_dir / f"{file_path.stem}_prepared.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in df['text_to_embed']:
                    f.write(json.dumps({"text": text}) + "\n")
            
            record_count = len(df)
            
            # Save the original DataFrame for later merging
            metadata_path = self.temp_dir / f"{file_path.stem}_metadata.parquet"
            df.drop(columns=['text_to_embed']).to_parquet(metadata_path)
            
            logger.info(f"Prepared {record_count} records from {file_path}")
            return output_path, record_count
            
        except Exception as e:
            logger.error(f"Error preparing input data from {file_path}: {str(e)}")
            raise
    
    def create_transform_job(self, input_s3_uri: str, output_s3_uri: str, file_id: str) -> str:
        """
        Create a SageMaker batch transform job with support for spot instances.
        
        Args:
            input_s3_uri: S3 URI for the input data
            output_s3_uri: S3 URI for the output data
            file_id: Identifier for the file being processed
            
        Returns:
            The transform job name
        """
        timestamp = int(time.time())
        job_name = f"fda-ai-transform-{file_id}-{timestamp}"
        
        # Configure the transform job
        transform_job_config = {
            'TransformJobName': job_name,
            'ModelName': self.model_name,
            'MaxConcurrentTransforms': self.max_concurrent_transforms,
            'BatchStrategy': 'MultiRecord',
            'MaxPayloadInMB': 6,  # Adjust based on your data and model
            'TransformInput': {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': 'application/jsonlines',
                'SplitType': 'Line'
            },
            'TransformOutput': {
                'S3OutputPath': output_s3_uri,
                'Accept': 'application/jsonlines',
                'AssembleWith': 'Line'
            },
            'TransformResources': {
                'InstanceType': self.instance_type,
                'InstanceCount': self.instance_count
            }
        }
        
        # Add spot instance configuration if enabled
        if self.use_spot_instances:
            transform_job_config['TransformResources']['VolumeKmsKeyId'] = 'string'
            transform_job_config['TransformResources'].update({
                'VolumeKmsKeyId': '',
                'VpcConfig': {
                    'SecurityGroupIds': [],
                    'Subnets': []
                }
            })
            
            # Add spot instance configuration
            transform_job_config.update({
                'DataProcessing': {
                    'InputFilter': '',
                    'OutputFilter': '',
                    'JoinSource': 'None'
                },
                'Environment': {},
                'ExperimentConfig': {
                    'ExperimentName': '',
                    'TrialName': '',
                    'TrialComponentDisplayName': ''
                }
            })
            
        try:
            response = self.sagemaker_client.create_transform_job(**transform_job_config)
            logger.info(f"Created transform job: {job_name}")
            
            if self.use_spot_instances:
                logger.info(f"Job {job_name} using spot instances for cost savings")
                
            self.transform_jobs.append(job_name)
            return job_name
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error creating transform job: {str(e)}")
            raise
    
    def wait_for_transform_job(self, job_name: str, timeout: int = 3600, retry_on_failure: bool = True) -> bool:
        """
        Wait for a SageMaker transform job to complete with support for retrying
        spot instance interruptions.
        
        Args:
            job_name: The name of the transform job
            timeout: Maximum time to wait in seconds
            retry_on_failure: Whether to retry on spot instance interruption
            
        Returns:
            True if the job completed successfully, False otherwise
        """
        start_time = time.time()
        status = None
        
        logger.info(f"Waiting for transform job {job_name} to complete")
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_transform_job(TransformJobName=job_name)
                status = response['TransformJobStatus']
                
                if status == 'Completed':
                    logger.info(f"Transform job {job_name} completed successfully")
                    return True
                elif status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown reason')
                    logger.error(f"Transform job {job_name} failed: {failure_reason}")
                    
                    # Check if the failure was due to spot instance interruption
                    if retry_on_failure and self.use_spot_instances and 'Spot instance' in failure_reason:
                        logger.info(f"Spot instance was interrupted. Will retry job {job_name}")
                        return False  # Signal for retry
                    else:
                        return False  # Permanent failure
                        
                elif status == 'Stopped':
                    logger.error(f"Transform job {job_name} was stopped")
                    return False
                else:  # InProgress
                    logger.debug(f"Transform job {job_name} still in progress ({status})")
                    time.sleep(30)  # Check every 30 seconds
                    
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Error checking transform job status: {str(e)}")
                time.sleep(10)  # Wait a bit before retrying
        
        logger.error(f"Timeout waiting for transform job {job_name}")
        return False
    
    def process_transform_output(self, job_name: str, metadata_path: Path) -> Tuple[Path, int]:
        """
        Process the output of a transform job and combine with original metadata.
        
        Args:
            job_name: The name of the transform job
            metadata_path: Path to the metadata file
            
        Returns:
            Tuple of (path to the processed output file, number of records)
        """
        logger.info(f"Processing output for transform job {job_name}")
        
        try:
            # Determine the S3 path where the output is stored
            response = self.sagemaker_client.describe_transform_job(TransformJobName=job_name)
            output_path = response['TransformOutput']['S3OutputPath']
            
            # Extract the bucket and prefix from the S3 URI
            output_uri = output_path.replace('s3://', '')
            output_bucket, output_prefix = output_uri.split('/', 1)
            
            # List the output files
            paginator = self.s3_client.get_paginator('list_objects_v2')
            output_files = []
            
            for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        output_files.append(obj['Key'])
            
            if not output_files:
                logger.error(f"No output files found for transform job {job_name}")
                return None, 0
            
            # Download and process each output file
            all_embeddings = []
            
            for file_key in output_files:
                temp_path = self.temp_dir / Path(file_key).name
                self.s3_client.download_file(output_bucket, file_key, str(temp_path))
                
                # Read the embeddings file
                with open(temp_path, 'r') as f:
                    for line in f:
                        embedding_data = json.loads(line)
                        all_embeddings.append(embedding_data)
            
            # Load the original metadata
            metadata_df = pd.read_parquet(metadata_path)
            
            if len(all_embeddings) != len(metadata_df):
                logger.warning(
                    f"Mismatch between number of embeddings ({len(all_embeddings)}) "
                    f"and metadata records ({len(metadata_df)})"
                )
                # Adjust to match the smaller count
                min_count = min(len(all_embeddings), len(metadata_df))
                all_embeddings = all_embeddings[:min_count]
                metadata_df = metadata_df.iloc[:min_count]
                
            # Combine metadata with embeddings
            output_data = []
            
            for i, (_, article) in enumerate(metadata_df.iterrows()):
                if i >= len(all_embeddings):
                    break
                    
                # Convert metadata to a dictionary
                article_dict = article.to_dict()
                
                # Add the embedding
                article_dict['embedding'] = all_embeddings[i].get('embedding', [])
                
                # Clean up null/NaN values for JSON compatibility
                for key, value in article_dict.items():
                    if pd.isna(value):
                        article_dict[key] = None
                
                output_data.append(article_dict)
            
            # Save the combined data
            result_path = self.temp_dir / f"{metadata_path.stem}_with_embeddings.jsonl"
            
            with open(result_path, 'w') as f:
                for article in output_data:
                    f.write(json.dumps(article) + '\n')
            
            logger.info(f"Created combined output with {len(output_data)} records")
            return result_path, len(output_data)
            
        except Exception as e:
            logger.error(f"Error processing transform output: {str(e)}")
            raise
    
    def format_for_weaviate(self, article: Dict) -> Dict:
        """
        Format article properties to match the Weaviate schema.
        
        Args:
            article: Article dictionary to format
            
        Returns:
            Formatted article
        """
        formatted = article.copy()
        
        # Format date fields
        if 'publication_date' in formatted:
            try:
                # Convert to ISO format for Weaviate
                date_str = formatted['publication_date']
                if isinstance(date_str, str) and date_str:
                    # Try to parse the date - accept various formats
                    from dateutil import parser
                    date_obj = parser.parse(date_str)
                    formatted['publicationDate'] = date_obj.strftime('%Y-%m-%d')
                
                # Remove the original field
                del formatted['publication_date']
            except Exception:
                # If date parsing fails, just leave it as is
                pass
                
        # Format list fields
        for list_field in ['authors', 'meshTerms', 'keywords']:
            if list_field.lower() in formatted:
                orig_field = list_field.lower()
                value = formatted[orig_field]
                
                # Convert value to a list if it's a string
                if isinstance(value, str):
                    if ';' in value:
                        formatted[list_field] = [item.strip() for item in value.split(';')]
                    elif ',' in value:
                        formatted[list_field] = [item.strip() for item in value.split(',')]
                    else:
                        formatted[list_field] = [value]
                elif isinstance(value, list):
                    formatted[list_field] = value
                else:
                    formatted[list_field] = []
                    
                # Remove the original field if it's different
                if orig_field != list_field and orig_field in formatted:
                    del formatted[orig_field]
        
        # Ensure all text fields are strings
        for field in ['title', 'abstract', 'journal', 'doi', 'url', 'source']:
            if field.lower() in formatted and not isinstance(formatted[field.lower()], str):
                formatted[field] = str(formatted[field.lower()])
                
                # Remove the original field if it's different
                if field.lower() != field and field.lower() in formatted:
                    del formatted[field.lower()]
                    
        return formatted
    
    def upload_to_weaviate(self, jsonl_path: Path) -> Dict:
        """
        Upload a JSONL file with embeddings to Weaviate.
        
        Args:
            jsonl_path: Path to the JSONL file
            
        Returns:
            Dictionary with upload results
        """
        logger.info(f"Uploading {jsonl_path} to Weaviate")
        
        if not self.weaviate_client:
            logger.error("Weaviate client not initialized")
            return {
                'file': str(jsonl_path),
                'status': 'error',
                'error': 'Weaviate client not initialized'
            }
        
        result = {
            'file': str(jsonl_path),
            'total_articles': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'errors': []
        }
        
        try:
            # Create a batch for uploading
            with self.weaviate_client.batch as batch:
                # Configure the batch
                batch.batch_size = self.weaviate_batch_size
                batch.dynamic = True
                
                # Read and process the file line by line
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(tqdm(f, desc=f"Uploading {jsonl_path.name}", unit="articles")):
                        try:
                            # Parse the JSON object
                            article = json.loads(line)
                            result['total_articles'] += 1
                            
                            # Extract the embedding
                            embedding = article.pop('embedding', None)
                            if not embedding:
                                result['failed_uploads'] += 1
                                result['errors'].append(f"Article at line {i} has no embedding")
                                continue
                                
                            # Format for Weaviate
                            formatted_article = self.format_for_weaviate(article)
                            
                            # Create a unique ID for the article
                            article_id = article.get('pmid', None)
                            if not article_id:
                                # Generate an ID if PMID is not available
                                article_id = str(uuid.uuid4())
                            
                            # Add to batch
                            batch.add_data_object(
                                data_object=formatted_article,
                                class_name=self.weaviate_class_name,
                                uuid=article_id,
                                vector=embedding
                            )
                            
                            result['successful_uploads'] += 1
                            
                        except json.JSONDecodeError as e:
                            result['failed_uploads'] += 1
                            result['errors'].append(f"JSON decode error at line {i}: {str(e)}")
                        except Exception as e:
                            result['failed_uploads'] += 1
                            result['errors'].append(f"Error processing article at line {i}: {str(e)}")
            
            # Log a summary
            logger.info(
                f"Processed {result['total_articles']} articles: "
                f"{result['successful_uploads']} successful, {result['failed_uploads']} failed"
            )
            
            # If there were errors, log the first few
            if result['errors']:
                logger.warning(f"First few errors ({min(5, len(result['errors']))}):")
                for error in result['errors'][:5]:
                    logger.warning(f"  - {error}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error uploading to Weaviate: {str(e)}")
            result['status'] = 'error'
            result['error'] = str(e)
            return result
    
    def process_file(self, file_key: str, retry_count: int = 0) -> Dict:
        """
        Process a single file through the entire pipeline.
        
        Args:
            file_key: S3 key of the file to process
            retry_count: Number of retries attempted so far
            
        Returns:
            Dictionary with processing results
        """
        file_id = Path(file_key).stem
        logger.info(f"Starting processing for file: {file_key} (ID: {file_id})")
        
        # Skip if already processed (from checkpoint)
        if file_key in self.processed_files:
            logger.info(f"Skipping already processed file: {file_key}")
            return {
                'file_key': file_key,
                'file_id': file_id,
                'status': 'skipped',
                'message': 'Already processed'
            }
        
        result = {
            'file_key': file_key,
            'file_id': file_id,
            'status': 'failed',
            'record_count': 0,
            'error': None,
            'output_uri': None
        }
        
        try:
            # Create a subdirectory for this file
            file_dir = self.temp_dir / file_id
            os.makedirs(file_dir, exist_ok=True)
            
            # Download the file
            local_path = self.download_file(file_key, file_dir)
            
            # Prepare the input data
            prepared_path, record_count = self.prepare_input_data(local_path)
            if prepared_path is None:
                result['error'] = f"Failed to prepare input data for {file_key}"
                return result
            
            result['record_count'] = record_count
            
            # Upload the prepared data to S3
            input_s3_key = f"input/{file_id}/data.jsonl"
            input_s3_uri = self.upload_file(prepared_path, input_s3_key)
            
            # Create the output S3 path
            output_s3_uri = f"s3://{self.output_bucket}/output/{file_id}/"
            
            # Create and run the transform job
            job_name = self.create_transform_job(
                input_s3_uri=f"s3://{self.output_bucket}/{input_s3_key}", 
                output_s3_uri=output_s3_uri,
                file_id=file_id
            )
            
            # Wait for the job to complete
            job_success = self.wait_for_transform_job(job_name)
            
            # Handle spot instance interruption
            if not job_success and self.use_spot_instances and retry_count < self.max_spot_retry:
                logger.info(f"Retrying job for {file_key} after spot interruption (attempt {retry_count + 1})")
                return self.process_file(file_key, retry_count + 1)
                
            if not job_success:
                result['error'] = f"Transform job {job_name} failed or timed out"
                return result
            
            # Process the transform output and combine with metadata
            metadata_path = self.temp_dir / f"{local_path.stem}_metadata.parquet"
            output_path, output_count = self.process_transform_output(job_name, metadata_path)
            if output_path is None:
                result['error'] = f"Failed to process transform output for {job_name}"
                return result
            
            # Upload the final result to S3
            final_s3_key = f"final/{file_id}/articles_with_embeddings.jsonl"
            final_s3_uri = self.upload_file(output_path, final_s3_key)
            
            result['status'] = 'success'
            result['output_uri'] = final_s3_uri
            
            # Add to the queue for Weaviate upload if parallel upload is enabled
            if self.enable_parallel_upload and self.weaviate_url:
                self.files_for_upload.put({
                    'path': output_path,
                    'file_key': file_key,
                    'record_count': output_count
                })
            
            # Mark as processed for checkpointing
            self.processed_files.add(file_key)
            self._save_checkpoint()
            
            logger.info(f"Successfully processed file {file_key} with {output_count} articles")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_key}: {str(e)}", exc_info=True)
            result['error'] = str(e)
            return result
    
    def _upload_worker(self):
        """Background thread that uploads processed files to Weaviate."""
        logger.info("Starting Weaviate upload worker thread")
        
        upload_results = []
        
        while not self.upload_stop_event.is_set() or not self.files_for_upload.empty():
            try:
                # Get the next file to upload with a timeout
                try:
                    file_info = self.files_for_upload.get(timeout=5)
                except queue.Empty:
                    continue
                
                # Upload to Weaviate
                result = self.upload_to_weaviate(file_info['path'])
                result['file_key'] = file_info['file_key']
                upload_results.append(result)
                
                # Log progress
                logger.info(
                    f"Uploaded {result.get('successful_uploads', 0)}/{file_info['record_count']} "
                    f"articles from {file_info['file_key']} to Weaviate"
                )
                
                # Mark as done
                self.files_for_upload.task_done()
                
            except Exception as e:
                logger.error(f"Error in upload worker: {str(e)}", exc_info=True)
                
        # Save upload results to S3
        try:
            results_path = self.temp_dir / "weaviate_upload_results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'total_files': len(upload_results),
                    'details': upload_results
                }, f, indent=2)
                
            self.upload_file(results_path, "weaviate_upload_results.json")
            logger.info("Weaviate upload worker completed")
            
        except Exception as e:
            logger.error(f"Error saving upload results: {str(e)}")
    
    def process_files_in_chunks(self, file_keys: List[str], max_workers: int) -> List[Dict]:
        """
        Process files in chunks to enable parallel uploading.
        
        Args:
            file_keys: List of file keys to process
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of processing results
        """
        all_results = []
        
        # Start the upload worker if parallel upload is enabled
        if self.enable_parallel_upload and self.weaviate_url:
            self.upload_thread = threading.Thread(target=self._upload_worker)
            self.upload_thread.daemon = True
            self.upload_thread.start()
        
        # Process files in chunks
        for i in range(0, len(file_keys), self.chunk_size):
            chunk = file_keys[i:i + self.chunk_size]
            logger.info(f"Processing chunk {i//self.chunk_size + 1}/{(len(file_keys) + self.chunk_size - 1)//self.chunk_size} ({len(chunk)} files)")
            
            # Process this chunk of files
            chunk_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.process_file, file_key): file_key for file_key in chunk}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(chunk), desc="Processing Files"):
                    file_key = future_to_file[future]
                    try:
                        result = future.result()
                        chunk_results.append(result)
                    except Exception as e:
                        logger.error(f"Unhandled exception processing {file_key}: {str(e)}", exc_info=True)
                        chunk_results.append({
                            'file_key': file_key,
                            'status': 'failed',
                            'error': str(e)
                        })
            
            all_results.extend(chunk_results)
            
            # Update checkpoint after each chunk
            self._save_checkpoint()
            
            # Log chunk completion
            successful = sum(1 for r in chunk_results if r['status'] == 'success')
            logger.info(f"Chunk completed: {successful}/{len(chunk)} files processed successfully")
        
        # Wait for the upload queue to be empty if parallel upload is enabled
        if self.enable_parallel_upload and self.weaviate_url:
            logger.info("Waiting for Weaviate upload queue to complete...")
            self.files_for_upload.join()  # Wait for all uploads to complete
            self.upload_stop_event.set()  # Signal upload thread to stop
            self.upload_thread.join(timeout=300)  # Wait for the thread to finish
        
        return all_results
    
    def run_pipeline(self, max_files: Optional[int] = None, max_workers: int = 4) -> Dict:
        """
        Run the entire pipeline on all files or a subset.
        
        Args:
            max_files: Maximum number of files to process (None for all)
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        logger.info(f"Starting FDA Consultant AI pipeline")
        
        # List all files in the input bucket
        file_keys = self.list_input_files()
        
        if max_files:
            file_keys = file_keys[:max_files]
            logger.info(f"Processing {len(file_keys)} of {len(self.article_files)} files")
        
        # Process files in chunks if parallel upload is enabled, otherwise all at once
        if self.enable_parallel_upload:
            results = self.process_files_in_chunks(file_keys, max_workers)
        else:
            # Process files in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.process_file, file_key): file_key for file_key in file_keys}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_keys), desc="Processing Files"):
                    file_key = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Unhandled exception processing {file_key}: {str(e)}", exc_info=True)
                        results.append({
                            'file_key': file_key,
                            'status': 'failed',
                            'error': str(e)
                        })
        
        # Summarize results
        successful = sum(1 for r in results if r['status'] == 'success')
        total_records = sum(r.get('record_count', 0) for r in results)
        
        pipeline_result = {
            'total_files': len(file_keys),
            'successful_files': successful,
            'failed_files': len(file_keys) - successful,
            'total_records': total_records,
            'elapsed_time': time.time() - start_time
        }
        
        # Save detailed results
        results_path = self.temp_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': pipeline_result,
                'details': results
            }, f, indent=2)
        
        # Upload results to S3
        self.upload_file(results_path, "pipeline_results.json")
        
        logger.info(f"Pipeline completed: {successful}/{len(file_keys)} files processed successfully")
        logger.info(f"Total records processed: {total_records}")
        logger.info(f"Total time: {pipeline_result['elapsed_time']:.2f} seconds")
        
        return pipeline_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FDA Consultant AI Pipeline')
    
    # S3 and AWS configuration
    parser.add_argument('--input-bucket', required=True, help='S3 bucket with input CSV files')
    parser.add_argument('--output-bucket', required=True, help='S3 bucket for output files')
    parser.add_argument('--input-prefix', default='', help='S3 prefix (folder path) within input bucket')
    parser.add_argument('--output-prefix', default='', help='S3 prefix (folder path) within output bucket')
    parser.add_argument('--model-name', required=True, help='SageMaker model name for embeddings')
    parser.add_argument('--region', default='us-east-1', help='AWS region name')
    
    # SageMaker configuration
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge', 
                        help='SageMaker instance type for batch transform')
    parser.add_argument('--instance-count', type=int, default=1, 
                        help='Number of instances for batch transform')
    parser.add_argument('--max-concurrent', type=int, default=10, 
                        help='Maximum concurrent transform jobs')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Number of records per mini-batch')
    
    # Processing options
    parser.add_argument('--max-files', type=int, default=None, 
                        help='Maximum number of files to process (None for all)')
    parser.add_argument('--max-workers', type=int, default=4, 
                        help='Maximum number of concurrent workers')
    parser.add_argument('--temp-dir', default=None, 
                        help='Directory for temporary files')
    parser.add_argument('--log-level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    # Resilience options
    parser.add_argument('--checkpoint-enabled', action='store_true',
                        help='Enable checkpointing for resilience')
    parser.add_argument('--use-spot-instances', action='store_true',
                        help='Use spot instances for cost savings')
    parser.add_argument('--max-spot-retry', type=int, default=5,
                        help='Maximum number of retries for spot instance interruptions')
    
    # Parallel upload options
    parser.add_argument('--enable-parallel-upload', action='store_true',
                        help='Start uploading to Weaviate while processing continues')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Number of files to process before starting parallel upload')
    
    # Weaviate options
    parser.add_argument('--weaviate-url', default=None,
                        help='URL of the Weaviate instance')
    parser.add_argument('--weaviate-api-key', default=None,
                        help='API key for Weaviate instance')
    parser.add_argument('--weaviate-class-name', default='PubMedArticle',
                        help='Name of the Weaviate class to use')
    parser.add_argument('--weaviate-batch-size', type=int, default=200,
                        help='Batch size for Weaviate uploads')
    parser.add_argument('--weaviate-upload-workers', type=int, default=8,
                        help='Number of concurrent workers for Weaviate uploads')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set logging level
    logging.getLogger('fda-ai-pipeline').setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize and run the pipeline
        pipeline = FDAConsultantPipeline(
            input_bucket=args.input_bucket,
            output_bucket=args.output_bucket,
            model_name=args.model_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            max_concurrent_transforms=args.max_concurrent,
            batch_size=args.batch_size,
            region_name=args.region,
            temp_dir=args.temp_dir,
            checkpoint_enabled=args.checkpoint_enabled,
            use_spot_instances=args.use_spot_instances,
            max_spot_retry=args.max_spot_retry,
            enable_parallel_upload=args.enable_parallel_upload,
            chunk_size=args.chunk_size,
            weaviate_url=args.weaviate_url,
            weaviate_api_key=args.weaviate_api_key,
            weaviate_class_name=args.weaviate_class_name,
            weaviate_batch_size=args.weaviate_batch_size,
            weaviate_upload_workers=args.weaviate_upload_workers,
            input_prefix=args.input_prefix,
            output_prefix=args.output_prefix
        )
        
        result = pipeline.run_pipeline(
            max_files=args.max_files,
            max_workers=args.max_workers
        )
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"ERROR: {str(e)}")
        exit(1)