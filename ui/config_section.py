import streamlit as st
import json
import yaml

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