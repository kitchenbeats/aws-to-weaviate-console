import streamlit as st
from datetime import datetime

def render_execution_section(pipeline_runner, config_manager):
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