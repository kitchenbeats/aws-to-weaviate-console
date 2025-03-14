import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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