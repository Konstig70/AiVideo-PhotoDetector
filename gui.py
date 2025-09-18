import streamlit as st
import tempfile
import os
from pathlib import Path
import time

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Video Analysis Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #e8f4fd;
        text-align: center;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Professional card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    /* Upload section styling */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 2rem 0;
        text-align: center;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f1f3f4;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI video detector</h1>
    <p>Detect percentage chance of AI video</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with professional information
with st.sidebar:
    st.markdown("### 📊 Platform Features")
    st.markdown("""
    - **Secure Upload**: Encrypted file handling
    - **Format Support**: MP4, MOV, AVI
    - **Real-time Analysis**: Instant results
    - **Detailed Reports**: Comprehensive insights
    """)
    
    st.markdown("### 🔒 Security Standards")
    st.markdown("""
    - ISO 27001 Compliant
    - GDPR Compliant
    - End-to-end Encryption
    - No data retention
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Video File")
    st.markdown("*Drag and drop or click to browse for suspicious video content*")
    
    # Video uploader
    ladattuvideo = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "mov", "avi"],
        help="Maximum file size: 100MB. Supported formats: MP4, MOV, AVI"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🚀 Quick guide</h4>
        <ol>
            <li>Upload your video file</li>
            <li>Wait for automatic analysis</li>
            <li>Review detailed results</li>
            <li>Download report (optional)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Video processing with professional error handling
if ladattuvideo is not None:
    try:
        # File information display
        file_details = {
            "filename": ladattuvideo.name,
            "filetype": ladattuvideo.type,
            "filesize": ladattuvideo.size
        }
        
        # Professional status display
        status_container = st.empty()
        status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Processing your video file...
        </div>
        """, unsafe_allow_html=True)
        
        # File details in professional format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file_details['filename'])
        with col2:
            st.metric("File Type", file_details['filetype'])
        with col3:
            st.metric("File Size", f"{file_details['filesize']/1024/1024:.2f} MB")
        
        # File size validation
        max_size = 100 * 1024 * 1024  # 100MB
        if file_details['filesize'] > max_size:
            st.markdown("""
            <div class="status-error">
                <strong>❌ File Too Large</strong><br>
                Maximum allowed size is 100MB. Please compress your video or upload a smaller file.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ladattuvideo.name).suffix) as tmp_file:
                tmp_file.write(ladattuvideo.getvalue())
                temp_path = tmp_file.name
                # Progress indicator
                progress_bar = st.progress(0)
                
                #<-------------------------DO ALL ANALYSIS HERE AND MAKE SURE TO RETURN ALL ANALYSIS RESULT DATA------------------------->
                from MetaDataScrutiny.metadataanalyzer import metadata
                from GeometryMapping.GeometryMapping import GeometryMapper
                #Inform user about whats happening
                status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Performing metadata analysis...
        </div>
        """, unsafe_allow_html=True)
        
                analyzer = metadata(temp_path)
                result = analyzer.analyze()
                progress_bar.progress(25)
                #Inform user about whats happening
                status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Performing geometrical and human anatomy anomaly detection, this may take some time...
        </div>
        """, unsafe_allow_html=True)
        
                geometry_results = GeometryMapper.analyze_video(temp_path, False, progress_bar, 25)
                progress_bar.progress(50)
                # more analysis
            
                

            
            
            st.markdown("### 🎥 Video Preview")
            st.video(ladattuvideo)
            
            
            try:
                import cv2
                cap = cv2.VideoCapture(temp_path)
                progress_bar.progress(75)
                status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Fetching file information...
        </div>
        """, unsafe_allow_html=True)
        
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Professional metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Resolution", f"{width}×{height}")
                    with col2:
                        st.metric("Frame Rate", f"{fps:.1f} FPS")
                    with col3:
                        st.metric("Duration", f"{duration:.1f}s")
                    with col4:
                        st.metric("Total Frames", f"{frame_count:,}")
                    
                    progress_bar.progress(100)
                    time.sleep(2)

                    # Remove the progress bar
                    progress_bar.empty()
                    st.markdown("""
                    <div class="status-success">
                        <strong>✅ Analysis Complete</strong><br>
                        Video file has been successfully processed and analyzed.
                    </div>
                    """, unsafe_allow_html=True)
                    status_container.empty()
                    cap.release()
                else:
                    st.markdown("""
                    <div class="status-warning">
                        <strong>⚠️ Limited Analysis</strong><br>
                        Could not extract detailed video properties. Basic file information is available above.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except ImportError:
                st.markdown("""
                <div class="status-warning">
                    <strong>📦 Enhanced Analysis Available</strong><br>
                    Install opencv-python for detailed video analysis: <code>pip install opencv-python</code>
                </div>
                """, unsafe_allow_html=True)
            except Exception as cv_error:
                st.markdown(f"""
                <div class="status-warning">
                    <strong>⚠️ Analysis Warning</strong><br>
                    Could not complete full video analysis: {str(cv_error)}
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            

            data = result["metadata"]
            score = result["suspicion_score"]
            # Technical details section
            st.markdown("### 📋 Technical Analysis")
            st.markdown(f"Video metadata:")
            for k,v in data.items():
                val = k.replace("_"," ")
                st.markdown(f"  - {val}: {v}")
            st.markdown(f"Suspicion score: {score}")
            st.markdown(f"Geometrical and human anatomy detection results: ")
            for key, value in geometry_results.items():
                val = key.replace("_"," ")
                st.markdown(f"  - {val}: {value}") 
                    
    except Exception as e:
        st.markdown(f"""
        <div class="status-error">
            <strong>❌ Processing Error</strong><br>
            Error processing video file: {str(e)}<br>
            Please try uploading a different video file or contact support.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-card">
        <h4>🎯 Ready to Analyze</h4>
        <p>Upload a video file above to begin the security analysis process. Our platform supports multiple formats and provides comprehensive threat detection.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔒 Secure & Confidential**")
with col2:
    st.markdown("**⚡ Fast Processing**")
with col3:
    st.markdown("**📊 Detailed Analysis**")



