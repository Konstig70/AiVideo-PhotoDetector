import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from pytube import YouTube
import yt_dlp
from io import BytesIO

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None

if "file_path" not in st.session_state:
    st.session_state.file_path = None

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
    <p>Get reliable video authenticity scores by combining anatomy, metadata, and motion analysis, with a clear and easy to understand verdict. Optional technical breakdown is also available.</p>
</div>
""", unsafe_allow_html=True)

# Main content area
import streamlit as st
import yt_dlp
import os

col3, col1, col2 = st.columns([3, 2, 1])

with col3:
    st.markdown("### Paste YouTube link")
    youtube_link = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=example")
    youtubevideo = None
    if youtube_link:
        # Ensure URL starts with https://
        url = youtube_link if youtube_link.startswith("https://") else "https://" + youtube_link

        if st.button("Analyze video from YouTube"):
            st.markdown(f"""
            <div class="status-success">
                <strong>✅ YouTube Analysis Initiated</strong><br>
                Analyzing video from: {youtube_link}
            </div>
            """, unsafe_allow_html=True)

            os.makedirs("downloads", exist_ok=True)

            ydl_opts = {
                "outtmpl": "downloads/youtube_video.%(ext)s",
                "format": "mp4",
                "quiet": True,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
                    st.success(f"✅ Video downloaded successfully: {video_path}")

                    # Read video as bytes so it can be fed to your file uploader code
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                        youtubevideo = BytesIO(video_bytes)
                        youtubevideo.name = os.path.basename(video_path)  # mimic UploadedFile.name

                    # Now you can pass ladattuvideo to your existing processing functions
                    # e.g., analyze_video(ladattuvideo)

            except Exception as e:
                st.error(f"❌ Error downloading video: {e}")

with col1:
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
        <h4>Quick guide</h4>
        <ol>
            <li>Upload your video file</li>
            <li>Wait for automatic analysis</li>
            <li>Review detailed results</li>
            <li>Download report (optional)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Video processing with professional error handling
if ladattuvideo or youtubevideo is not None:
    try:
        if youtubevideo is not None:
            ladattuvideo = youtubevideo
        # File information display
        else:
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
        if youtubevideo is None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", file_details['filename'])
            with col2:
                st.metric("File Type", file_details['filetype'])
            with col3:
                st.metric("File Size", f"{file_details['filesize']/1024/1024:.2f} MB")
            
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ladattuvideo.name).suffix) as tmp_file:
            tmp_file.write(ladattuvideo.getvalue())
            temp_path = tmp_file.name
            # Progress indicator
            progress_bar = st.progress(0)            
            #<-------------------------DO ALL ANALYSIS HERE AND MAKE SURE TO RETURN ALL ANALYSIS RESULT DATA------------------------->
            from MetaDataScrutiny.metadataanalyzer import metadata
            from GeometryMapping.GeometryMapping import GeometryMapper
            from Justification.justification import VideoJustificationAgent
            from Justification.justification import PDFGenerator
            #Inform user about whats happening
            status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Performing metadata analysis...
        </div>
        """, unsafe_allow_html=True)
            time.sleep(1)
            analyzer = metadata(temp_path)
            result = analyzer.analyze()
            progress_bar.progress(20)
            #Inform user about whats happening
            status_container.markdown("""
        <div class="status-success">
            <strong>✅ File Upload Successful</strong><br>
            Performing geometrical and human anatomy anomaly detection, this may take some time...
        </div>
        """, unsafe_allow_html=True)
            print(ladattuvideo.name)
            print(st.session_state.file_path)
            if st.session_state.file_path == ladattuvideo.name:
                    if st.session_state.results:     
                        geometry_results = st.session_state.results
                        st.session_state.file_path = ladattuvideo.name
                    else:
                        geometry_results = GeometryMapper.analyze_video(temp_path, False, progress_bar, 25)
                        st.session_state.results = geometry_results
                        st.session_state.file_path = ladattuvideo.name
            else:
                    geometry_results = GeometryMapper.analyze_video(temp_path, False, progress_bar, 25)
                    st.session_state.results = geometry_results
                    st.session_state.file_path = ladattuvideo.name
                    
            # Justification based on data
            agent = VideoJustificationAgent("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")
            response = agent.analyze({**result, **geometry_results})

                

            
            
            st.markdown("### Video Preview")
            st.video(ladattuvideo)
            
            
            try:
                import cv2
                cap = cv2.VideoCapture(temp_path)
                progress_bar.progress(90)
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
                    status_container.markdown("""
        <div class="status-success">
            <strong>✅ Analysis Complete Video file has been successfully processed and analyzed.</strong><br>
        </div>
        """, unsafe_allow_html=True)
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
            st.markdown("## Summary for video:")
            st.markdown(f"##### {response}")
            data = result["metadata"]
            score = result["metadata_anomaly_score"]
            # Technical details section
            with st.expander("### 📋 Technical Analysis (click to expand)"):
                st.markdown(" #### Here is all of the data that our algorithms were able to detect. The anomaly score is our own scoring system that is calculated by combining different data values and different detected anomalies with some anomalies weigthed more than others (which can be seen below):")
                st.markdown(f"Video metadata:")
                for k,v in data.items():
                    val = k.replace("_"," ")
                    st.markdown(f"  - {val}: {v}")
                st.markdown(f"Suspicion score: {score}")
                st.markdown(f"Human anatomy anomaly detection results: ")
                for key, value in geometry_results.items():
                    val = key.replace("_"," ")
                    st.markdown(f"  - {val}: {value}")
            #Possible news crosscheck
            with st.expander("### Perform news/search crosscheck"):
                st.markdown("#### A news crosscheck can be performed to see if any news articles were published about the contents of the video. This can help to verify the authenticity of the video.")
                st.markdown("#####  This is an experimental feature that is still in alpha phase. It uses LLM to generate search queries based on the contents of the video and then searches news articles based on those queries. Finally it summarizes the results and tells if any of the articles were relevant to the contents of the video. Please note that this feature is not always accurate and should be used with caution.")
                if st.button("Try our experimental video context detection system by clicking here"):
                    print("CrossCheck!")
                st.markdown("##### You can also perform a news crosscheck by simply describing the contents of video below, this might work better than the automatic context detection, how ever keep in mind that you need to present sufficiently relatable information for the video.")
                input = st.text_input("Please describe the videos contents:")
                if st.button("submit"):
                    results = agent.perform_news_cross_check(input)
                    st.markdown("### Here is the results of the news crosscheck")
                    st.markdown(f"#### {results}")    
            #Authenticity report
            
            with st.expander("### Download authenticity report"):
                st.markdown("#### Authenticity report is created by using the data created from analysis")
                if st.button("Download PDF report from your video"):
                    pdf_creator = PDFGenerator(data, geometry_results, score)
                    pdf_file = pdf_creator.generate_pdf()
                    st.download_button(
                        label="Download report PDF",
                        data=pdf_file,
                        file_name="AuthenticityReport.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:
        print(e)
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
        <p>Upload a video file above to begin the security analysis process. Our platform supports multiple formats and provides comprehensive anomaly detection.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📊 Detailed Analysis**")
with col2:
    st.markdown("**⚡ Fast Processing**")


