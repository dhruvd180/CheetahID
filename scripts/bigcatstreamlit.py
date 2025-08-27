import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from pathlib import Path


# Import inference pipeline (with fallback for demo mode)
try:
    from inference_pipeline import EnhancedCheetahIdentifier
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
# Functions
if "identifier" not in st.session_state:
    if INFERENCE_AVAILABLE:
        st.session_state.identifier = EnhancedCheetahIdentifier(
            model_path="models/cheetah_cropped_embedder.pt",
            reference_embeddings_path="models/reference_embeddings_cropped.pkl",
            device="cpu"  # or "cuda" if you ever run on a machine with NVIDIA GPU
        )
        st.session_state.model_loaded = True
    else:
        st.session_state.identifier = None
        st.session_state.model_loaded = False


def run_identification(image, threshold_level, return_top_k):
    """Run cheetah identification"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing image...")
        progress_bar.progress(25)
        
        if INFERENCE_AVAILABLE and st.session_state.identifier:
            # Save temporary image file
            temp_path = "temp_upload.jpg"
            image.save(temp_path)
            
            status_text.text("Running identification...")
            progress_bar.progress(75)
            
            # Run identification
            result = st.session_state.identifier.identify(
                temp_path,
                rejection_level=threshold_level,
                return_top_k=return_top_k
            )
            
            # Clean up
            os.remove(temp_path)
        else:
            # Demo mode
            status_text.text("Demo mode - generating results...")
            progress_bar.progress(50)
            time.sleep(1)  # Simulate processing
            result = create_demo_result(threshold_level)
        
        progress_bar.progress(100)
        status_text.text("Identification complete!")
        
        # Store result and add to history
        st.session_state.identification_result = result
        result['timestamp'] = datetime.now()
        result['threshold_used'] = threshold_level
        st.session_state.identification_history.append(result)
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during identification: {str(e)}")

def create_demo_result(threshold_level):
    """Create demo result for testing"""
    import random
    
    demo_individuals = ['individual_001', 'individual_042', 'individual_087', 'individual_124']
    
    if random.random() < 0.7:  # 70% success rate
        individual = random.choice(demo_individuals)
        confidence = random.uniform(0.6, 0.95)
        similarity = random.uniform(0.65, 0.92)
        
        # Generate top matches
        all_matches = [{
            'individual': individual,
            'similarity': similarity,
            'confidence': confidence
        }]
        
        # Add other matches
        other_individuals = [ind for ind in demo_individuals if ind != individual]
        for i, other_ind in enumerate(random.sample(other_individuals, min(2, len(other_individuals)))):
            all_matches.append({
                'individual': other_ind,
                'similarity': similarity - random.uniform(0.1, 0.3) * (i + 1),
                'confidence': confidence - random.uniform(0.1, 0.2) * (i + 1)
            })
        
        return {
            'status': 'identified',
            'individual': individual,
            'confidence': confidence,
            'similarity': similarity,
            'confidence_level': 'High' if confidence > 0.8 else 'Medium',
            'all_matches': all_matches
        }
    else:
        return {
            'status': 'unknown',
            'individual': None,
            'best_match_similarity': random.uniform(0.3, 0.5),
            'threshold_value': 0.6,
            'all_matches': []
        }

def display_results(result):
    """Display identification results with enhanced UI"""
    if result['status'] == 'identified':
        # Successful identification
        individual = result['individual']
        confidence = result['confidence']
        similarity = result.get('similarity', 0)
        
        st.markdown(f"""
        <div class="results-card success-result">
            <h3>Identification Successful</h3>
            <h4>Individual: {individual}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{similarity:.1%}</div>
                <div class="metric-label">Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            color = "#00C851" if confidence > 0.8 else "#FF8C42" if confidence > 0.6 else "#FF4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{confidence_level}</div>
                <div class="metric-label">Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display reference image if available
        display_reference_image(individual)
        
        # Top matches visualization
        if 'all_matches' in result and len(result['all_matches']) > 1:
            st.markdown("#### Top Matches")
            display_matches_chart(result['all_matches'])
    
    elif result['status'] == 'unknown':
        # Unknown individual
        best_sim = result.get('best_match_similarity', 0)
        
        st.markdown(f"""
        <div class="results-card warning-result">
            <h3>Unknown Individual</h3>
            <p>This cheetah is not in the reference database.</p>
            <p><strong>Best match similarity:</strong> {best_sim:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Error
        st.markdown(f"""
        <div class="results-card error-result">
            <h3>Identification Failed</h3>
            <p>An error occurred during identification.</p>
        </div>
        """, unsafe_allow_html=True)

def display_reference_image(individual_name):
    """Display reference image for identified individual"""
    if not st.session_state.reference_images_dir:
        st.warning("Reference images directory not configured")
        return
    
    # Look for reference image
    reference_path = None
    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Try different folder naming patterns
    possible_folder_names = [
        individual_name,
        f"individual_{individual_name.split('_')[-1].zfill(3)}",
        f"individual_{individual_name.split('_')[-1]}",
    ]
    
    for folder_name in possible_folder_names:
        folder_path = os.path.join(st.session_state.reference_images_dir, folder_name)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in possible_extensions):
                        reference_path = os.path.join(folder_path, file)
                        break
                if reference_path:
                    break
            except:
                continue
    
    if reference_path:
        st.markdown("#### Reference Image")
        ref_image = Image.open(reference_path)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(ref_image, caption=f"Reference: {individual_name}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"No reference image found for {individual_name}")

def display_matches_chart(matches):
    """Display top matches as an interactive chart"""
    df = pd.DataFrame(matches[:5])  # Top 5 matches
    
    fig = px.bar(
        df, 
        x='individual', 
        y='similarity',
        color='confidence',
        color_continuous_scale='Oranges',
        title="Top Similarity Matches",
        labels={'similarity': 'Similarity Score', 'individual': 'Individual ID'}
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#1A1A1A'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_analytics():
    """Display analytics dashboard"""
    if not st.session_state.identification_history:
        st.info("No identification history available yet. Run some identifications to see analytics!")
        return
    
    history = st.session_state.identification_history
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_ids = len(history)
    successful_ids = len([r for r in history if r.get('status') == 'identified'])
    avg_confidence = np.mean([r.get('confidence', 0) for r in history if r.get('confidence')])
    unique_individuals = len(set([r.get('individual') for r in history if r.get('individual')]))
    
    with col1:
        st.metric("Total Identifications", total_ids)
    with col2:
        st.metric("Success Rate", f"{(successful_ids/total_ids)*100:.1f}%")
    with col3:
        st.metric("Avg. Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
    with col4:
        st.metric("Unique Individuals", unique_individuals)
    
    # Charts
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Success/failure pie chart
        status_counts = pd.Series([r.get('status', 'unknown') for r in history]).value_counts()
        fig_pie = px.pie(
            values=status_counts.values, 
            names=status_counts.index,
            title="Identification Results",
            color_discrete_map={
                'identified': '#00C851',
                'unknown': '#FF8C42',
                'error': '#FF4444'
            }
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1A1A1A')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_b:
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in history if r.get('confidence')]
        if confidences:
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            fig_hist.update_traces(marker_color='#FF8C42')
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1A1A1A')
            st.plotly_chart(fig_hist, use_container_width=True)

def display_history():
    """Display identification history"""
    if not st.session_state.identification_history:
        st.info("No identification history available yet.")
        return
    
    st.markdown("#### Recent Identifications")
    
    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "Identified", "Unknown", "Error"],
            key="history_filter"
        )
    with col2:
        if st.button("Clear History"):
            st.session_state.identification_history = []
            st.rerun()
    
    # Filter history
    history = st.session_state.identification_history
    if filter_status != "All":
        history = [r for r in history if r.get('status', '').title() == filter_status]
    
    # Display history
    for i, result in enumerate(reversed(history[-20:])):  # Show last 20
        timestamp = result.get('timestamp', datetime.now())
        status = result.get('status', 'unknown')
        individual = result.get('individual', 'Unknown')
        confidence = result.get('confidence', 0)
        
        status_icon = "" if status == "identified" else "" if status == "unknown" else ""
        
        with st.expander(f"{status_icon} {timestamp.strftime('%H:%M:%S')} - {individual}"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write(f"**Status:** {status.title()}")
                st.write(f"**Individual:** {individual}")
            with col_b:
                if confidence:
                    st.write(f"**Confidence:** {confidence:.1%}")
                st.write(f"**Threshold:** {result.get('threshold_used', 'N/A')}")
            with col_c:
                st.write(f"**Time:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

def display_image_tools():
    """Display image enhancement and analysis tools"""
    st.markdown("#### Image Enhancement Tools")
    
    if st.session_state.current_image is None:
        st.info("Upload an image first to use these tools.")
        return
    
    image = st.session_state.current_image.copy()
    
    # Enhancement controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Adjustments")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("Color Saturation", 0.0, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
    
    with col2:
        st.markdown("##### Filters")
        apply_grayscale = st.checkbox("Convert to Grayscale")
        apply_blur = st.checkbox("Apply Blur")
        blur_radius = st.slider("Blur Radius", 0.5, 5.0, 1.0, 0.5) if apply_blur else 1.0
        
        # Crop tool
        st.markdown("##### Crop")
        crop_enabled = st.checkbox("Enable Cropping")
        if crop_enabled:
            left = st.slider("Left", 0, image.size[0], 0)
            top = st.slider("Top", 0, image.size[1], 0)
            right = st.slider("Right", left, image.size[0], image.size[0])
            bottom = st.slider("Bottom", top, image.size[1], image.size[1])
    
    # Apply enhancements
    enhanced_image = image.copy()
    
    # Basic adjustments
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(sharpness)
    
    # Filters
    if apply_grayscale:
        enhanced_image = enhanced_image.convert('L').convert('RGB')
    
    if apply_blur:
        from PIL import ImageFilter
        enhanced_image = enhanced_image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Cropping
    if crop_enabled and (left != 0 or top != 0 or right != image.size[0] or bottom != image.size[1]):
        enhanced_image = enhanced_image.crop((left, top, right, bottom))
    
    # Display enhanced image
    col_display1, col_display2 = st.columns(2)
    
    with col_display1:
        st.markdown("##### Original")
        st.image(image, use_container_width=True)
    
    with col_display2:
        st.markdown("##### Enhanced")
        st.image(enhanced_image, use_container_width=True)
    
    # Download enhanced image
    if st.button("Use Enhanced Image for Identification"):
        st.session_state.current_image = enhanced_image
        st.success("Enhanced image is now ready for identification!")
        st.rerun()
    
    # Image analysis
    st.markdown("#### Image Analysis")
    
    # Image statistics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Original Size", f"{image.size[0]}×{image.size[1]}")
        st.metric("Enhanced Size", f"{enhanced_image.size[0]}×{enhanced_image.size[1]}")
    
    with col_stat2:
        # Calculate file size approximation
        import io
        original_bytes = io.BytesIO()
        image.save(original_bytes, format='JPEG')
        original_size = len(original_bytes.getvalue()) / 1024  # KB
        
        enhanced_bytes = io.BytesIO()
        enhanced_image.save(enhanced_bytes, format='JPEG')
        enhanced_size = len(enhanced_bytes.getvalue()) / 1024  # KB
        
        st.metric("Original Size", f"{original_size:.1f} KB")
        st.metric("Enhanced Size", f"{enhanced_size:.1f} KB")
    
    with col_stat3:
        # Color analysis
        original_colors = len(image.getcolors(maxcolors=256*256*256)) if image.getcolors(maxcolors=256*256*256) else "256M+"
        enhanced_colors = len(enhanced_image.getcolors(maxcolors=256*256*256)) if enhanced_image.getcolors(maxcolors=256*256*256) else "256M+"
        
        st.metric("Original Colors", str(original_colors))
        st.metric("Enhanced Colors", str(enhanced_colors))
    
    # Histogram
    if st.checkbox("Show Color Histogram"):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Original', 'Enhanced'))
        
        # Original histogram
        for i, color in enumerate(['red', 'green', 'blue']):
            hist = np.array(image.histogram())[i*256:(i+1)*256]
            fig.add_trace(
                go.Scatter(x=list(range(256)), y=hist, name=f'Original {color}', line=dict(color=color)),
                row=1, col=1
            )
        
        # Enhanced histogram
        for i, color in enumerate(['red', 'green', 'blue']):
            hist = np.array(enhanced_image.histogram())[i*256:(i+1)*256]
            fig.add_trace(
                go.Scatter(x=list(range(256)), y=hist, name=f'Enhanced {color}', line=dict(color=color)),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Color Histogram Comparison",
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1A1A1A'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_about():
    """Display about information and system details"""
    st.markdown("#### About BigCatID")
    
    st.markdown("""
    <div class="custom-card">
        <h4>BigCatID System</h4>
        <p>
            BigCatID is an advanced computer vision system designed to identify individual cheetahs 
            based on their unique spot patterns. Using deep learning techniques, the system can 
            distinguish between different cheetah individuals with high accuracy.
        
    </div>
    """, unsafe_allow_html=True)
    
    # System Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### System Information")
        
        # Check system capabilities
        try:
            import torch
            torch_available = True
            cuda_available = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        except ImportError:
            torch_available = False
            cuda_available = False
            device_name = "PyTorch not available"
        
        system_info = {
            "Inference Available": "Yes" if INFERENCE_AVAILABLE else "Demo Mode",
            "PyTorch Available": "Yes" if torch_available else "No",
            "CUDA Available": "Yes" if cuda_available else "No",
            "Processing Device": device_name,
            "Model Status": "Loaded" if st.session_state.model_loaded else "Not Loaded"
        }
        
        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown("####  Performance Metrics")
        
        if st.session_state.identification_history:
            history = st.session_state.identification_history
            
            # Calculate metrics
            processing_times = []
            confidences = []
            
            for result in history:
                if 'processing_time' in result:
                    processing_times.append(result['processing_time'])
                if result.get('confidence'):
                    confidences.append(result['confidence'])
            
            avg_confidence = np.mean(confidences) if confidences else 0
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            
            metrics = {
                "Total Identifications": len(history),
                "Success Rate": f"{(len([r for r in history if r.get('status') == 'identified'])/len(history))*100:.1f}%",
                "Average Confidence": f"{avg_confidence:.1%}" if avg_confidence else "N/A",
                "Avg Processing Time": f"{avg_processing_time:.2f}s" if avg_processing_time else "N/A"
            }
            
            for key, value in metrics.items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("No performance data available yet.")
    
    # Model Details
    with st.expander(" Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - Base Model: ResNet-50 or similar CNN backbone
        - Feature Dimension: 512 or 1024 dimensional embeddings
        - Training Data: Curated cheetah image dataset
        - Preprocessing: Image normalization and augmentation
        
        **Matching Algorithm:**
        - Similarity Metric: Cosine similarity or Euclidean distance
        - Threshold Levels:
          - Conservative: Higher precision, lower recall
          - Moderate: Balanced precision and recall
          - Liberal: Lower precision, higher recall
        
        **Confidence Calculation:**
        - Based on similarity scores and statistical analysis
        - Calibrated using validation datasets
        - Includes rejection mechanism for unknown individuals
        """)
    
    # Export/Import functionality
    st.markdown("#### Data Management")
    
    col_export, col_import = st.columns(2)
    
    with col_export:
        if st.button("Export Session Data"):
            export_session_data()
    
    with col_import:
        uploaded_session = st.file_uploader(
            "Import Session Data",
            type=['json'],
            help="Import previously exported session data"
        )
        
        if uploaded_session is not None:
            try:
                session_data = json.load(uploaded_session)
                st.session_state.identification_history = session_data.get('history', [])
                # Convert timestamp strings back to datetime objects
                for result in st.session_state.identification_history:
                    if 'timestamp' in result and isinstance(result['timestamp'], str):
                        result['timestamp'] = datetime.fromisoformat(result['timestamp'])
                st.success("Session data imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing session data: {e}")

def export_session_data():
    """Export session data as JSON"""
    if not st.session_state.identification_history:
        st.warning("No session data to export.")
        return
    
    # Prepare data for export
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_identifications': len(st.session_state.identification_history),
        'history': []
    }
    
    # Convert history to serializable format
    for result in st.session_state.identification_history:
        export_result = result.copy()
        if 'timestamp' in export_result:
            export_result['timestamp'] = export_result['timestamp'].isoformat()
        export_data['history'].append(export_result)
    
    # Create download
    json_str = json.dumps(export_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    
    filename = f"bigcatid_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    st.markdown(f"""
    <a href="data:application/json;base64,{b64}" download="{filename}">
        <button style="
            background: linear-gradient(135deg, var(--safari-orange) 0%, var(--safari-orange-dark) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
        ">
            Download Session Data
        </button>
    </a>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="BigCatID - Cheetah Identification System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Safari theme
def load_css():
    st.markdown("""
    <style>
    /* Safari Color Palette Variables */
    :root {
        --safari-orange: #FF8C42;
        --safari-orange-light: #FFB366;
        --safari-orange-dark: #E5661A;
        --serengeti-cream: #ffd6a8;
        --savanna-white: #FFFFFF;
        --midnight-black: #1A1A1A;
        --charcoal: #2C2C2C;
        --dust-beige: #F5F1E8;
        --acacia-gold: #D4A574;
        --glass-overlay: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 140, 66, 0.2);
        --background: var(--serengeti-cream);
        --surface: var(--dust-beige);
        --text-primary: var(--midnight-black);
        --text-secondary: var(--charcoal);
        --accent: var(--safari-orange);
        --accent-light: var(--safari-orange-light);
        --accent-dark: var(--safari-orange-dark);
        --border: rgba(255, 140, 66, 0.15);
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--serengeti-cream) 0%, var(--dust-beige) 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--safari-orange) 0%, var(--safari-orange-dark) 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(229, 102, 26, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--savanna-white) 0%, var(--dust-beige) 100%);
        border-right: 2px solid var(--border);
    }
    
    /* Card styling */
    .custom-card {
        background: var(--savanna-white);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 140, 66, 0.1);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
    }
    
    .glass-card {
        background: var(--glass-overlay);
        border: 1px solid var(--glass-border);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    
    /* Results card */
    .results-card {
        background: linear-gradient(135deg, var(--savanna-white) 0%, var(--dust-beige) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 140, 66, 0.15);
        border: 2px solid var(--border);
    }
    
    .success-result {
        border-left: 5px solid #00C851;
        background: linear-gradient(135deg, #E8F5E8 0%, var(--savanna-white) 100%);
    }
    
    .warning-result {
        border-left: 5px solid var(--safari-orange);
        background: linear-gradient(135deg, #FFF4E6 0%, var(--savanna-white) 100%);
    }
    
    .error-result {
        border-left: 5px solid #FF4444;
        background: linear-gradient(135deg, #FFEBEE 0%, var(--savanna-white) 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--safari-orange) 0%, var(--safari-orange-dark) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 140, 66, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--safari-orange-dark) 0%, var(--safari-orange) 100%);
        box-shadow: 0 6px 20px rgba(255, 140, 66, 0.4);
        transform: translateY(-2px);
    }
    
    .secondary-button {
        background: linear-gradient(135deg, var(--acacia-gold) 0%, var(--safari-orange-light) 100%);
    }
    
    /* Metrics styling */
    .metric-card {
        background: var(--savanna-white);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(255, 140, 66, 0.1);
        border: 1px solid var(--border);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--safari-orange);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(255, 140, 66, 0.15);
        border: 2px solid var(--border);
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-ready {
        color: #00C851;
        font-weight: bold;
    }
    
    .status-processing {
        color: var(--safari-orange);
        font-weight: bold;
    }
    
    .status-error {
        color: #FF4444;
        font-weight: bold;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: var(--safari-orange);
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed var(--safari-orange);
        background: var(--glass-overlay);
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background: var(--savanna-white);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--safari-orange-light);
        border-radius: 10px;
        color: var(--text-primary);
        border: 1px solid var(--border);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--safari-orange);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--dust-beige);
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    
    /* Custom animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .custom-card, .results-card {
            padding: 1rem;
        }
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dust-beige);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--safari-orange);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--safari-orange-dark);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'identifier' not in st.session_state:
        st.session_state.identifier = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'identification_result' not in st.session_state:
        st.session_state.identification_result = None
    if 'reference_images_dir' not in st.session_state:
        st.session_state.reference_images_dir = "data/raw/"
    if 'identification_history' not in st.session_state:
        st.session_state.identification_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

# Load custom CSS
load_css()

# Initialize session state
initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>BigCatID</h1>
    <p>Advanced Cheetah Identification System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("###  Control Panel")
    
    # Model Status
    st.markdown("####  System Status")
    if INFERENCE_AVAILABLE and st.session_state.model_loaded:
        st.markdown('<p class="status-ready">Model Ready</p>', unsafe_allow_html=True)
    elif INFERENCE_AVAILABLE:
        st.markdown('<p class="status-error">Model Not Loaded</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">Demo Mode</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Configuration Section
    with st.expander("Model Configuration"):
        model_path = st.text_input(
            "Model Path", 
            value="models/cheetah_cropped_embedder.pt",
            help="Path to the trained model file"
        )
        
        embeddings_path = st.text_input(
            "Embeddings Path", 
            value="models/reference_embeddings_cropped.pkl",
            help="Path to reference embeddings file"
        )
        
        st.session_state.reference_images_dir = st.text_input(
            "Reference Images Directory", 
            value=st.session_state.reference_images_dir,
            help="Directory containing reference images"
        )
        
        if st.button("Load Model"):
            if INFERENCE_AVAILABLE:
                with st.spinner("Loading model..."):
                    try:
                        import torch
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        st.session_state.identifier = EnhancedCheetahIdentifier(
                            model_path, embeddings_path, device
                        )
                        st.session_state.model_loaded = True
                        st.success(f"Model loaded successfully on {device}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
            else:
                st.warning("Inference pipeline not available - demo mode")
    
    st.divider()
    
    # Identification Settings
    st.markdown("#### Identification Settings")
    threshold_level = st.selectbox(
        "Confidence Threshold",
        ["conservative", "moderate", "liberal"],
        index=1,
        help="Conservative: Higher confidence required, Liberal: Lower confidence accepted"
    )
    
    return_top_k = st.slider(
        "Return Top Matches",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of top matches to return"
    )
    
    st.divider()
    
    # Statistics
    if st.session_state.identification_history:
        st.markdown("#### Session Statistics")
        total_ids = len(st.session_state.identification_history)
        successful_ids = len([r for r in st.session_state.identification_history if r.get('status') == 'identified'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_ids}</div>
                <div class="metric-label">Total IDs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{successful_ids}</div>
                <div class="metric-label">Successful</div>
            </div>
            """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 1])

# Left column - Image upload and processing
with col1:
    st.markdown("### Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a cheetah image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear image of a cheetah for identification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image information
        st.markdown("####  Image Information")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Width", f"{image.size[0]}px")
        with col_b:
            st.metric("Height", f"{image.size[1]}px")
        with col_c:
            st.metric("Format", image.format)
        
        # Identification button
        if st.button(" Identify Cheetah", type="primary", use_container_width=True):
            run_identification(image, threshold_level, return_top_k)

# Right column - Results
with col2:
    st.markdown("### Identification Results")
    
    if st.session_state.identification_result is not None:
        display_results(st.session_state.identification_result)
    else:
        st.markdown("""
        <div class="custom-card" style="text-align: center; padding: 3rem;">
            <h4> Ready for Identification</h4>
            <p>Upload an image and click "Identify Cheetah" to see results</p>
        </div>
        """, unsafe_allow_html=True)

# Additional tabs for enhanced features
st.markdown("""
    <style>
    /* Target tab labels */
    div[data-baseweb="tab"] button {
        margin-right: 20px;  /* Adjust spacing between tabs */
        padding: 8px 20px;   /* Optional: increase padding inside tabs */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Targets tab content panels */
        div[data-testid="stTabsTab"] > div:first-child {
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Add CSS for padding/margin between tab headers
st.markdown("""
    <style>
        /* Select all tab buttons */
        div[data-baseweb="tab"] button {
            padding-left: 20px;   /* horizontal padding inside the tab */
            padding-right: 20px;
            margin-right: 10px;   /* space between tabs */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Tab buttons */
        div[data-baseweb="tab"] button {
            padding: 10px 30px !important;    /* vertical | horizontal padding */
            margin-right: 10px !important;    /* space between tabs */
            border-radius: 20px !important;   /* pill shape */
            font-size: 16px !important;       /* text size */
        }
        /* Optional: add a little more spacing above tabs */
        .stTabs {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "ID History", "Image Editing Tools", "About"])

with tab1:
    display_analytics()

with tab2:
    display_history()

with tab3:
    display_image_tools()

with tab4:
    display_about()
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>BigCatID - Advanced Cheetah Identification System</p>
    <p>For Wildlife Conservation</p>
</div>
""", unsafe_allow_html=True)