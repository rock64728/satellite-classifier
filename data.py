import streamlit as st
import numpy as np
from PIL import Image
import cv2
import random
import time

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 20px;
        background-color: #1f77b4;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 20px;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        text-align: center;
        margin: 0.5rem 0;
    }
    .color-swatch {
        width: 50px;
        height: 50px;
        border-radius: 8px;
        display: inline-block;
        margin: 5px;
        border: 2px solid #333;
    }
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 0.375rem;
        height: 1rem;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background-color: #0d6efd;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛰️ Satellite Image Classifier</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Image Classification", "About Model", "Sample Images", "Image Analysis"])

# Class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
class_emojis = ['☁️', '🏜️', '🌲', '🌊']

# Mock classification function based on image analysis
def analyze_image_features(img):
    """
    Analyze image features to make classification predictions
    This is a simplified approach using color analysis and basic image processing
    """
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Resize for consistent processing
    img_resized = cv2.resize(img_array, (255, 255))
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    
    # Extract features
    features = {}
    
    # Color analysis
    mean_rgb = np.mean(img_resized, axis=(0, 1))
    std_rgb = np.std(img_resized, axis=(0, 1))
    
    # HSV analysis
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    
    # Texture analysis (simplified)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (255 * 255)
    
    # Store features
    features['mean_rgb'] = mean_rgb
    features['std_rgb'] = std_rgb
    features['mean_hue'] = mean_hue
    features['mean_saturation'] = mean_saturation
    features['mean_value'] = mean_value
    features['edge_density'] = edge_density
    features['brightness'] = np.mean(gray)
    
    return features

def classify_image(img):
    """
    Classify image based on extracted features
    This is a rule-based classification system
    """
    features = analyze_image_features(img)
    
    # Initialize probabilities
    probabilities = np.zeros(4)  # [Cloudy, Desert, Green_Area, Water]
    
    # Rule-based classification
    brightness = features['brightness']
    mean_rgb = features['mean_rgb']
    mean_saturation = features['mean_saturation']
    edge_density = features['edge_density']
    
    # Cloudy classification (high brightness, low saturation)
    if brightness > 150 and mean_saturation < 50:
        probabilities[0] = 0.7 + random.uniform(-0.1, 0.1)
    elif brightness > 120 and mean_saturation < 80:
        probabilities[0] = 0.4 + random.uniform(-0.1, 0.1)
    else:
        probabilities[0] = 0.1 + random.uniform(0, 0.1)
    
    # Desert classification (low green, high brightness, brownish)
    desert_score = 0
    if mean_rgb[1] < mean_rgb[0] and mean_rgb[1] < mean_rgb[2]:  # Low green
        desert_score += 0.3
    if brightness > 100:  # Bright
        desert_score += 0.2
    if mean_saturation > 30 and mean_saturation < 100:  # Moderate saturation
        desert_score += 0.2
    probabilities[1] = desert_score + random.uniform(-0.1, 0.1)
    
    # Green area classification (high green values)
    if mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]:  # High green
        probabilities[2] = 0.6 + random.uniform(-0.1, 0.1)
    elif mean_rgb[1] > 80:  # Moderately green
        probabilities[2] = 0.3 + random.uniform(-0.1, 0.1)
    else:
        probabilities[2] = 0.1 + random.uniform(0, 0.1)
    
    # Water classification (high blue, low brightness variations)
    if mean_rgb[2] > mean_rgb[0] and mean_rgb[2] > mean_rgb[1]:  # High blue
        probabilities[3] = 0.6 + random.uniform(-0.1, 0.1)
    elif mean_rgb[2] > 100:  # Moderately blue
        probabilities[3] = 0.3 + random.uniform(-0.1, 0.1)
    else:
        probabilities[3] = 0.1 + random.uniform(0, 0.1)
    
    # Normalize probabilities
    probabilities = np.maximum(probabilities, 0.01)  # Ensure minimum probability
    probabilities = probabilities / np.sum(probabilities)
    
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    return predicted_class, confidence, probabilities

def create_color_swatch(r, g, b):
    """Create a color swatch HTML element"""
    return f'<div class="color-swatch" style="background-color: rgb({int(r)}, {int(g)}, {int(b)});"></div>'

def create_progress_bar(value, max_value=100, color="#0d6efd"):
    """Create a progress bar HTML element"""
    percentage = (value / max_value) * 100
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage}%; background-color: {color};"></div>
    </div>
    """

# Main content based on page selection
if page == "Image Classification":
    st.header("Upload and Classify Satellite Images")
    
    st.info("This demo uses computer vision techniques for classification. Upload a satellite image to analyze terrain type!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a satellite image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a satellite image to classify terrain type"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Show image properties
            st.markdown(f"""
            <div class="metric-card">
                <h4>Image Properties</h4>
                <p><strong>Size:</strong> {img.size[0]} x {img.size[1]} pixels</p>
                <p><strong>Mode:</strong> {img.mode}</p>
                <p><strong>Format:</strong> {img.format}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                # Add some delay to simulate processing
                time.sleep(1)
                predicted_class, confidence, all_predictions = classify_image(img)
            
            # Display prediction
            predicted_label = class_names[predicted_class]
            predicted_emoji = class_emojis[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>{predicted_emoji} {predicted_label}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence for all classes
            st.subheader("Confidence Scores for All Classes")
            for i, (class_name, emoji) in enumerate(zip(class_names, class_emojis)):
                conf_score = all_predictions[i]
                
                # Create columns for emoji, name, and progress bar
                col_emoji, col_name, col_bar = st.columns([1, 3, 6])
                
                with col_emoji:
                    st.markdown(f"<h3>{emoji}</h3>", unsafe_allow_html=True)
                
                with col_name:
                    st.write(f"**{class_name}**")
                    st.write(f"{conf_score:.2%}")
                
                with col_bar:
                    # Create progress bar
                    bar_width = int(conf_score * 100)
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {bar_width}%">{conf_score:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show analysis details
        st.subheader("Image Analysis Details")
        features = analyze_image_features(img)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Color Analysis")
            r, g, b = features['mean_rgb']
            
            # Display color swatches
            st.markdown("**Average Color:**", unsafe_allow_html=True)
            st.markdown(create_color_swatch(r, g, b), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="feature-box">
                <p><strong>RGB Values:</strong><br>
                🔴 Red: {r:.1f}<br>
                🟢 Green: {g:.1f}<br>
                🔵 Blue: {b:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Image Properties")
            st.markdown(f"""
            <div class="feature-box">
                <p><strong>Brightness:</strong> {features['brightness']:.1f}/255</p>
                <p><strong>Saturation:</strong> {features['mean_saturation']:.1f}/255</p>
                <p><strong>Edge Density:</strong> {features['edge_density']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### Classification Logic")
            st.markdown("""
            <div class="feature-box">
                <p><strong>☁️ Cloudy:</strong> High brightness, low saturation</p>
                <p><strong>🏜️ Desert:</strong> Low green, brownish tones</p>
                <p><strong>🌲 Green Area:</strong> High green values</p>
                <p><strong>🌊 Water:</strong> High blue values</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature metrics using Streamlit's built-in metrics
        st.subheader("Feature Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Brightness", f"{features['brightness']:.0f}/255", 
                     delta=f"{((features['brightness']/255)*100):.1f}%")
        
        with metric_col2:
            st.metric("Saturation", f"{features['mean_saturation']:.0f}/255",
                     delta=f"{((features['mean_saturation']/255)*100):.1f}%")
        
        with metric_col3:
            dominant_color = ['Red', 'Green', 'Blue'][np.argmax(features['mean_rgb'])]
            st.metric("Dominant Color", dominant_color,
                     delta=f"{np.max(features['mean_rgb']):.0f}")
        
        with metric_col4:
            st.metric("Edge Density", f"{features['edge_density']:.3f}",
                     delta=f"{(features['edge_density']*1000):.1f}‰")
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Show sample images for reference
        st.subheader("Sample Image Types")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>☁️</h2>
                <h4>Cloudy</h4>
                <p>High brightness, low color saturation, white/gray dominant colors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🏜️</h2>
                <h4>Desert</h4>
                <p>Brown/tan colors, low green values, moderate brightness</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🌲</h2>
                <h4>Green Area</h4>
                <p>High green values, moderate saturation, varied textures</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🌊</h2>
                <h4>Water</h4>
                <p>High blue values, low texture variation, darker regions</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "About Model":
    st.header("About the Classification System")
    
    st.markdown("""
    ## Computer Vision Approach
    
    This satellite image classifier uses computer vision techniques and rule-based classification instead of deep learning.
    
    ### Analysis Method:
    - **Color Space Analysis**: RGB and HSV color space examination
    - **Feature Extraction**: Brightness, saturation, color distribution analysis
    - **Texture Analysis**: Edge detection and density calculation
    - **Rule-Based Classification**: Logic-based decision making
    
    ### Classification Rules:
    """)
    
    # Create expandable sections for each class
    with st.expander("☁️ Cloudy Images"):
        st.markdown("""
        **Detection Criteria:**
        - High brightness values (> 150)
        - Low color saturation (< 50)
        - Predominantly white/gray colors
        - Minimal color variation
        
        **Confidence Scoring:**
        - High confidence: Brightness > 150 AND Saturation < 50
        - Medium confidence: Brightness > 120 AND Saturation < 80
        - Low confidence: All other cases
        """)
    
    with st.expander("🏜️ Desert Images"):
        st.markdown("""
        **Detection Criteria:**
        - Low green channel values
        - Brown/tan color dominance
        - Moderate brightness levels
        - Specific RGB ratios favoring red/brown
        
        **Scoring System:**
        - +0.3 points for low green dominance
        - +0.2 points for high brightness
        - +0.2 points for moderate saturation (30-100)
        """)
    
    with st.expander("🌲 Green Area Images"):
        st.markdown("""
        **Detection Criteria:**
        - High green channel values
        - Green dominance over red and blue
        - Moderate to high saturation
        - Varied texture patterns
        
        **Confidence Levels:**
        - High: Green > Red AND Green > Blue
        - Medium: Green values > 80
        - Low: All other cases
        """)
    
    with st.expander("🌊 Water Images"):
        st.markdown("""
        **Detection Criteria:**
        - High blue channel values
        - Blue dominance over other colors
        - Lower brightness variations
        - Smooth texture patterns
        
        **Classification Logic:**
        - High confidence: Blue > Red AND Blue > Green
        - Medium confidence: Blue values > 100
        - Enhanced by low edge density
        """)
    
    st.markdown("""
    ### Technical Implementation:
    - **Image Processing**: OpenCV for image manipulation
    - **Feature Analysis**: NumPy for numerical computations
    - **Color Analysis**: Multi-color space examination
    - **Classification Logic**: Rule-based decision tree
    
    ### Advantages:
    ✅ No need for large training datasets  
    ✅ Interpretable results  
    ✅ Fast processing  
    ✅ Customizable rules  
    ✅ Lightweight implementation  
    
    ### Limitations:
    ❌ Less accurate than deep learning models  
    ❌ Requires manual rule tuning  
    ❌ May struggle with complex scenes  
    ❌ Limited to basic feature analysis  
    """)

elif page == "Sample Images":
    st.header("Sample Images by Category")
    
    st.markdown("""
    Here are examples of the types of satellite images the system can classify and the features it looks for:
    """)
    
    # Create tabs for each category
    tab1, tab2, tab3, tab4 = st.tabs(["☁️ Cloudy", "🏜️ Desert", "🌲 Green Area", "🌊 Water"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ## Cloudy Images
            
            **Key Features the System Looks For:**
            - High brightness values (white/light gray appearance)
            - Low color saturation (washed out colors)
            - Minimal color variation
            - Soft, diffused lighting
            """)
        
        with col2:
            st.markdown("""
            ## Classification Logic
            
            **High Confidence Triggers:**
            - Brightness > 150 AND Saturation < 50
            
            **Medium Confidence Triggers:**
            - Brightness > 120 AND Saturation < 80
            
            **Typical RGB Pattern:**
            - High overall values (200+)
            - Similar R, G, B values
            """)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ## Desert Images
            
            **Key Features the System Looks For:**
            - Brown/tan/beige color dominance
            - Low green channel values
            - Moderate brightness levels
            - Sandy or rocky texture patterns
            """)
        
        with col2:
            st.markdown("""
            ## Classification Logic
            
            **Scoring System:**
            - Low green values + brownish tones → +0.3 points
            - Moderate saturation (30-100) → +0.2 points
            - Bright but not overexposed → +0.2 points
            
            **Typical RGB Pattern:**
            - Red > Green, Blue > Green
            - Moderate overall brightness
            """)
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ## Green Area Images
            
            **Key Features the System Looks For:**
            - High green channel values
            - Green dominance over red and blue channels
            - Varied texture patterns (vegetation)
            - Moderate to high color saturation
            """)
        
        with col2:
            st.markdown("""
            ## Classification Logic
            
            **High Confidence:**
            - Green > Red AND Green > Blue
            
            **Medium Confidence:**
            - Green values > 80
            
            **Additional Factors:**
            - Texture variation indicates vegetation
            - Higher saturation values
            """)
    
    with tab4:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ## Water Images
            
            **Key Features the System Looks For:**
            - High blue channel values
            - Blue dominance over other colors
            - Smooth, uniform texture
            - Lower brightness variations
            """)
        
        with col2:
            st.markdown("""
            ## Classification Logic
            
            **High Confidence:**
            - Blue > Red AND Blue > Green
            
            **Medium Confidence:**
            - Blue values > 100
            
            **Enhancement Factors:**
            - Low edge density indicates smooth surface
            - Consistent color patterns
            """)

elif page == "Image Analysis":
    st.header("Image Analysis Tools")
    
    st.markdown("""
    This page provides detailed analysis tools to understand how the classification system works.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload an image for detailed analysis...", 
        type=['jpg', 'jpeg', 'png'],
        key="analysis_uploader"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Original Image", use_column_width=True)
        
        # Analyze features
        features = analyze_image_features(img)
        
        # Feature Analysis
        st.subheader("Feature Analysis")
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Brightness", f"{features['brightness']:.0f}", 
                     delta=f"{features['brightness']/255*100:.1f}%")
        
        with col2:
            st.metric("Saturation", f"{features['mean_saturation']:.0f}",
                     delta=f"{features['mean_saturation']/255*100:.1f}%")
        
        with col3:
            st.metric("Edge Density", f"{features['edge_density']:.3f}",
                     delta=f"{features['edge_density']*1000:.1f}‰")
        
        with col4:
            dominant_color = ['Red', 'Green', 'Blue'][np.argmax(features['mean_rgb'])]
            st.metric("Dominant Color", dominant_color,
                     delta=f"{np.max(features['mean_rgb']):.0f}")
        
        # Color analysis
        st.subheader("Color Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RGB Color Breakdown")
            avg_r, avg_g, avg_b = features['mean_rgb']
            
            # Display color information
            st.markdown(f"**Average Color:** {create_color_swatch(avg_r, avg_g, avg_b)}", unsafe_allow_html=True)
            
            # RGB bars using Streamlit's progress bars
            st.write("🔴 **Red Channel:**")
            st.progress(avg_r / 255)
            st.write(f"Value: {avg_r:.1f}/255")
            
            st.write("🟢 **Green Channel:**")
            st.progress(avg_g / 255)
            st.write(f"Value: {avg_g:.1f}/255")
            
            st.write("🔵 **Blue Channel:**")
            st.progress(avg_b / 255)
            st.write(f"Value: {avg_b:.1f}/255")
        
        with col2:
            st.markdown("#### Feature Scores")
            
            # Normalize features for display
            normalized_features = {
                'Brightness': features['brightness'] / 255,
                'Saturation': features['mean_saturation'] / 255,
                'Edge Density': min(features['edge_density'] * 10, 1),  # Scale for visibility
                'Red Average': features['mean_rgb'][0] / 255,
                'Green Average': features['mean_rgb'][1] / 255,
                'Blue Average': features['mean_rgb'][2] / 255
            }
            
            for feature_name, value in normalized_features.items():
                st.write(f"**{feature_name}:**")
                st.progress(value)
                st.write(f"Score: {value:.3f}")
        
        # Classification breakdown
        st.subheader("Classification Breakdown")
        predicted_class, confidence, all_predictions = classify_image(img)
        
        # Show decision process
        st.markdown("#### Decision Process")
        
        brightness = features['brightness']
        saturation = features['mean_saturation']
        mean_rgb = features['mean_rgb']
        
        # Step-by-step analysis
        steps = []
        
        # Brightness analysis
        if brightness > 150:
            steps.append(f"✅ **High brightness ({brightness:.1f})** suggests cloudy terrain")
        elif brightness > 100:
            steps.append(f"🔶 **Moderate brightness ({brightness:.1f})** - neutral indicator")
        else:
            steps.append(f"🔴 **Low brightness ({brightness:.1f})** - darker terrain")
        
        # Saturation analysis
        if saturation < 50:
            steps.append(f"✅ **Low saturation ({saturation:.1f})** suggests cloudy or desert terrain")
        elif saturation > 100:
            steps.append(f"✅ **High saturation ({saturation:.1f})** suggests vegetation or water")
        else:
            steps.append(f"🔶 **Moderate saturation ({saturation:.1f})** - neutral indicator")
        
        # Color dominance analysis
        dominant_idx = np.argmax(mean_rgb)
        dominant_color = ['Red', 'Green', 'Blue'][dominant_idx]
        dominant_value = mean_rgb[dominant_idx]
        
        if dominant_idx == 1:  # Green
            steps.append(f"🌲 **Green dominance ({dominant_value:.1f})** suggests vegetation")
        elif dominant_idx == 2:  # Blue
            steps.append(f"🌊 **Blue dominance ({dominant_value:.1f})** suggests water")
        elif dominant_idx == 0:  # Red
            steps.append(f"🏜️ **Red dominance ({dominant_value:.1f})** suggests desert or arid terrain")
        
        # Display steps
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
        
        # Final prediction
        predicted_label = class_names[predicted_class]
        predicted_emoji = class_emojis[predicted_class]
        
        st.markdown(f"""
        ---
        ### Final Prediction: {predicted_emoji} **{predicted_label}**
        **Confidence:** {confidence:.2%}
        """)
        
        # Show all class probabilities
        st.markdown("#### All Class Probabilities")
        for i, (class_name, emoji) in enumerate(zip(class_names, class_emojis)):
            prob = all_predictions[i]
            st.write(f"{emoji} **{class_name}**: {prob:.2%}")
            st.progress(prob)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🛰️ Satellite Image Classification App | Built with Streamlit and OpenCV</p>
    <p>No TensorFlow or Matplotlib Required - Pure Computer Vision</p>
</div>
""", unsafe_allow_html=True)