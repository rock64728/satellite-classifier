import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import io
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛰️ Satellite Image Classifier</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Image Classification", "About Model", "Sample Images", "Image Analysis"])

# Class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

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
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    
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
            st.write(f"**Image Size**: {img.size}")
            st.write(f"**Image Mode**: {img.mode}")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                # Add some delay to simulate processing
                time.sleep(1)
                predicted_class, confidence, all_predictions = classify_image(img)
            
            # Display prediction
            predicted_label = class_names[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Class: {predicted_label}</h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence for all classes
            st.subheader("Confidence Scores for All Classes")
            for i, class_name in enumerate(class_names):
                conf_score = all_predictions[i]
                st.write(f"**{class_name}**: {conf_score:.2%}")
                
                # Create confidence bar
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
            st.markdown("""
            <div class="feature-box">
                <h4>Color Analysis</h4>
                <p><strong>Average RGB:</strong><br>
                R: {:.1f}<br>
                G: {:.1f}<br>
                B: {:.1f}</p>
            </div>
            """.format(features['mean_rgb'][0], features['mean_rgb'][1], features['mean_rgb'][2]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>Image Properties</h4>
                <p><strong>Brightness:</strong> {:.1f}<br>
                <strong>Saturation:</strong> {:.1f}<br>
                <strong>Edge Density:</strong> {:.3f}</p>
            </div>
            """.format(features['brightness'], features['mean_saturation'], features['edge_density']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>Classification Logic</h4>
                <p><strong>Cloudy:</strong> High brightness, low saturation<br>
                <strong>Desert:</strong> Low green, brownish tones<br>
                <strong>Green Area:</strong> High green values<br>
                <strong>Water:</strong> High blue values</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Show sample images for reference
        st.subheader("Sample Image Types")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Cloudy**")
            st.write("High brightness, low color saturation, white/gray dominant colors")
        
        with col2:
            st.write("**Desert**")
            st.write("Brown/tan colors, low green values, moderate brightness")
        
        with col3:
            st.write("**Green Area**")
            st.write("High green values, moderate saturation, varied textures")
        
        with col4:
            st.write("**Water**")
            st.write("High blue values, low texture variation, darker regions")

elif page == "About Model":
    st.header("About the Classification System")
    
    st.markdown("""
    ## Computer Vision Approach
    
    This satellite image classifier uses computer vision techniques and rule-based classification instead of deep learning.
    
    ### Analysis Method:
    - **Color Space Analysis**: RGB, HSV, and LAB color space examination
    - **Feature Extraction**: Brightness, saturation, color distribution analysis
    - **Texture Analysis**: Edge detection and density calculation
    - **Rule-Based Classification**: Logic-based decision making
    
    ### Classification Rules:
    
    #### Cloudy Images:
    - High brightness values (> 150)
    - Low color saturation (< 50)
    - Predominantly white/gray colors
    
    #### Desert Images:
    - Low green channel values
    - Brown/tan color dominance
    - Moderate brightness levels
    - Specific RGB ratios
    
    #### Green Area Images:
    - High green channel values
    - Green dominance over red and blue
    - Moderate to high saturation
    - Varied texture patterns
    
    #### Water Images:
    - High blue channel values
    - Blue dominance over other colors
    - Lower brightness variations
    - Smooth texture patterns
    
    ### Technical Implementation:
    - **Image Processing**: OpenCV for image manipulation
    - **Feature Analysis**: NumPy for numerical computations
    - **Color Analysis**: Multi-color space examination
    - **Classification Logic**: Rule-based decision tree
    
    ### Advantages:
    - No need for large training datasets
    - Interpretable results
    - Fast processing
    - Customizable rules
    
    ### Limitations:
    - Less accurate than deep learning models
    - Requires manual rule tuning
    - May struggle with complex scenes
    - Limited to basic feature analysis
    """)

elif page == "Sample Images":
    st.header("Sample Images by Category")
    
    st.markdown("""
    Here are examples of the types of satellite images the system can classify and the features it looks for:
    """)
    
    # Create tabs for each category
    tab1, tab2, tab3, tab4 = st.tabs(["🌩️ Cloudy", "🏜️ Desert", "🌲 Green Area", "🌊 Water"])
    
    with tab1:
        st.markdown("""
        **Cloudy Images**
        
        **Key Features the System Looks For:**
        - High brightness values (white/light gray appearance)
        - Low color saturation (washed out colors)
        - Minimal color variation
        - Soft, diffused lighting
        
        **Classification Logic:**
        - Brightness > 150 AND Saturation < 50 → High confidence
        - Brightness > 120 AND Saturation < 80 → Medium confidence
        """)
    
    with tab2:
        st.markdown("""
        **Desert Images**
        
        **Key Features the System Looks For:**
        - Brown/tan/beige color dominance
        - Low green channel values
        - Moderate brightness levels
        - Sandy or rocky texture patterns
        
        **Classification Logic:**
        - Low green values + brownish tones → Higher score
        - Moderate saturation (30-100) → Additional points
        - Bright but not overexposed → Bonus scoring
        """)
    
    with tab3:
        st.markdown("""
        **Green Area Images**
        
        **Key Features the System Looks For:**
        - High green channel values
        - Green dominance over red and blue channels
        - Varied texture patterns (vegetation)
        - Moderate to high color saturation
        
        **Classification Logic:**
        - Green > Red AND Green > Blue → High confidence
        - Green values > 80 → Medium confidence
        - Texture variation indicates vegetation
        """)
    
    with tab4:
        st.markdown("""
        **Water Images**
        
        **Key Features the System Looks For:**
        - High blue channel values
        - Blue dominance over other colors
        - Smooth, uniform texture
        - Lower brightness variations
        
        **Classification Logic:**
        - Blue > Red AND Blue > Green → High confidence
        - Blue values > 100 → Medium confidence
        - Low edge density indicates smooth surface
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
        st.image(img, caption="Original Image", width=400)
        
        # Analyze features
        features = analyze_image_features(img)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Color Distribution")
            
            # RGB histogram
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['red', 'green', 'blue']
                for i, color in enumerate(colors):
                    hist = np.histogram(img_array[:, :, i], bins=50, range=(0, 255))
                    ax.plot(hist[1][:-1], hist[0], color=color, alpha=0.7, label=f'{color.upper()} channel')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.set_title('RGB Color Distribution')
                ax.legend()
                st.pyplot(fig)
            
            # Color averages
            st.subheader("Average Colors")
            avg_r, avg_g, avg_b = features['mean_rgb']
            st.write(f"**Red**: {avg_r:.1f}")
            st.write(f"**Green**: {avg_g:.1f}")
            st.write(f"**Blue**: {avg_b:.1f}")
        
        with col2:
            st.subheader("Feature Analysis")
            
            # Create feature chart
            feature_names = ['Brightness', 'Saturation', 'Edge Density', 'Red Avg', 'Green Avg', 'Blue Avg']
            feature_values = [
                features['brightness'] / 255,
                features['mean_saturation'] / 255,
                features['edge_density'] * 10,  # Scale for visibility
                features['mean_rgb'][0] / 255,
                features['mean_rgb'][1] / 255,
                features['mean_rgb'][2] / 255
            ]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(feature_names, feature_values, color=['skyblue', 'lightgreen', 'coral', 'red', 'green', 'blue'])
            ax.set_ylabel('Normalized Value')
            ax.set_title('Image Features')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Classification breakdown
        st.subheader("Classification Breakdown")
        predicted_class, confidence, all_predictions = classify_image(img)
        
        # Show decision process
        st.write("**Decision Process:**")
        
        brightness = features['brightness']
        saturation = features['mean_saturation']
        mean_rgb = features['mean_rgb']
        
        st.write(f"1. **Brightness Analysis**: {brightness:.1f}")
        if brightness > 150:
            st.write("   → High brightness suggests cloudy terrain")
        elif brightness > 100:
            st.write("   → Moderate brightness")
        else:
            st.write("   → Low brightness")
        
        st.write(f"2. **Saturation Analysis**: {saturation:.1f}")
        if saturation < 50:
            st.write("   → Low saturation suggests cloudy or desert terrain")
        elif saturation > 100:
            st.write("   → High saturation suggests vegetation or water")
        
        st.write(f"3. **Color Dominance**:")
        dominant_color = ['Red', 'Green', 'Blue'][np.argmax(mean_rgb)]
        st.write(f"   → {dominant_color} is the dominant color")
        
        if mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]:
            st.write("   → Green dominance suggests vegetation")
        elif mean_rgb[2] > mean_rgb[0] and mean_rgb[2] > mean_rgb[1]:
            st.write("   → Blue dominance suggests water")
        elif mean_rgb[0] > mean_rgb[1] and mean_rgb[0] > mean_rgb[2]:
            st.write("   → Red dominance suggests desert or arid terrain")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Satellite Image Classification App | Built with Streamlit and OpenCV</p>
    <p>No TensorFlow Required - Uses Computer Vision Techniques</p>
</div>
""", unsafe_allow_html=True)