"""Streamlit aerial image classifier — CSC3109 Group 30."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = Path('model/efficientnet_b0_best.pth')
CLASSES_PATH = Path('model/classes.json')
CONFIDENCE_THRESHOLD = 0.50

CLASS_ICONS = {
    'oil_well':                    '🛢️',
    'solar_panel':                 '☀️',
    'transformer_station':         '⚡',
    'wastewater_treatment_plant':  '💧',
}
CLASS_LABELS = {
    'oil_well':                    'Oil Well',
    'solar_panel':                 'Solar Panel',
    'transformer_station':         'Transformer Station',
    'wastewater_treatment_plant':  'Wastewater Treatment Plant',
}
CLASS_COLORS = {
    'oil_well':                    '#e67e22',
    'solar_panel':                 '#f1c40f',
    'transformer_station':         '#3498db',
    'wastewater_treatment_plant':  '#2ecc71',
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CLASSES_PATH) as f:
        idx2cls = json.load(f)
    num_classes = len(idx2cls)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, idx2cls


def predict(model, idx2cls, image: Image.Image):
    tensor = TRANSFORM(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    scores = {idx2cls[str(i)]: float(probs[i]) for i in range(len(idx2cls))}
    top    = max(scores, key=scores.get)
    return top, scores


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Aerial Image Classifier',
    page_icon='🛰️',
    layout='centered',
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    .warning-box {
        background-color: #3e1f00;
        border: 1px solid #e64a19;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #ffccbc;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🛰️ Aerial Image Classifier</h1></div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">CSC3109 Machine Learning &nbsp;|&nbsp; Group 30 '
            '&nbsp;|&nbsp; EfficientNet-B0</div>', unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
with st.spinner('Loading model...'):
    model, idx2cls = load_model()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    'Upload an aerial image (JPG, PNG)',
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help='Best results with overhead/satellite images of infrastructure'
)

if uploaded:
    image = Image.open(uploaded).convert('RGB')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with col2:
        with st.spinner('Classifying...'):
            top_class, scores = predict(model, idx2cls, image)

        confidence = scores[top_class]
        icon  = CLASS_ICONS.get(top_class, '❓')
        label = CLASS_LABELS.get(top_class, top_class)
        color = CLASS_COLORS.get(top_class, '#ffffff')

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="warning-box">'
                '<b>⚠️ Low Confidence</b><br>'
                'This image may not belong to any trained category. '
                f'Max confidence: {confidence:.1%}'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown(f'### {icon} {label}')
        st.markdown(
            f'<span style="font-size:2rem; font-weight:700; color:{color}">'
            f'{confidence:.2%}</span> confidence',
            unsafe_allow_html=True
        )

    # ── Probability bars ──────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('#### All Class Probabilities')

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_scores:
        icon  = CLASS_ICONS.get(cls, '')
        label = CLASS_LABELS.get(cls, cls)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(prob, text=f'{icon} {label}')
        with col_b:
            st.markdown(
                f'<div style="padding-top:0.4rem; text-align:right; font-weight:600">'
                f'{prob:.2%}</div>',
                unsafe_allow_html=True
            )

    # ── JSON output ───────────────────────────────────────────────────────
    st.markdown('---')
    with st.expander('Raw JSON Response'):
        st.json({
            'filename':        uploaded.name,
            'predicted_class': top_class,
            'confidence':      round(confidence, 6),
            'probabilities':   {k: round(v, 6) for k, v in scores.items()},
            **(
                {'warning': f'Low confidence ({confidence:.2%}) — image may not be in training categories'}
                if confidence < CONFIDENCE_THRESHOLD else {}
            )
        })

else:
    st.info('Upload an aerial image above to get a classification result.')
    st.markdown("""
    **Supported categories:**
    | Icon | Category |
    |------|----------|
    | 🛢️ | Oil Well |
    | ☀️ | Solar Panel |
    | ⚡ | Transformer Station |
    | 💧 | Wastewater Treatment Plant |
    """)
