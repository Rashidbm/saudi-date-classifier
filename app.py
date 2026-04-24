"""
Saudi Date Fruit Variety Classifier - Gradio Web App

Upload a date fruit image, get the variety prediction from the ensemble model
(ResNet + EfficientNet + ViT) with confidence percentages, heritage info,
and a Grad-CAM heatmap showing what the model looked at.

Supports English and Arabic with a language switcher.

Run locally: python app.py
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import get_val_transforms
from src.ensemble import (
    CHECKPOINTS,
    CLASS_NAMES,
    HF_REPO_ID,
    PretrainedViTClassifier,
    build_efficientnet,
    build_resnet50,
    load_checkpoint,
)
from src.explainability import reshape_transform
from src.utils import HERITAGE_INFO, get_device, load_config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

print("Loading models...")
config = load_config()
device = get_device()
transform = get_val_transforms(config)

paths = {
    name: hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
    for name, fname in CHECKPOINTS.items()
}

models_dict = {
    "resnet": load_checkpoint(build_resnet50(num_classes=9), paths["resnet"], device),
    "efficientnet": load_checkpoint(build_efficientnet(num_classes=9), paths["efficientnet"], device),
    "vit": load_checkpoint(PretrainedViTClassifier(num_classes=9), paths["vit"], device),
}
print(f"All models loaded on {device}")

vit_target_layer = models_dict["vit"].backbone.vit.encoder.layer[-1].layernorm_before
gradcam = GradCAM(
    model=models_dict["vit"],
    target_layers=[vit_target_layer],
    reshape_transform=reshape_transform,
)


# ---------------------------- i18n ----------------------------

UI = {
    "en": {
        "lang_label": "Language",
        "eyebrow_hero": "Saudi Heritage · Deep Learning",
        "hero_title_pre": "Saudi Date",
        "hero_title_em": "Variety",
        "hero_title_post": "Classifier",
        "hero_subtitle": "Upload a photograph of a date fruit to identify its variety, see where the model looked with Grad-CAM, and learn the heritage behind each cultivar.",
        "stat_varieties": "Varieties",
        "stat_ensemble": "Model Ensemble",
        "stat_accuracy": "Test Accuracy",
        "tab_classify": "Classify",
        "tab_features": "Feature Space",
        "step1": "Step 1 · Input",
        "step2": "Step 2 · Result",
        "image_label": "Date fruit image",
        "model_label": "Model",
        "model_info": "Ensemble combines all three models for best accuracy.",
        "classify_btn": "Classify",
        "top_preds": "Top predictions",
        "gradcam_eyebrow": "Model Attention · Grad-CAM",
        "gradcam_caption": "Warm regions show where the Vision Transformer attended most when forming its prediction — a window into the model's reasoning.",
        "heritage_eyebrow": "Heritage",
        "empty_pred_label": "Predicted Variety",
        "empty_pred_hint": "Upload an image to begin",
        "empty_heritage_title": "Heritage Information",
        "empty_heritage_body": "Upload a date image to see cultural details about its variety — origin region, flavor profile, and significance in Saudi heritage.",
        "pred_confidence": "confidence",
        "heritage_region": "Region",
        "heritage_desc": "Description",
        "heritage_flavor": "Flavor",
        "heritage_sig": "Cultural Significance",
        "tsne_eyebrow": "t-SNE Projection",
        "tsne_title": "How the model clusters varieties",
        "tsne_subtitle": "2-D projection of Vision Transformer embeddings on the held-out test set. Each point is one image; well-separated clusters mean the model has learned to distinguish varieties clearly.",
        "tsne_missing": "_t-SNE plot not available. Run_ `python -m src.explainability --tsne` _first._",
        "footer": "Classifies 9 Saudi date varieties — Ajwa, Galaxy, Medjool, Meneifi, Nabtat Ali, Rutab, Shaishe, Sokari, and Sugaey — using a ResNet-50 + EfficientNet-B0 + Vision Transformer ensemble.",
        "footer_link": "View model weights on Hugging Face",
    },
    "ar": {
        "lang_label": "اللغة",
        "eyebrow_hero": "تراث سعودي · تعلّم عميق",
        "hero_title_pre": "مصنّف",
        "hero_title_em": "أصناف",
        "hero_title_post": "التمور السعودية",
        "hero_subtitle": "ارفع صورة لثمرة تمر لتحديد صنفها، وشاهد أين ركّز النموذج عبر Grad-CAM، واكتشف التراث وراء كل صنف من أصناف التمر السعودي.",
        "stat_varieties": "أصناف",
        "stat_ensemble": "نماذج مجمّعة",
        "stat_accuracy": "دقة الاختبار",
        "tab_classify": "التصنيف",
        "tab_features": "فضاء السمات",
        "step1": "الخطوة الأولى · الإدخال",
        "step2": "الخطوة الثانية · النتيجة",
        "image_label": "صورة التمرة",
        "model_label": "النموذج",
        "model_info": "النموذج المجمّع يدمج النماذج الثلاثة للحصول على أعلى دقة.",
        "classify_btn": "تصنيف",
        "top_preds": "أفضل التوقّعات",
        "gradcam_eyebrow": "انتباه النموذج · Grad-CAM",
        "gradcam_caption": "المناطق الدافئة تُظهر أين ركّز محوّل الرؤية عند تكوين توقّعه — نافذة على استدلال النموذج.",
        "heritage_eyebrow": "التراث",
        "empty_pred_label": "الصنف المتوقّع",
        "empty_pred_hint": "ارفع صورة للبدء",
        "empty_heritage_title": "المعلومات التراثية",
        "empty_heritage_body": "ارفع صورة تمرة لرؤية تفاصيل ثقافية عن صنفها — المنطقة الأصلية، والنكهة، والأهمية في التراث السعودي.",
        "pred_confidence": "ثقة",
        "heritage_region": "المنطقة",
        "heritage_desc": "الوصف",
        "heritage_flavor": "النكهة",
        "heritage_sig": "الأهمية الثقافية",
        "tsne_eyebrow": "إسقاط t-SNE",
        "tsne_title": "كيف يجمّع النموذج الأصناف",
        "tsne_subtitle": "إسقاط ثنائي الأبعاد لتضمينات محوّل الرؤية على مجموعة الاختبار. كل نقطة تمثّل صورة واحدة؛ التباعد بين المجموعات يعني أن النموذج تعلّم تمييز الأصناف بوضوح.",
        "tsne_missing": "_مخطط t-SNE غير متوفر. شغّل_ `python -m src.explainability --tsne` _أولاً._",
        "footer": "يصنّف تسعة من أصناف التمور السعودية — عجوة، قلاكسي، مجدول، منيفي، نبتة علي، رطب، شيشي، سكري، وصقعي — عبر نموذج مجمّع من ResNet-50 و EfficientNet-B0 ومحوّل الرؤية.",
        "footer_link": "عرض أوزان النماذج على Hugging Face",
    },
}


# ---------------------------- Theme & CSS ----------------------------

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"),
).set(
    body_background_fill="#FFFBEB",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="#F1E8E2",
    block_shadow="0 1px 2px rgba(146, 64, 14, 0.04)",
    block_radius="16px",
    block_label_text_color="#57534E",
    block_label_text_size="0.75rem",
    block_label_text_weight="600",
    button_primary_background_fill="#92400E",
    button_primary_background_fill_hover="#78350F",
    button_primary_text_color="white",
    button_primary_border_color="#92400E",
    button_large_radius="12px",
    button_large_text_size="16px",
    input_background_fill="white",
    input_border_color="#F1E8E2",
    input_border_color_focus="#A16207",
    input_radius="10px",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

@font-face {
  font-family: 'ThmanyahSans';
  src: url('/gradio_api/file=static/fonts/thmanyahsans-Light.woff2') format('woff2');
  font-weight: 300; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSans';
  src: url('/gradio_api/file=static/fonts/thmanyahsans-Regular.woff2') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSans';
  src: url('/gradio_api/file=static/fonts/thmanyahsans-Medium.woff2') format('woff2');
  font-weight: 500; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSans';
  src: url('/gradio_api/file=static/fonts/thmanyahsans-Bold.woff2') format('woff2');
  font-weight: 700; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSans';
  src: url('/gradio_api/file=static/fonts/thmanyahsans-Black.woff2') format('woff2');
  font-weight: 900; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSerif';
  src: url('/gradio_api/file=static/fonts/thmanyahserifdisplay-Regular.woff2') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSerif';
  src: url('/gradio_api/file=static/fonts/thmanyahserifdisplay-Medium.woff2') format('woff2');
  font-weight: 500; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSerif';
  src: url('/gradio_api/file=static/fonts/thmanyahserifdisplay-Bold.woff2') format('woff2');
  font-weight: 700; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'ThmanyahSerif';
  src: url('/gradio_api/file=static/fonts/thmanyahserifdisplay-Black.woff2') format('woff2');
  font-weight: 900; font-style: normal; font-display: swap;
}

:root {
  --date-brown: #92400E;
  --date-brown-dark: #78350F;
  --date-gold: #A16207;
  --date-amber: #D97706;
  --warm-bg: #FFFBEB;
  --warm-bg-2: #FDF4DD;
  --warm-muted: #F8F3F0;
  --warm-border: #F1E8E2;
  --deep-fg: #1C1917;
  --body-fg: #3F3A36;
  --muted-fg: #57534E;
  --subtle-fg: #78716C;
  --radius-lg: 16px;
  --radius-md: 12px;
  --shadow-sm: 0 1px 2px rgba(146, 64, 14, 0.04);
  --shadow-md: 0 4px 12px rgba(146, 64, 14, 0.08);
  --shadow-lg: 0 12px 28px rgba(146, 64, 14, 0.10);
  --transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);

  /* Default font (English) */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-serif: 'Playfair Display', Georgia, serif;
}

.gradio-container.lang-ar {
  --font-sans: 'ThmanyahSans', system-ui, sans-serif;
  --font-serif: 'ThmanyahSerif', Georgia, serif;
  direction: rtl;
}
.gradio-container.lang-ar button,
.gradio-container.lang-ar input,
.gradio-container.lang-ar label,
.gradio-container.lang-ar textarea {
  direction: rtl;
  text-align: right;
}
.gradio-container.lang-ar .stat-card__label,
.gradio-container.lang-ar .eyebrow,
.gradio-container.lang-ar .hero__eyebrow {
  text-transform: none;
  letter-spacing: 0.04em;
}

.gradio-container {
  background: linear-gradient(180deg, #FFFBEB 0%, #FDF4DD 100%) !important;
  font-family: var(--font-sans) !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}

/* --- Language switcher --- */
.lang-switcher {
  display: flex; justify-content: flex-end;
  padding: 0.75rem 1rem 0;
}

/* --- Hero --- */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero__eyebrow {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--date-brown);
  background: rgba(161, 98, 7, 0.1);
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  margin-bottom: 1rem;
}
.hero__title {
  font-family: var(--font-serif);
  font-size: clamp(2rem, 5.5vw, 3.25rem);
  font-weight: 600;
  color: var(--deep-fg);
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin: 0 0 0.875rem 0;
}
.hero__title em {
  font-style: italic;
  color: var(--date-brown);
  font-weight: 500;
}
.lang-ar .hero__title em { font-style: normal; }
.hero__subtitle {
  font-size: 1.0625rem;
  line-height: 1.6;
  color: var(--body-fg);
  max-width: 620px;
  margin: 0 auto 1.75rem;
}
.hero__stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.875rem;
  max-width: 640px;
  margin: 0 auto;
}
.stat-card {
  background: white;
  border: 1px solid var(--warm-border);
  border-radius: var(--radius-md);
  padding: 1.125rem 0.75rem;
  text-align: center;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}
.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: rgba(161, 98, 7, 0.3);
}
.stat-card__value {
  font-family: var(--font-serif);
  font-size: clamp(1.75rem, 4vw, 2.25rem);
  font-weight: 600;
  color: var(--date-brown);
  line-height: 1;
  margin-bottom: 0.375rem;
  font-variant-numeric: tabular-nums;
}
.stat-card__value sup {
  font-size: 0.55em;
  font-weight: 500;
  margin-left: 0.05em;
}
.stat-card__label {
  font-size: 0.72rem;
  color: var(--muted-fg);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500;
}

/* --- Eyebrow / section labels --- */
.eyebrow {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--date-gold);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 0.25rem 0 0.625rem 0;
}
.section-title {
  font-family: var(--font-serif);
  font-size: clamp(1.5rem, 3vw, 1.875rem);
  font-weight: 600;
  color: var(--deep-fg);
  margin: 0 0 0.5rem 0;
  letter-spacing: -0.01em;
}
.section-subtitle {
  font-size: 0.9375rem;
  line-height: 1.6;
  color: var(--muted-fg);
  margin: 0 auto 1.5rem;
  max-width: 640px;
}

/* --- Prediction card --- */
.prediction-card {
  background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
  border: 1px solid var(--warm-border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.25rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  min-height: 180px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.prediction-card__label {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--muted-fg);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.625rem;
}
.prediction-card__value {
  font-family: var(--font-serif);
  font-size: clamp(1.75rem, 4vw, 2.375rem);
  font-weight: 600;
  color: var(--date-brown);
  line-height: 1.15;
  margin: 0;
  letter-spacing: -0.01em;
}
.prediction-card__arabic {
  font-family: 'ThmanyahSerif', Georgia, serif;
  font-size: 1.25rem;
  color: var(--date-gold);
  margin-top: 0.35rem;
  direction: rtl;
  font-weight: 500;
}
.prediction-card__confidence {
  margin-top: 0.875rem;
  font-size: 0.9375rem;
  color: var(--body-fg);
  font-variant-numeric: tabular-nums;
}
.prediction-card__confidence strong {
  color: var(--date-brown);
  font-weight: 600;
}
.prediction-card__bar {
  margin: 0.875rem auto 0;
  max-width: 280px;
  height: 6px;
  background: #F5EEE5;
  border-radius: 999px;
  overflow: hidden;
}
.prediction-card__bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--date-gold), var(--date-amber));
  border-radius: 999px;
  transition: width 450ms cubic-bezier(0.4, 0, 0.2, 1);
}
.prediction-card--empty {
  background: white;
  border: 1px dashed var(--warm-border);
}
.prediction-card--empty .prediction-card__value {
  color: #D6D3D1;
  font-size: 2rem;
}

/* --- Heritage card --- */
.heritage-wrap {
  background: white !important;
  border: 1px solid var(--warm-border) !important;
  border-left: 3px solid var(--date-gold) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.25rem 1.375rem !important;
  box-shadow: var(--shadow-sm);
  min-height: 320px;
}
.lang-ar .heritage-wrap {
  border-left: 1px solid var(--warm-border) !important;
  border-right: 3px solid var(--date-gold) !important;
}
.heritage-wrap h2,
.heritage-wrap h3 {
  font-family: var(--font-serif) !important;
  color: var(--date-brown) !important;
  margin: 0 0 0.75rem 0 !important;
  font-weight: 600 !important;
  font-size: 1.375rem !important;
  letter-spacing: -0.01em !important;
}
.heritage-wrap p {
  line-height: 1.65 !important;
  color: var(--body-fg) !important;
  margin: 0.5rem 0 !important;
  font-size: 0.9375rem !important;
}
.heritage-wrap strong {
  color: var(--date-brown) !important;
  font-weight: 600 !important;
}

/* --- Grad-CAM caption --- */
.gradcam-caption {
  font-size: 0.8125rem;
  color: var(--muted-fg);
  line-height: 1.55;
  margin-top: 0.5rem;
  padding: 0.625rem 0.875rem;
  background: var(--warm-muted);
  border-radius: 10px;
  border-left: 2px solid var(--date-gold);
}
.lang-ar .gradcam-caption {
  border-left: none;
  border-right: 2px solid var(--date-gold);
}

/* --- Footer --- */
.footer {
  text-align: center;
  padding: 2rem 1rem 1rem;
  margin-top: 2.5rem;
  border-top: 1px solid var(--warm-border);
  color: var(--muted-fg);
  font-size: 0.8125rem;
  line-height: 1.65;
}
.footer a {
  color: var(--date-brown);
  text-decoration: none;
  font-weight: 500;
  transition: var(--transition);
  border-bottom: 1px solid transparent;
}
.footer a:hover,
.footer a:focus-visible {
  color: var(--date-gold);
  border-bottom-color: var(--date-gold);
}

/* --- Buttons --- */
button.primary,
.gradio-container button.primary {
  cursor: pointer !important;
  transition: var(--transition) !important;
  min-height: 48px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
}
button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(146, 64, 14, 0.18) !important;
}
.gradio-container button:focus-visible {
  outline: 3px solid rgba(161, 98, 7, 0.5) !important;
  outline-offset: 2px !important;
}

/* --- Tabs --- */
.gradio-container button[role="tab"] {
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  padding: 0.75rem 1.25rem !important;
  cursor: pointer !important;
  min-height: 44px !important;
  transition: var(--transition) !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
  color: var(--date-brown) !important;
  font-weight: 600 !important;
}

/* --- Responsive --- */
@media (max-width: 640px) {
  .hero { padding: 1.5rem 0.875rem 1rem; }
  .hero__stats { gap: 0.5rem; }
  .stat-card { padding: 0.875rem 0.5rem; }
  .stat-card__value { font-size: 1.5rem; }
  .prediction-card { padding: 1.25rem 1rem; }
  .heritage-wrap { padding: 1rem 1.125rem !important; min-height: auto; }
}

/* --- Reduced motion --- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
  .stat-card:hover,
  button.primary:hover { transform: none !important; }
}
"""


# JavaScript to toggle the language class on the body/container
LANG_JS = """
function(lang) {
  const c = document.querySelector('.gradio-container');
  if (c) {
    c.classList.remove('lang-en', 'lang-ar');
    c.classList.add('lang-' + lang);
    if (lang === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', 'en');
    }
  }
  return lang;
}
"""


# ---------------------------- HTML builders (per language) ----------------------------

def hero_html(lang: str) -> str:
    t = UI[lang]
    if lang == "ar":
        # Arabic title needs different structure (no italic splitting)
        title = f"{t['hero_title_pre']} <em>{t['hero_title_em']}</em> {t['hero_title_post']}"
    else:
        title = f"{t['hero_title_pre']} <em>{t['hero_title_em']}</em> {t['hero_title_post']}"
    return f"""
<div class="hero">
  <div class="hero__eyebrow">{t['eyebrow_hero']}</div>
  <h1 class="hero__title">{title}</h1>
  <p class="hero__subtitle">{t['hero_subtitle']}</p>
  <div class="hero__stats">
    <div class="stat-card">
      <div class="stat-card__value">9</div>
      <div class="stat-card__label">{t['stat_varieties']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-card__value">3</div>
      <div class="stat-card__label">{t['stat_ensemble']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-card__value">90.4<sup>%</sup></div>
      <div class="stat-card__label">{t['stat_accuracy']}</div>
    </div>
  </div>
</div>
"""


def footer_html(lang: str) -> str:
    t = UI[lang]
    return f"""
<div class="footer">
  {t['footer']}<br/>
  <a href="https://huggingface.co/Rashidbm/saudi-date-classifier" target="_blank" rel="noopener">
    {t['footer_link']}
  </a>
</div>
"""


def tsne_header_html(lang: str) -> str:
    t = UI[lang]
    return f"""
<div style="text-align:center; padding: 1.75rem 1rem 0.75rem;">
  <div class="eyebrow" style="text-align:center;">{t['tsne_eyebrow']}</div>
  <h2 class="section-title">{t['tsne_title']}</h2>
  <p class="section-subtitle">{t['tsne_subtitle']}</p>
</div>
"""


def eyebrow_html(text: str) -> str:
    return f'<div class="eyebrow">{text}</div>'


def empty_prediction_html(lang: str) -> str:
    t = UI[lang]
    return f"""
<div class="prediction-card prediction-card--empty">
  <div class="prediction-card__label">{t['empty_pred_label']}</div>
  <div class="prediction-card__value">&mdash;</div>
  <div class="prediction-card__confidence">{t['empty_pred_hint']}</div>
</div>
"""


def empty_heritage_md(lang: str) -> str:
    t = UI[lang]
    return f"### {t['empty_heritage_title']}\n\n{t['empty_heritage_body']}\n"


def gradcam_caption_html(lang: str) -> str:
    return f'<div class="gradcam-caption">{UI[lang]["gradcam_caption"]}</div>'


# ---------------------------- Prediction ----------------------------

def format_prediction_html(lang: str, variety: str, confidence: float, arabic: str = "") -> str:
    t = UI[lang]
    bar_width = max(0.0, min(confidence, 100.0))
    arabic_html = (
        f'<div class="prediction-card__arabic">{arabic}</div>' if arabic else ""
    )
    return f"""
<div class="prediction-card">
  <div class="prediction-card__label">{t['empty_pred_label']}</div>
  <div class="prediction-card__value">{variety}</div>
  {arabic_html}
  <div class="prediction-card__confidence"><strong>{confidence:.1f}%</strong> {t['pred_confidence']}</div>
  <div class="prediction-card__bar">
    <div class="prediction-card__bar-fill" style="width: {bar_width:.1f}%"></div>
  </div>
</div>
"""


def format_heritage_md(lang: str, variety: str, h: dict) -> str:
    t = UI[lang]
    arabic = h.get("arabic", "")
    title = f"### {variety}" + (f" &mdash; {arabic}" if arabic else "")
    return f"""{title}

**{t['heritage_region']}:** {h.get('region', 'N/A')}

**{t['heritage_desc']}:** {h.get('description', 'N/A')}

**{t['heritage_flavor']}:** {h.get('flavor', 'N/A')}

**{t['heritage_sig']}:** {h.get('significance', 'N/A')}
"""


def compute_gradcam(input_tensor: torch.Tensor, original_rgb: np.ndarray) -> np.ndarray:
    grayscale_cam = gradcam(input_tensor=input_tensor, targets=None)[0]
    resized_rgb = cv2.resize(original_rgb, (224, 224)).astype(np.float32) / 255.0
    return show_cam_on_image(resized_rgb, grayscale_cam, use_rgb=True)


def predict(image: np.ndarray, model_choice: str, lang: str):
    if image is None:
        empty = np.zeros((224, 224, 3), dtype=np.uint8)
        return {}, empty_prediction_html(lang), empty_heritage_md(lang), empty

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        if "Ensemble" in model_choice or "مجمّع" in model_choice:
            probs_sum = torch.zeros(9).to(device)
            for m in models_dict.values():
                probs_sum += F.softmax(m(input_tensor), dim=1)[0]
            probs = probs_sum / len(models_dict)
        elif "ViT" in model_choice or "محوّل" in model_choice:
            probs = F.softmax(models_dict["vit"](input_tensor), dim=1)[0]
        elif "EfficientNet" in model_choice:
            probs = F.softmax(models_dict["efficientnet"](input_tensor), dim=1)[0]
        else:
            probs = F.softmax(models_dict["resnet"](input_tensor), dim=1)[0]

    confidences = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(9)}

    top_idx = probs.argmax().item()
    top_variety = CLASS_NAMES[top_idx]
    top_conf = probs[top_idx].item() * 100

    h = HERITAGE_INFO.get(top_variety, {})
    pred_html = format_prediction_html(lang, top_variety, top_conf, h.get("arabic", ""))
    heritage_md = format_heritage_md(lang, top_variety, h)

    try:
        cam_image = compute_gradcam(input_tensor, image)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        cam_image = cv2.resize(image, (224, 224))

    return confidences, pred_html, heritage_md, cam_image


# ---------------------------- Model choices per language ----------------------------

def model_choices(lang: str) -> list[str]:
    if lang == "ar":
        return [
            "النموذج المجمّع (النماذج الثلاثة)",
            "محوّل الرؤية (ViT)",
            "EfficientNet-B0",
            "ResNet-50",
        ]
    return [
        "Ensemble (All 3 Models)",
        "ViT (Vision Transformer)",
        "EfficientNet (EfficientNet-B0)",
        "ResNet (ResNet-50)",
    ]


# ---------------------------- App ----------------------------

def build_app():
    tsne_path = Path("results/tsne.png")
    tsne_exists = tsne_path.exists()

    with gr.Blocks(title="Saudi Date Fruit Classifier", elem_classes=["lang-en"]) as app:
        # Language state
        lang_state = gr.State(value="en")

        # Language switcher (top right)
        with gr.Row(elem_classes=["lang-switcher"]):
            with gr.Column(scale=4):
                pass
            with gr.Column(scale=1, min_width=160):
                lang_radio = gr.Radio(
                    choices=[("English", "en"), ("العربية", "ar")],
                    value="en",
                    label="",
                    show_label=False,
                    container=False,
                )

        hero = gr.HTML(hero_html("en"))

        with gr.Tabs() as tabs:
            with gr.Tab(UI["en"]["tab_classify"]) as tab_classify:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        step1_eyebrow = gr.HTML(eyebrow_html(UI["en"]["step1"]))
                        image_input = gr.Image(
                            label=UI["en"]["image_label"],
                            type="numpy",
                            sources=["upload", "webcam"],
                            height=360,
                            show_label=False,
                        )
                        model_dropdown = gr.Dropdown(
                            choices=model_choices("en"),
                            value=model_choices("en")[0],
                            label=UI["en"]["model_label"],
                            info=UI["en"]["model_info"],
                        )
                        predict_btn = gr.Button(
                            UI["en"]["classify_btn"],
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=6, min_width=320):
                        step2_eyebrow = gr.HTML(eyebrow_html(UI["en"]["step2"]))
                        prediction_html = gr.HTML(value=empty_prediction_html("en"))
                        confidence_output = gr.Label(
                            label=UI["en"]["top_preds"],
                            num_top_classes=5,
                            show_label=True,
                        )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        gradcam_eyebrow = gr.HTML(eyebrow_html(UI["en"]["gradcam_eyebrow"]))
                        gradcam_output = gr.Image(
                            label="Grad-CAM",
                            height=320,
                            show_label=False,
                        )
                        gradcam_caption = gr.HTML(gradcam_caption_html("en"))

                    with gr.Column(scale=1, min_width=320):
                        heritage_eyebrow = gr.HTML(eyebrow_html(UI["en"]["heritage_eyebrow"]))
                        heritage_output = gr.Markdown(
                            value=empty_heritage_md("en"),
                            elem_classes=["heritage-wrap"],
                        )

            with gr.Tab(UI["en"]["tab_features"]) as tab_features:
                tsne_header = gr.HTML(tsne_header_html("en"))
                if tsne_exists:
                    gr.Image(
                        value=str(tsne_path),
                        show_label=False,
                        height=560,
                        container=False,
                    )
                else:
                    tsne_missing_md = gr.Markdown(UI["en"]["tsne_missing"])

        footer = gr.HTML(footer_html("en"))

        # Prediction events
        event_outputs = [confidence_output, prediction_html, heritage_output, gradcam_output]
        predict_btn.click(fn=predict, inputs=[image_input, model_dropdown, lang_state], outputs=event_outputs)
        image_input.change(fn=predict, inputs=[image_input, model_dropdown, lang_state], outputs=event_outputs)
        model_dropdown.change(fn=predict, inputs=[image_input, model_dropdown, lang_state], outputs=event_outputs)

        # Language switching
        def on_lang_change(lang, current_image, current_model):
            t = UI[lang]
            choices = model_choices(lang)
            new_model = choices[0]  # reset to ensemble
            # Re-run prediction if there's an image
            if current_image is not None:
                conf, pred_h, herit_md, cam = predict(current_image, new_model, lang)
            else:
                conf = {}
                pred_h = empty_prediction_html(lang)
                herit_md = empty_heritage_md(lang)
                cam = np.zeros((224, 224, 3), dtype=np.uint8)

            return (
                lang,  # state
                hero_html(lang),  # hero
                eyebrow_html(t["step1"]),  # step1 eyebrow
                gr.update(choices=choices, value=new_model, label=t["model_label"], info=t["model_info"]),  # dropdown
                gr.update(value=t["classify_btn"]),  # button
                eyebrow_html(t["step2"]),  # step2 eyebrow
                gr.update(label=t["top_preds"]),  # confidence label
                eyebrow_html(t["gradcam_eyebrow"]),  # gradcam eyebrow
                gradcam_caption_html(lang),  # gradcam caption
                eyebrow_html(t["heritage_eyebrow"]),  # heritage eyebrow
                tsne_header_html(lang),  # tsne header
                footer_html(lang),  # footer
                conf,  # confidence output
                pred_h,  # prediction html
                herit_md,  # heritage markdown
                cam,  # gradcam image
            )

        lang_radio.change(
            fn=on_lang_change,
            inputs=[lang_radio, image_input, model_dropdown],
            outputs=[
                lang_state,
                hero,
                step1_eyebrow,
                model_dropdown,
                predict_btn,
                step2_eyebrow,
                confidence_output,
                gradcam_eyebrow,
                gradcam_caption,
                heritage_eyebrow,
                tsne_header,
                footer,
                confidence_output,
                prediction_html,
                heritage_output,
                gradcam_output,
            ],
            js=LANG_JS,
        )

    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        theme=THEME,
        css=CUSTOM_CSS,
        allowed_paths=["static/fonts", "static"],
    )
