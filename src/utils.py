"""
Utility helpers: seeding, device selection, logging, config loading.
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger("date-fruit")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Heritage knowledge base for Saudi date varieties.
# season_start / season_end are month numbers (1..12). peak_month is the strongest month.
# distinguish_* describes visual cues to identify the variety at a glance.
HERITAGE_INFO = {
    "Ajwa": {
        "arabic": "عجوة",
        "region": "المدينة المنورة (Madinah)",
        "description": "One of the most prized dates in Saudi Arabia, known for its dark color and soft texture.",
        "description_ar": "من أثمن تمور المملكة، يتميّز بلونه الداكن وقوامه الليّن.",
        "significance": "Mentioned in Hadith. Considered sacred and highly valued across the Muslim world.",
        "significance_ar": "ورد ذكره في الحديث الشريف، ويحظى بمكانة مقدّسة عند المسلمين.",
        "flavor": "Rich, sweet with subtle caramel and prune notes.",
        "flavor_ar": "حلاوة غنيّة مع لمحة كراميل وبرقوق.",
        "season_start": 8, "season_end": 10, "peak_month": 9,
        "distinguish": "Small, matte black-brown surface with fine wrinkles. Very soft.",
        "distinguish_ar": "صغيرة، سطحها أسود مائل للبنّي مع تجاعيد ناعمة. ليّنة جدًا.",
    },
    "Galaxy": {
        "arabic": "قلاكسي",
        "region": "المملكة العربية السعودية (Saudi Arabia)",
        "description": "A premium commercial variety known for its large size and golden-brown color.",
        "description_ar": "صنف تجاري متميّز بحجمه الكبير ولونه الذهبي البنّي.",
        "significance": "Popular modern variety widely exported from Saudi Arabia.",
        "significance_ar": "صنف حديث مشهور يُصدَّر من السعودية بكثرة.",
        "flavor": "Sweet, chewy with honey-like undertones.",
        "flavor_ar": "حلاوة مطاطيّة بلمسة عسل.",
        "season_start": 8, "season_end": 10, "peak_month": 9,
        "distinguish": "Large, glossy, amber-golden skin. Elongated with a pointed tip.",
        "distinguish_ar": "كبيرة، قشرتها ذهبيّة لامعة، مستطيلة مدبّبة الأطراف.",
    },
    "Medjool": {
        "arabic": "مجدول",
        "region": "المدينة المنورة (Madinah) / الرياض (Riyadh)",
        "description": "Known as the 'King of Dates' for its large size and rich taste.",
        "description_ar": "يُلقَّب بـ«ملك التمور» لحجمه الكبير ومذاقه الفاخر.",
        "significance": "One of the most recognized date varieties worldwide. Historically reserved for royalty.",
        "significance_ar": "من أشهر التمور عالميًا، وكان قديمًا حكرًا على الملوك.",
        "flavor": "Caramel-like sweetness, soft and creamy texture.",
        "flavor_ar": "حلاوة الكراميل، بقوام ليّن كريمي.",
        "season_start": 9, "season_end": 11, "peak_month": 10,
        "distinguish": "Very large, plump, wrinkled amber-brown skin. Tender and moist.",
        "distinguish_ar": "كبيرة جدًا، ممتلئة، قشرتها بنّية كهرمانيّة متجعّدة. رطبة طريّة.",
    },
    "Meneifi": {
        "arabic": "منيفي",
        "region": "القصيم (Qassim) / المدينة المنورة (Madinah)",
        "description": "A traditional Saudi variety with elongated shape and amber color.",
        "description_ar": "صنف سعودي أصيل، شكله مستطيل ولونه كهرماني.",
        "significance": "Well-known in local Saudi markets, a household favorite.",
        "significance_ar": "معروف في الأسواق السعودية، ومفضَّل في البيوت.",
        "flavor": "Moderately sweet with a firm, slightly chewy texture.",
        "flavor_ar": "حلاوته متوسّطة، قوامه متماسك مائل للمضغ.",
        "season_start": 8, "season_end": 10, "peak_month": 9,
        "distinguish": "Elongated, narrow, light amber body. Firmer than Medjool.",
        "distinguish_ar": "مستطيلة نحيلة، لونها كهرماني فاتح. أكثر صلابة من المجدول.",
    },
    "Nabtat Ali": {
        "arabic": "نبتة علي",
        "region": "المدينة المنورة (Madinah)",
        "description": "A Madinah variety with distinctive reddish-brown color.",
        "description_ar": "صنف مدني يتميّز بلونه البنّي المائل للاحمرار.",
        "significance": "Named variety from the date palm farms of Madinah.",
        "significance_ar": "من أشهر أصناف نخيل المدينة المنوّرة.",
        "flavor": "Sweet and tender with a moist, soft flesh.",
        "flavor_ar": "حلوة طريّة، لحمها رطب ليّن.",
        "season_start": 9, "season_end": 10, "peak_month": 9,
        "distinguish": "Medium size, reddish-brown tone, slightly wrinkled.",
        "distinguish_ar": "متوسّطة الحجم، لونها بنّي محمرّ، بتجاعيد خفيفة.",
    },
    "Rutab": {
        "arabic": "رطب",
        "region": "جميع مناطق المملكة (All regions)",
        "description": "Refers to the soft, ripe stage of the date fruit before full drying.",
        "description_ar": "المرحلة الطريّة الناضجة من ثمرة النخلة قبل الجفاف الكامل.",
        "significance": "Deeply embedded in Saudi culture. Fresh rutab is a seasonal delicacy during harvest.",
        "significance_ar": "جزء أصيل من الثقافة السعودية. الرطب الطازج أكلة موسمية في الجني.",
        "flavor": "Extremely soft, juicy, and intensely sweet.",
        "flavor_ar": "ليّنة جدًا، عصيريّة، شديدة الحلاوة.",
        "season_start": 7, "season_end": 9, "peak_month": 8,
        "distinguish": "Glossy, half-amber half-brown transition skin. Very moist.",
        "distinguish_ar": "قشرتها لامعة، نصف كهرمانيّة ونصف بنّيّة. رطبة جدًا.",
    },
    "Shaishe": {
        "arabic": "شيشي",
        "region": "المدينة المنورة (Madinah)",
        "description": "A Madinah-specific variety with small to medium size.",
        "description_ar": "صنف مدني خاص، صغير إلى متوسّط الحجم.",
        "significance": "Traditional variety cultivated in Madinah palm groves for generations.",
        "significance_ar": "صنف تقليدي يُزرع في نخيل المدينة لأجيال.",
        "flavor": "Balanced sweetness with a smooth, soft texture.",
        "flavor_ar": "حلاوة متوازنة وقوام ناعم.",
        "season_start": 9, "season_end": 10, "peak_month": 9,
        "distinguish": "Small, smooth, dark amber. Uniform shape.",
        "distinguish_ar": "صغيرة ناعمة القشرة، لونها كهرماني داكن. شكلها منتظم.",
    },
    "Sokari": {
        "arabic": "سكري",
        "region": "القصيم (Qassim)",
        "description": "The most popular date variety in Saudi Arabia, known for its golden color and sugar-like sweetness.",
        "description_ar": "أشهر تمور السعودية، يتميّز بلونه الذهبي وحلاوته السكريّة.",
        "significance": "Named 'Sokari' (sugary) for its exceptional sweetness. A staple in every Saudi home.",
        "significance_ar": "سُمِّي «سكري» لحلاوته الاستثنائيّة. حاضر في كلّ بيت سعودي.",
        "flavor": "Crisp outer layer with a soft, sweet interior. Tastes like caramel candy.",
        "flavor_ar": "قشرة مقرمشة ولبّ طريّ حلو، طعمه كحلوى الكراميل.",
        "season_start": 7, "season_end": 9, "peak_month": 8,
        "distinguish": "Two-tone: golden tip with brown base. Crunchy skin, soft inside.",
        "distinguish_ar": "بلونين: ذهبي في الطرف وبنّي في القاعدة. قشرة مقرمشة ولبّ طريّ.",
    },
    "Sugaey": {
        "arabic": "صقعي",
        "region": "القصيم (Qassim) / الرياض (Riyadh)",
        "description": "A popular variety often eaten at the Khalal (yellow) or Rutab stage.",
        "description_ar": "صنف شهير يؤكل غالبًا في مرحلة الخلال (الأصفر) أو الرطب.",
        "significance": "Widely consumed during Ramadan. A favorite for stuffing with nuts.",
        "significance_ar": "يُستهلك بكثرة في رمضان، ومفضَّل للحشو بالمكسّرات.",
        "flavor": "Mildly sweet, firm texture, often enjoyed fresh.",
        "flavor_ar": "حلاوة خفيفة وقوام متماسك، يُؤكل طازجًا غالبًا.",
        "season_start": 7, "season_end": 9, "peak_month": 8,
        "distinguish": "Oblong, yellow-to-amber gradient. Firm, dry-looking skin.",
        "distinguish_ar": "مستطيلة، لونها يتدرّج من الأصفر إلى الكهرماني. قشرتها متماسكة وشبه جافّة.",
    },
}
