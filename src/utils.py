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


# Heritage knowledge base for Saudi date varieties
HERITAGE_INFO = {
    "Ajwa": {
        "arabic": "عجوة",
        "region": "المدينة المنورة (Madinah)",
        "description": "One of the most prized dates in Saudi Arabia, known for its dark color and soft texture.",
        "significance": "Mentioned in Hadith. Considered sacred and highly valued across the Muslim world.",
        "flavor": "Rich, sweet with subtle caramel and prune notes.",
    },
    "Galaxy": {
        "arabic": "قلاكسي",
        "region": "المملكة العربية السعودية (Saudi Arabia)",
        "description": "A premium commercial variety known for its large size and golden-brown color.",
        "significance": "Popular modern variety widely exported from Saudi Arabia.",
        "flavor": "Sweet, chewy with honey-like undertones.",
    },
    "Medjool": {
        "arabic": "مجدول",
        "region": "المدينة المنورة (Madinah) / الرياض (Riyadh)",
        "description": "Known as the 'King of Dates' for its large size and rich taste.",
        "significance": "One of the most recognized date varieties worldwide. Historically reserved for royalty.",
        "flavor": "Caramel-like sweetness, soft and creamy texture.",
    },
    "Meneifi": {
        "arabic": "منيفي",
        "region": "القصيم (Qassim) / المدينة المنورة (Madinah)",
        "description": "A traditional Saudi variety with elongated shape and amber color.",
        "significance": "Well-known in local Saudi markets, a household favorite.",
        "flavor": "Moderately sweet with a firm, slightly chewy texture.",
    },
    "Nabtat Ali": {
        "arabic": "نبتة علي",
        "region": "المدينة المنورة (Madinah)",
        "description": "A Madinah variety with distinctive reddish-brown color.",
        "significance": "Named variety from the date palm farms of Madinah.",
        "flavor": "Sweet and tender with a moist, soft flesh.",
    },
    "Rutab": {
        "arabic": "رطب",
        "region": "جميع مناطق المملكة (All regions)",
        "description": "Refers to the soft, ripe stage of the date fruit before full drying.",
        "significance": "Deeply embedded in Saudi culture. Fresh rutab is a seasonal delicacy during harvest.",
        "flavor": "Extremely soft, juicy, and intensely sweet.",
    },
    "Shaishe": {
        "arabic": "شيشي",
        "region": "المدينة المنورة (Madinah)",
        "description": "A Madinah-specific variety with small to medium size.",
        "significance": "Traditional variety cultivated in Madinah palm groves for generations.",
        "flavor": "Balanced sweetness with a smooth, soft texture.",
    },
    "Sokari": {
        "arabic": "سكري",
        "region": "القصيم (Qassim)",
        "description": "The most popular date variety in Saudi Arabia, known for its golden color and sugar-like sweetness.",
        "significance": "Named 'Sokari' (sugary) for its exceptional sweetness. A staple in every Saudi home.",
        "flavor": "Crisp outer layer with a soft, sweet interior. Tastes like caramel candy.",
    },
    "Sugaey": {
        "arabic": "صقعي",
        "region": "القصيم (Qassim) / الرياض (Riyadh)",
        "description": "A popular variety often eaten at the Khalal (yellow) or Rutab stage.",
        "significance": "Widely consumed during Ramadan. A favorite for stuffing with nuts.",
        "flavor": "Mildly sweet, firm texture, often enjoyed fresh.",
    },
}
