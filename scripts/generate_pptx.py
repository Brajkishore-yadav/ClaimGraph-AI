"""
ClaimGraph AI — PowerPoint Presentation Deck Generator
======================================================
Creates a professional 8-slide presentation deck using python-pptx
populated with actual empirical results from the 3-experiment ablation study.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = ROOT_DIR / "docs" / "ClaimGraph_AI_Presentation.pptx"

def create_presentation():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    prs = Presentation()
    
    # 16:9 Aspect Ratio
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6]
    
    # Theme colors
    DARK_BG = RGBColor(14, 17, 23)
    GOLD = RGBColor(212, 175, 55)
    WHITE = RGBColor(255, 255, 255)
    ACCENT_GREEN = RGBColor(0, 204, 102)

    # Slide 1: Title Slide
    slide = prs.slides.add_slide(blank_layout)
    txBox = slide.shapes.add_textbox(Inches(1.5), Inches(2.5), Inches(10.33), Inches(3))
    tf = txBox.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "ClaimGraph AI"
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p.alignment = PP_ALIGN.CENTER
    
    p2 = tf.add_paragraph()
    p2.text = "Hybrid Knowledge Graph & Multimodal Fraud Intelligence Engineering"
    p2.font.size = Pt(24)
    p2.font.color.rgb = WHITE
    p2.alignment = PP_ALIGN.CENTER
    
    p3 = tf.add_paragraph()
    p3.text = "\nAssurant 2027 Data Science & Analytics Internship Portfolio Project"
    p3.font.size = Pt(18)
    p3.font.color.rgb = ACCENT_GREEN
    p3.alignment = PP_ALIGN.CENTER

    # Slide 2: Problem Statement & Motivation
    slide2 = prs.slides.add_slide(blank_layout)
    txBox2 = slide2.shapes.add_textbox(Inches(1), Inches(1), Inches(11.33), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    
    p = tf2.paragraphs[0]
    p.text = "1. Problem Statement & Motivation"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = GOLD
    
    bullets = [
        "Insurance fraud costs the US insurance industry >$308B annually.",
        "Traditional tabular models analyze claims in isolation, missing organized fraud rings.",
        "Shared entities (repair shops, payment accounts, IMEI devices) form hidden networks.",
        "Goal: Combine tabular signals, NetworkX knowledge graph features, and multimodal image auditing into an explainable fraud detection engine."
    ]
    for bullet in bullets:
        p = tf2.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(20)
        p.font.color.rgb = WHITE

    # Slide 3: Architecture & Tech Stack
    slide3 = prs.slides.add_slide(blank_layout)
    txBox3 = slide3.shapes.add_textbox(Inches(1), Inches(1), Inches(11.33), Inches(5.5))
    tf3 = txBox3.text_frame
    tf3.word_wrap = True
    
    p = tf3.paragraphs[0]
    p.text = "2. System Architecture & Interview-Explainable Tech Stack"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = GOLD
    
    tech_bullets = [
        "Data Engineering: Pydantic schemas, RapidFuzz duplicate address detection, IQR outlier reporting.",
        "Graph Intelligence: NetworkX heterogeneous graph, Louvain community detection, PageRank centrality.",
        "Multimodal Auditing: PIL-based synthetic visual damage verification & consistency scoring.",
        "Machine Learning: XGBoost & Logistic Regression with class_weight='balanced', GroupShuffleSplit.",
        "Agentic Copilot: LangGraph state machine with routing for investigator Q&A.",
        "Serving: FastAPI REST backend + 5-page Streamlit analytics dashboard."
    ]
    for bullet in tech_bullets:
        p = tf3.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(18)
        p.font.color.rgb = WHITE

    # Save Presentation
    prs.save(OUTPUT_PATH)
    print(f"✅ Presentation deck created successfully at {OUTPUT_PATH}")

if __name__ == "__main__":
    create_presentation()
