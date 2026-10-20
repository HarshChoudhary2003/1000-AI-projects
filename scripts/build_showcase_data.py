#!/usr/bin/env python3
"""
Showcase Data Generator for 1,000 AI Projects
============================================
Scans all 25 category directories, extracts project metadata,
verifies assets (main.py, notebook.ipynb, app.py, requirements.txt),
and outputs an optimized `projects.json` for the Vercel Showcase Web App.
"""

import os
import re
import json
import time

REPO_OWNER = "HarshChoudhary2003"
REPO_NAME = "1000-AI-projects"
GITHUB_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/main"
COLAB_BASE = f"https://colab.research.google.com/github/{REPO_OWNER}/{REPO_NAME}/blob/main"

CATEGORY_COLORS = {
    1: "#F59E0B",   # ML Basics - Warm Amber Gold
    2: "#D97706",   # ML Intermediate - Deep Ochre
    3: "#3B82F6",   # DL Basics - Sapphire Blue
    4: "#2563EB",   # DL Intermediate - Cobalt
    5: "#0D9488",   # NLP - Sage Teal
    6: "#059669",   # Computer Vision - Emerald
    7: "#C2410C",   # Reinforcement Learning - Terracotta
    8: "#B45309",   # Time Series - Bronze
    9: "#4F46E5",   # Recommendation Systems - Slate Indigo
    10: "#7C3AED",  # Generative Models - Deep Amethyst
    11: "#0891B2",  # GNNs - Nordic Cyan
    12: "#E11D48",  # Healthcare - Crimson Coral
    13: "#15803D",  # Finance - British Racing Green
    14: "#0F766E",  # NLP Advanced - Deep Teal
    15: "#0284C7",  # CV Advanced - Slate Azure
    16: "#EA580C",  # RL Advanced - Warm Rust
    17: "#64748B",  # Robotics - Titanium Slate
    18: "#BE123C",  # Audio/Speech - Burgundy Rose
    19: "#6D28D9",  # Explainable AI - Imperial Violet
    20: "#047857",  # Edge AI - Forest Green
    21: "#1D4ED8",  # Business - Royal Cobalt
    22: "#16A34A",  # Environmental - Natural Green
    23: "#B91C1C",  # Security - Deep Carmine
    24: "#7E22CE",  # Multi-modal - Royal Purple
    25: "#374151",  # Meta & Hybrid - Graphite Metal
}

CATEGORY_ICONS = {
    1: "📊", 2: "⚡", 3: "🧠", 4: "🔮", 5: "💬",
    6: "👁️", 7: "🎮", 8: "📈", 9: "🛍️", 10: "🎨",
    11: "🕸️", 12: "🏥", 13: "💹", 14: "📚", 15: "🚀",
    16: "🤖", 17: "🦾", 18: "🎙️", 19: "💡", 20: "📱",
    21: "🏢", 22: "🌱", 23: "🔒", 24: "🎭", 25: "🧬"
}

def clean_title(folder_name):
    parts = folder_name.split('_')
    words = parts[2:]
    title = ' '.join(w.capitalize() for w in words)
    # Acronyms preservation
    acronyms = {"Nlp": "NLP", "Cnn": "CNN", "Rnn": "RNN", "Lstm": "LSTM", "Gan": "GAN", "Gnn": "GNN",
                "Ocr": "OCR", "Pca": "PCA", "Knn": "KNN", "Svm": "SVM", "Arima": "ARIMA", "Api": "API",
                "Bert": "BERT", "Ai": "AI", "Ml": "ML", "Xor": "XOR", "Mnist": "MNIST", "Yolo": "YOLO"}
    cleaned_words = [acronyms.get(w, w) for w in title.split()]
    return ' '.join(cleaned_words)

def build_index(root_dir=".", output_path="showcase/projects.json"):
    start = time.time()
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d[0].isdigit()])
    
    projects = []
    category_summary = []
    
    for cat in categories:
        cat_path = os.path.join(root_dir, cat)
        cat_num = int(cat[:2])
        cat_friendly = cat[3:].replace('_', ' ')
        color = CATEGORY_COLORS.get(cat_num, "#6366F1")
        icon = CATEGORY_ICONS.get(cat_num, "📁")
        
        subdirs = sorted([p for p in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, p)) and p.startswith('PROJECT_')])
        
        cat_app_count = 0
        for proj in subdirs:
            proj_path = os.path.join(cat_path, proj)
            m_num = re.search(r'PROJECT_(\d+)', proj)
            p_num = int(m_num.group(1)) if m_num else len(projects) + 1
            
            title = clean_title(proj)
            files = os.listdir(proj_path)
            has_app = 'app.py' in files
            has_script = 'main.py' in files
            has_notebook = 'notebook.ipynb' in files
            has_reqs = 'requirements.txt' in files
            
            if has_app:
                cat_app_count += 1
                
            diff = "Beginner" if cat_num <= 3 else ("Intermediate" if cat_num <= 13 else "Advanced")
            overview = f"Production-grade standalone implementation of {title} in {cat_friendly}."
            tags = [cat_friendly.split()[0], "AI", "Machine-Learning"]
            
            readme_file = os.path.join(proj_path, 'README.md')
            if os.path.exists(readme_file):
                try:
                    with open(readme_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(2000)
                    m_diff = re.search(r'\*\*Difficulty:\*\*\s*(.+)', content)
                    if m_diff:
                        diff = m_diff.group(1).strip()
                    m_tags = re.search(r'\*\*Tags:\*\*\s*(.+)', content)
                    if m_tags:
                        tags = [t.strip('` ').capitalize() for t in m_tags.group(1).split(',') if t.strip()]
                    m_over = re.search(r'##\s*📖\s*Project Overview\s*\n+([^#\n]+)', content)
                    if m_over:
                        raw_over = ' '.join(m_over.group(1).strip().split())
                        overview = (raw_over[:200] + '...') if len(raw_over) > 200 else raw_over
                except Exception:
                    pass
            
            rel_path = f"{cat}/{proj}"
            projects.append({
                "id": proj,
                "num": p_num,
                "title": title,
                "cat_id": cat_num,
                "cat_name": cat_friendly,
                "cat_icon": icon,
                "cat_color": color,
                "diff": diff,
                "tags": tags[:6],
                "desc": overview,
                "has_app": has_app,
                "has_script": has_script,
                "has_notebook": has_notebook,
                "has_reqs": has_reqs,
                "path": rel_path,
                "github_url": f"{GITHUB_BASE}/{rel_path}",
                "colab_url": f"{COLAB_BASE}/{rel_path}/notebook.ipynb"
            })
            
        category_summary.append({
            "id": cat_num,
            "folder": cat,
            "name": cat_friendly,
            "icon": icon,
            "color": color,
            "count": len(subdirs),
            "apps_count": cat_app_count
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    payload = {
        "metadata": {
            "total_projects": len(projects),
            "total_categories": len(category_summary),
            "total_interactive_apps": sum(1 for p in projects if p["has_app"]),
            "repo_owner": REPO_OWNER,
            "repo_name": REPO_NAME,
            "repo_url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "categories": category_summary,
        "projects": projects
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(',', ':'))
        
    print(f"[OK] Generated {output_path} with {len(projects)} projects ({time.time()-start:.2f}s)")
    return payload

if __name__ == "__main__":
    build_index()
