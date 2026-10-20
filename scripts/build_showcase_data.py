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
    1: "#3B82F6",   # ML Basics - Blue
    2: "#0284C7",   # ML Intermediate - Sky
    3: "#8B5CF6",   # DL Basics - Purple
    4: "#7C3AED",   # DL Intermediate - Violet
    5: "#EC4899",   # NLP - Pink
    6: "#10B981",   # Computer Vision - Emerald
    7: "#F59E0B",   # Reinforcement Learning - Amber
    8: "#F97316",   # Time Series - Orange
    9: "#6366F1",   # Recommendation - Indigo
    10: "#A855F7",  # Generative Models - Fuchsia
    11: "#14B8A6",  # GNNs - Teal
    12: "#EF4444",  # Healthcare - Red
    13: "#84CC16",  # Finance - Lime
    14: "#D946EF",  # NLP Advanced - Magenta
    15: "#06B6D4",  # CV Advanced - Cyan
    16: "#EAB308",  # RL Advanced - Yellow
    17: "#64748B",  # Robotics - Slate
    18: "#F43F5E",  # Audio/Speech - Rose
    19: "#8B5CF6",  # Explainable AI - Purple
    20: "#0D9488",  # Edge AI - Dark Teal
    21: "#3B82F6",  # Business - Blue
    22: "#22C55E",  # Environmental - Green
    23: "#DC2626",  # Security - Crimson
    24: "#9333EA",  # Multi-modal - Purple
    25: "#4F46E5",  # Meta & Hybrid - Indigo
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
