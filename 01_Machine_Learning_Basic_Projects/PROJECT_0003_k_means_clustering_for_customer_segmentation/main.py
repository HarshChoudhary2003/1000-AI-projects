"""
Project 003: K-Means Clustering for Customer Segmentation
=========================================================
A clean, modular implementation of customer segmentation using K-Means clustering.
Demonstrates realistic customer behavioral data generation, standard scaling,
optimal cluster evaluation (Inertia & Silhouette Score), and business persona assignment.


"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def generate_customer_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic customer demographic and spending data.
    Simulates 5 natural retail customer archetypes:
      1. Low Income, Low Spending (Frugal/Budget)
      2. Low Income, High Spending (Careless/Trendsetters)
      3. Middle Income, Moderate Spending (Average/Mainstream)
      4. High Income, Low Spending (Calculative/Savers)
      5. High Income, High Spending (VIP/High Rollers)
    """
    np.random.seed(random_state)
    
    # Define cluster centers: [Age, Annual Income ($k), Spending Score (1-100)]
    centers = [
        [45, 25, 20],   # Budget Conscious
        [24, 28, 78],   # Young Trendsetters
        [38, 55, 50],   # Mainstream Standard
        [42, 88, 18],   # Affluent Savers
        [32, 85, 82]    # VIP Enthusiasts
    ]
    
    per_cluster = n_samples // len(centers)
    data = []
    
    for cluster_id, center in enumerate(centers):
        age = np.random.normal(loc=center[0], scale=5, size=per_cluster).clip(18, 70)
        income = np.random.normal(loc=center[1], scale=7, size=per_cluster).clip(15, 140)
        spending = np.random.normal(loc=center[2], scale=8, size=per_cluster).clip(1, 99)
        gender = np.random.choice(["Male", "Female"], size=per_cluster, p=[0.48, 0.52])
        
        for a, inc, sp, gen in zip(age, income, spending, gender):
            data.append({
                "Age": int(round(a)),
                "Annual_Income_k$": round(float(inc), 1),
                "Spending_Score": int(round(sp)),
                "Gender": gen
            })
            
    df = pd.DataFrame(data)
    df.insert(0, "Customer_ID", [f"CUST-{1000 + i}" for i in range(len(df))])
    return df


def evaluate_k_values(X_scaled: np.ndarray, k_range=range(2, 9)):
    """Compute inertia (WCSS) and silhouette scores across multiple K values."""
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        
    return list(k_range), inertias, silhouettes


def profile_segments(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Summarize demographic and behavioral attributes per customer cluster."""
    summary = df.groupby(cluster_col).agg(
        Customer_Count=('Customer_ID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income_k=('Annual_Income_k$', 'mean'),
        Avg_Spending_Score=('Spending_Score', 'mean')
    ).round(1)
    
    # Calculate percentage distribution
    summary["Percentage"] = ((summary["Customer_Count"] / len(df)) * 100).round(1)
    
    # Assign intuitive marketing personas based on income and spending score
    personas = []
    for _, row in summary.iterrows():
        inc = row["Avg_Income_k"]
        sp = row["Avg_Spending_Score"]
        
        if inc >= 70 and sp >= 60:
            personas.append("VIP High Rollers")
        elif inc >= 70 and sp < 40:
            personas.append("Affluent Savers")
        elif inc < 45 and sp >= 60:
            personas.append("Young Trendsetters")
        elif inc < 45 and sp < 40:
            personas.append("Budget Conscious")
        else:
            personas.append("Mainstream Core")
            
    summary["Persona"] = personas
    return summary


def main():
    print("=" * 60)
    print("[*] Project 003: Customer Segmentation Pipeline (K-Means)")
    print("=" * 60)
    
    # 1. Generate realistic synthetic customer data
    df = generate_customer_dataset(n_samples=500, random_state=42)
    print(f"\n[+] Created customer dataset with {len(df)} records.")
    print("    Sample preview:")
    print(df.head(3).to_string(index=False))
    
    # 2. Select numerical features for clustering
    feature_cols = ["Annual_Income_k$", "Spending_Score"]
    X = df[feature_cols].values
    
    # 3. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Find optimal K using silhouette scores
    k_range, inertias, silhouettes = evaluate_k_values(X_scaled, range(2, 8))
    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\n[+] Evaluated K from 2 to 7.")
    print(f"    Highest Silhouette Score ({max(silhouettes):.4f}) observed at K = {best_k}")
    
    # 5. Fit model with chosen K (5 clusters matches our archetypes)
    k = 5
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, df["Cluster"])
    
    print(f"\n[+] Trained K-Means model with K={k}")
    print(f"    Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"    Silhouette Score: {final_sil:.4f}")
    
    # 6. Profile segments
    summary = profile_segments(df, "Cluster")
    print("\n[+] Customer Segments Overview:")
    print(summary.to_string())
    
    print("\n[Tip] Launch the interactive Streamlit UI by running:")
    print("      streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
