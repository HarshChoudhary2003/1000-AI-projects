import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .rec-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #818cf8;
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Default data initialization
def get_initial_dataset():
    return pd.DataFrame([
        [5, 4, 0, 0, 1],
        [4, 0, 0, 5, 1],
        [1, 1, 0, 5, 4],
        [0, 0, 5, 4, 5],
        [5, 4, 3, 0, 0]
    ], index=['Alice', 'Bob', 'Charlie', 'Diana', 'Evan'],
       columns=['Avatar', 'Titanic', 'Inception', 'Interstellar', 'Gladiator'])

if "ratings" not in st.session_state:
    st.session_state.ratings = get_initial_dataset()

# Helper recommendation logic
def calculate_user_similarity(df):
    sim = cosine_similarity(df)
    return pd.DataFrame(sim, index=df.index, columns=df.index)

def calculate_item_similarity(df):
    sim = cosine_similarity(df.T)
    return pd.DataFrame(sim, index=df.columns, columns=df.columns)

def recommend_user_based(df, target_user, k_peers=1):
    sim_df = calculate_user_similarity(df)
    similar_users = sim_df[target_user].drop(target_user).sort_values(ascending=False)
    
    unseen_items = df.loc[target_user][df.loc[target_user] == 0].index.tolist()
    if not unseen_items:
        return None, similar_users, "User has already rated all movies!"
    
    top_peer = similar_users.index[0]
    top_peer_sim = similar_users.iloc[0]
    
    # Method 1: Top peer highest rated
    peer_ratings = df.loc[top_peer, unseen_items].sort_values(ascending=False)
    
    # Method 2: Weighted average across all positive similar peers
    weights = similar_users[similar_users > 0]
    predicted_ratings = {}
    for item in unseen_items:
        ratings_for_item = df.loc[weights.index, item]
        rated_mask = ratings_for_item > 0
        if rated_mask.any():
            w = weights[rated_mask]
            r = ratings_for_item[rated_mask]
            predicted_ratings[item] = (w * r).sum() / w.sum()
        else:
            predicted_ratings[item] = 0.0
            
    weighted_series = pd.Series(predicted_ratings).sort_values(ascending=False)
    
    return {
        "top_peer": top_peer,
        "top_peer_sim": top_peer_sim,
        "peer_ratings": peer_ratings,
        "weighted_predictions": weighted_series,
        "similar_users": similar_users
    }

def recommend_item_based(df, target_user):
    item_sim = calculate_item_similarity(df)
    user_ratings = df.loc[target_user]
    seen_items = user_ratings[user_ratings > 0]
    unseen_items = user_ratings[user_ratings == 0].index.tolist()
    
    if not unseen_items:
        return None, "User has already rated all movies!"
    if seen_items.empty:
        return None, "User has no ratings yet to base item similarity on!"
        
    predictions = {}
    for item in unseen_items:
        sims = item_sim.loc[item, seen_items.index]
        weights = sims[sims > 0]
        if not weights.empty:
            ratings = user_ratings[weights.index]
            predictions[item] = (weights * ratings).sum() / weights.sum()
        else:
            predictions[item] = 0.0
            
    return pd.Series(predictions).sort_values(ascending=False), item_sim

# Sidebar Navigation
st.sidebar.markdown("## 🎬 Navigation")
mode = st.sidebar.radio(
    "Choose View:",
    [
        "🎯 User Recommendations",
        "⭐ Try Your Preferences",
        "📊 Similarity & Matrices",
        "✏️ Edit Ratings & Catalog",
        "📖 How It Works"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Actions")
if st.sidebar.button("🔄 Reset Dataset to Default", use_container_width=True):
    st.session_state.ratings = get_initial_dataset()
    st.sidebar.success("Reset to default matrix!")
    st.rerun()

st.sidebar.info(
    "**Project 012**: Simple Collaborative Filtering Recommendation System using **Cosine Similarity**."
)

# ----------------- VIEW 1: USER RECOMMENDATIONS -----------------
if mode == "🎯 User Recommendations":
    st.markdown('<div class="main-title">Collaborative Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore user-based collaborative filtering predictions and taste similarity.</div>', unsafe_allow_html=True)
    
    df = st.session_state.ratings
    
    col_sel, col_algo = st.columns([2, 2])
    with col_sel:
        target_user = st.selectbox("Select Target User:", df.index.tolist(), index=0)
    with col_algo:
        algo_choice = st.radio("Strategy:", ["Top-Peer Matching (Default)", "Weighted Multi-User Consensus"], horizontal=True)
        
    res = recommend_user_based(df, target_user)
    
    if res is None:
        st.warning(f"No unseen movies available for {target_user}.")
    else:
        top_peer = res["top_peer"]
        top_peer_sim = res["top_peer_sim"]
        peer_ratings = res["peer_ratings"]
        weighted_predictions = res["weighted_predictions"]
        similar_users = res["similar_users"]
        
        # User Stats Row
        col1, col2, col3, col4 = st.columns(4)
        user_movies_watched = (df.loc[target_user] > 0).sum()
        unwatched_count = len(df.columns) - user_movies_watched
        
        with col1:
            st.metric("Target User", target_user)
        with col2:
            st.metric("Movies Rated", f"{user_movies_watched} / {len(df.columns)}")
        with col3:
            st.metric("Closest Taste Twin", f"{top_peer}")
        with col4:
            st.metric("Peer Similarity", f"{top_peer_sim:.1%}")
            
        st.markdown("---")
        
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.subheader("🏆 Recommended Movies")
            
            if algo_choice == "Top-Peer Matching (Default)":
                st.caption(f"Ranked by ratings from **{top_peer}** (Cosine Similarity: {top_peer_sim:.3f})")
                for rank, (movie, rating) in enumerate(peer_ratings.items(), 1):
                    star_display = "★" * int(rating) + "☆" * (5 - int(rating))
                    badge_color = "#10b981" if rating >= 4 else "#f59e0b" if rating >= 2 else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background:#1e293b; border-left: 4px solid {badge_color}; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.15rem; font-weight:600; color:#f1f5f9;">#{rank} {movie}</span>
                            <span style="font-size:1rem; color:#facc15;">{star_display} ({rating}/5)</span>
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">
                            Rated <strong>{rating}/5</strong> by {top_peer}. {target_user} hasn't watched this yet.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("Ranked by weighted similarity predictions across all matching users.")
                for rank, (movie, pred_score) in enumerate(weighted_predictions.items(), 1):
                    stars = min(5, max(0, round(pred_score)))
                    star_display = "★" * stars + "☆" * (5 - stars)
                    st.markdown(f"""
                    <div style="background:#1e293b; border-left: 4px solid #6366f1; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.15rem; font-weight:600; color:#f1f5f9;">#{rank} {movie}</span>
                            <span style="font-size:1rem; color:#818cf8; font-weight:bold;">Predicted: {pred_score:.2f} / 5.0</span>
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">
                            Aggregate score weighted by peer similarity scores.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Display Target User's Current Ratings
            with st.expander(f"👀 View Current Ratings by {target_user}"):
                cur_ratings = df.loc[target_user]
                cur_df = pd.DataFrame({
                    "Movie": cur_ratings.index,
                    "Rating": cur_ratings.values,
                    "Status": ["Rated" if r > 0 else "Unseen / Unrated" for r in cur_ratings.values]
                })
                st.dataframe(cur_df, hide_index=True, use_container_width=True)

        with c_right:
            st.subheader(f"👥 Similarity to {target_user}")
            sim_plot_df = similar_users.reset_index()
            sim_plot_df.columns = ["User", "Cosine Similarity"]
            
            fig = px.bar(
                sim_plot_df,
                x="Cosine Similarity",
                y="User",
                orientation="h",
                color="Cosine Similarity",
                color_continuous_scale="Purples",
                text=sim_plot_df["Cosine Similarity"].apply(lambda v: f"{v:.3f}"),
                range_x=[0, 1]
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=20, b=20),
                height=260,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Insight**:  
            **{top_peer}** is the closest match for **{target_user}** with a cosine similarity of **{top_peer_sim:.3f}**. Their taste vectors align closest in multidimensional movie space.
            """)

# ----------------- VIEW 2: TRY YOUR PREFERENCES -----------------
elif mode == "⭐ Try Your Preferences":
    st.markdown('<div class="main-title">Personalized Recommendation Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Rate the movies below (0 = Not Watched) to test how the algorithm recommends movies for you!</div>', unsafe_allow_html=True)
    
    df = st.session_state.ratings
    movies = df.columns.tolist()
    
    col_name, col_algo = st.columns([2, 2])
    with col_name:
        custom_name = st.text_input("Your Name / Persona:", "Alex")
    with col_algo:
        rec_mode = st.radio("Recommendation Method:", ["User-Based (Find taste twins)", "Item-Based (Find similar movies)"], horizontal=True)
        
    st.markdown("#### 🌟 Your Movie Ratings (0 = Unseen / Skip)")
    
    rating_cols = st.columns(len(movies))
    user_inputs = {}
    for i, col in enumerate(rating_cols):
        with col:
            st.markdown(f"**{movies[i]}**")
            user_inputs[movies[i]] = st.slider(
                f"Rate {movies[i]}",
                min_value=0,
                max_value=5,
                value=5 if i == 0 else (4 if i == 1 else 0),
                key=f"custom_rate_{movies[i]}",
                label_visibility="collapsed"
            )
            val = user_inputs[movies[i]]
            if val == 0:
                st.caption("Not seen")
            else:
                st.caption(f"{'★' * val} ({val}/5)")
                
    if st.button("🚀 Calculate My Recommendations", type="primary", use_container_width=True):
        custom_series = pd.Series(user_inputs, name=custom_name)
        
        # Check if user rated at least one movie
        if (custom_series > 0).sum() == 0:
            st.error("Please rate at least one movie (rating >= 1) to generate recommendations!")
        elif (custom_series == 0).sum() == 0:
            st.warning("You have rated all movies! There are no unwatched movies left to recommend.")
        else:
            if rec_mode.startswith("User-Based"):
                temp_df = pd.concat([df, custom_series.to_frame().T])
                res = recommend_user_based(temp_df, custom_name)
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([3, 2])
                with col_res1:
                    st.subheader(f"🍿 Recommendations for {custom_name}")
                    for rank, (m, r) in enumerate(res["peer_ratings"].items(), 1):
                        st.markdown(f"""
                        <div style="background:#1e293b; border-radius:8px; padding:14px; margin-bottom:8px; border-left:4px solid #10b981;">
                            <span style="font-size:1.15rem; font-weight:600; color:#f8fafc;">#{rank} {m}</span>
                            <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                                Rated <strong>{r}/5</strong> by your taste match <strong>{res['top_peer']}</strong> (Similarity: {res['top_peer_sim']:.1%}).
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_res2:
                    st.subheader("Your Taste Twins")
                    twin_df = res["similar_users"].reset_index()
                    twin_df.columns = ["User", "Similarity"]
                    fig = px.bar(twin_df, x="Similarity", y="User", orientation="h", color="Similarity", color_continuous_scale="Viridis")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=250, margin=dict(l=0, r=0, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Item-based
                temp_df = pd.concat([df, custom_series.to_frame().T])
                item_preds, item_sim = recommend_item_based(temp_df, custom_name)
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([3, 2])
                with col_res1:
                    st.subheader(f"🍿 Item-Based Recommendations for {custom_name}")
                    for rank, (m, score) in enumerate(item_preds.items(), 1):
                        st.markdown(f"""
                        <div style="background:#1e293b; border-radius:8px; padding:14px; margin-bottom:8px; border-left:4px solid #8b5cf6;">
                            <span style="font-size:1.15rem; font-weight:600; color:#f8fafc;">#{rank} {m}</span>
                            <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                                Predicted Affinity: <strong>{score:.2f} / 5.0</strong> based on items you already liked.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_res2:
                    st.subheader("Movie Correlation Heatmap")
                    fig = px.imshow(item_sim.round(2), text_auto=True, color_continuous_scale="Blues")
                    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

# ----------------- VIEW 3: SIMILARITY & MATRICES -----------------
elif mode == "📊 Similarity & Matrices":
    st.markdown('<div class="main-title">Interactive Similarity Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Inspect the mathematical representation of user ratings, user-user similarities, and item correlations.</div>', unsafe_allow_html=True)
    
    df = st.session_state.ratings
    user_sim = calculate_user_similarity(df)
    item_sim = calculate_item_similarity(df)
    
    tab1, tab2, tab3 = st.tabs(["👥 User-User Cosine Similarity", "🎬 Item-Item Similarity", "📋 Raw User-Item Ratings"])
    
    with tab1:
        st.subheader("User-User Cosine Similarity Matrix")
        st.caption("Values range from 0.0 (completely dissimilar tastes) to 1.0 (identical rating patterns).")
        fig_u = px.imshow(
            user_sim.round(3),
            text_auto=True,
            color_continuous_scale="Magma",
            aspect="auto",
            labels=dict(x="User", y="User", color="Cosine Sim")
        )
        fig_u.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig_u, use_container_width=True)
        
    with tab2:
        st.subheader("Item-Item Cosine Similarity Matrix")
        st.caption("How similar two movies are based on how all users rated them.")
        fig_i = px.imshow(
            item_sim.round(3),
            text_auto=True,
            color_continuous_scale="Teal",
            aspect="auto",
            labels=dict(x="Movie", y="Movie", color="Cosine Sim")
        )
        fig_i.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig_i, use_container_width=True)
        
    with tab3:
        st.subheader("Raw User-Item Matrix")
        st.caption("0 represents missing/unseen items, while 1 to 5 represents explicit user ratings.")
        fig_r = px.imshow(
            df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(x="Movie", y="User", color="Rating")
        )
        fig_r.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_r, use_container_width=True)

# ----------------- VIEW 4: EDIT RATINGS & CATALOG -----------------
elif mode == "✏️ Edit Ratings & Catalog":
    st.markdown('<div class="main-title">Manage Dataset & Catalog</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Modify user ratings live, add new users, or introduce new movies to the catalog.</div>', unsafe_allow_html=True)
    
    df = st.session_state.ratings.copy()
    
    st.markdown("#### 📝 Live Editable Ratings Grid")
    st.caption("Edit values directly below (use ratings 0 to 5), then click 'Save Matrix Changes'.")
    edited_df = st.data_editor(df, use_container_width=True)
    
    if st.button("💾 Save Matrix Changes", type="primary"):
        # Validate values between 0 and 5
        if ((edited_df < 0) | (edited_df > 5)).any().any():
            st.error("Ratings must be between 0 and 5!")
        else:
            st.session_state.ratings = edited_df
            st.success("Rating matrix successfully updated!")
            st.rerun()
            
    st.markdown("---")
    col_u, col_m = st.columns(2)
    
    with col_u:
        st.subheader("➕ Add New User")
        new_user_name = st.text_input("New User Name:")
        if st.button("Add User"):
            if not new_user_name.strip():
                st.error("Please enter a valid user name.")
            elif new_user_name in st.session_state.ratings.index:
                st.error(f"User '{new_user_name}' already exists.")
            else:
                new_row = pd.Series(0, index=st.session_state.ratings.columns, name=new_user_name)
                st.session_state.ratings = pd.concat([st.session_state.ratings, new_row.to_frame().T])
                st.success(f"Added user '{new_user_name}'!")
                st.rerun()

    with col_m:
        st.subheader("➕ Add New Movie")
        new_movie_title = st.text_input("New Movie Title:")
        if st.button("Add Movie"):
            if not new_movie_title.strip():
                st.error("Please enter a valid movie title.")
            elif new_movie_title in st.session_state.ratings.columns:
                st.error(f"Movie '{new_movie_title}' already exists.")
            else:
                st.session_state.ratings[new_movie_title] = 0
                st.success(f"Added movie '{new_movie_title}'!")
                st.rerun()

# ----------------- VIEW 5: HOW IT WORKS -----------------
elif mode == "📖 How It Works":
    st.markdown('<div class="main-title">How Collaborative Filtering Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A visual and mathematical breakdown of the recommendation pipeline.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Vector Representation
    Each user is represented as an $n$-dimensional vector $\mathbf{u}$, where each dimension corresponds to a movie rating:
    $$\\mathbf{u}_{\\text{Alice}} = [5, 4, 0, 0, 1]$$
    
    ### 2. Cosine Similarity Formula
    To compare the affinity between two users $\\mathbf{A}$ and $\\mathbf{B}$, we compute the cosine of the angle between their rating vectors:
    
    $$\\text{Cosine Similarity}(\\mathbf{A}, \\mathbf{B}) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\cdot \\sqrt{\\sum_{i=1}^{n} B_i^2}}$$
    
    - **1.0**: Perfect alignment (users have identical preferences).
    - **0.0**: Orthogonal (no shared preferences or non-overlapping ratings).
    
    ### 3. Recommendation Generation
    1. **Find Unseen Items**: Extract all movies where $\\text{Rating}_{\\text{target}} = 0$.
    2. **Identify Top Peer**: Find the user $\\mathbf{U}_{\\text{peer}}$ with maximum cosine similarity to the target.
    3. **Rank Candidate Items**: Sort the unwatched movies by the top peer's rating.
    
    ---
    ### 🚀 Production Extensions
    - **Matrix Factorization (SVD / ALS)**: Solves sparsity and scales to millions of users.
    - **Cold Start Problem**: Use content-based signals (genres, actors, keywords) when a new user has zero ratings.
    - **Normalized Ratings (Mean-Centering)**: Subtract user average rating to adjust for generous vs. harsh raters.
    """)

