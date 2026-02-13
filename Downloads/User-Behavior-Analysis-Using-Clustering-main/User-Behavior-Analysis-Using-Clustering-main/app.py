"""
Streamlit Web Application for User Behavior Clustering Analysis
This app implements the clustering analysis from Part A of the Jupyter notebook
using the Ecommerce Customer dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set page configuration
st.set_page_config(
    page_title="User Behavior Clustering Analysis",
    page_icon="📊",
    layout="wide"
)

# ============================================
# Custom K-means Functions (from the notebook)
# ============================================

def scale_data(data_matrix):
    """Scale the data using StandardScaler"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    return scaled_data, scaler

def init_cluster_centers(data_mat, k):
    """Initialize cluster centers using k-means++"""
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans.fit(data_mat)
    return kmeans.cluster_centers_

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(x1 - x2)

def k_means_373(data_mat, k, initial_centers, max_iters=100, tol=1e-4):
    """Custom K-means implementation"""
    data_mat = np.array(data_mat)
    centroids = np.array(initial_centers)
    prev_centroids = np.zeros(initial_centers.shape)
    labels = np.zeros(len(data_mat))

    for iteration in range(max_iters):
        # Assignment Step
        clusters_populated = set()
        for i, point in enumerate(data_mat):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            assigned_cluster = np.argmin(distances)
            labels[i] = assigned_cluster
            clusters_populated.add(assigned_cluster)

        # Save old centroids for convergence check
        prev_centroids = centroids.copy()

        # Update Step
        for i in range(k):
            if i in clusters_populated:
                centroids[i] = np.mean(data_mat[labels == i], axis=0)
            else:
                centroids[i] = data_mat[np.random.choice(data_mat.shape[0])]

        # Convergence check
        if np.sum((centroids - prev_centroids)**2) < tol:
            break

    return centroids, labels

def intracluster_dist(data_mat, k, cluster_assignments, cluster_centers):
    """Calculate intra-cluster distances"""
    intra_distances = []
    for cluster in range(k):
        distances = np.linalg.norm(data_mat[cluster_assignments == cluster] - cluster_centers[cluster], axis=1)
        intra_distances.append(np.mean(distances))
    return intra_distances

def intercluster_dist(c1, c2, cluster_centers):
    """Calculate inter-cluster distance between two clusters"""
    return np.linalg.norm(cluster_centers[c1] - cluster_centers[c2])

def compute_silhouette_score(X, labels):
    """Compute silhouette score"""
    n_clusters = len(set(labels))
    if n_clusters == 1:
        return -1

    silhouette_vals = []
    for i, label in enumerate(labels):
        # Compute a(i) - intra-cluster distance
        intra_cluster_points = X[labels == label]
        if len(intra_cluster_points) > 1:
            intra_distance = np.mean([np.linalg.norm(x - X[i]) for x in intra_cluster_points if not np.array_equal(x, X[i])])
        else:
            intra_distance = 0

        # Compute b(i) - nearest inter-cluster distance
        inter_cluster_distances = []
        for other_cluster in set(labels):
            if other_cluster != label:
                inter_cluster_points = X[labels == other_cluster]
                if len(inter_cluster_points) > 0:
                    inter_cluster_distances.append(np.mean([np.linalg.norm(x - X[i]) for x in inter_cluster_points]))

        if inter_cluster_distances:
            b_i = min(inter_cluster_distances)
        else:
            b_i = 0

        # Compute Silhouette value
        if max(b_i, intra_distance) > 0:
            s_i = (b_i - intra_distance) / max(b_i, intra_distance)
        else:
            s_i = 0
        silhouette_vals.append(s_i)

    return np.mean(silhouette_vals)

# ============================================
# Load Data
# ============================================

@st.cache_data
def load_data():
    """Load the Ecommerce Customer data"""
    df = pd.read_csv('User-Behavior-Analysis-Using-Clustering-main/Ecommerce_Customer.csv', index_col=0)
    return df

# ============================================
# Main Application
# ============================================

def main():
    # Title and Introduction
    st.title("📊 User Behavior Clustering Analysis")
    st.markdown("""
    This application performs customer segmentation using K-means clustering on 
    Ecommerce customer data. It implements the custom K-means algorithm from scratch
    and compares it with the built-in sklearn implementation.
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Overview", "Clustering Analysis", "Model Comparison", "Visualizations"]
    )
    
    # Data Overview Page
    if page == "Data Overview":
        st.header("📁 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Number of customers:** {df.shape[0]}")
            st.write(f"**Number of features:** {df.shape[1]}")
            st.write(f"**Features:** {', '.join(df.columns.tolist())}")
        
        with col2:
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Feature Distributions")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(df.columns):
            axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide the last empty subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Clustering Analysis Page
    elif page == "Clustering Analysis":
        st.header("🔍 Clustering Analysis")
        
        # Feature selection
        st.subheader("Select Features for Clustering")
        
        all_features = df.columns.tolist()
        default_features = ['Length of Membership', 'Yearly Amount Spent']
        selected_features = st.multiselect(
            "Choose features to include in clustering:",
            all_features,
            default=default_features
        )
        
        if len(selected_features) < 2:
            st.error("Please select at least 2 features for clustering.")
            return
        
        # Get selected data
        df_selected = df[selected_features]
        
        # Scale the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_selected)
        
        st.subheader("Find Optimal Number of Clusters (K)")
        
        # Range of K values to test
        k_range = range(2, 11)
        
        # Calculate metrics for different K values
        inertia_list = []
        silhouette_scores_list = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(k_range):
            status_text.text(f"Testing K = {k}...")
            kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
            kmeans.fit(df_scaled)
            inertia_list.append(kmeans.inertia_)
            silhouette_scores_list.append(silhouette_score(df_scaled, kmeans.labels_))
            progress_bar.progress((i + 1) / len(k_range))
        
        progress_bar.empty()
        status_text.empty()
        
        # Plot Elbow Method
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elbow Method")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, inertia_list, marker='o', linestyle='--', color='blue')
            ax.set_xlabel('Number of Clusters (K)')
            ax.set_ylabel('Inertia (SSE)')
            ax.set_title('Elbow Method for Optimal K')
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Silhouette Score")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, silhouette_scores_list, marker='o', linestyle='--', color='red')
            ax.set_xlabel('Number of Clusters (K)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Score vs K')
            ax.grid(True)
            st.pyplot(fig)
        
        # Select K value
        st.subheader("Run Clustering with Selected K")
        
        optimal_k = st.slider("Select number of clusters (K):", 2, 10, 5)
        
        # Run K-means
        kmeans = KMeans(n_clusters=optimal_k, n_init=1, random_state=42)
        cluster_labels = kmeans.fit_predict(df_scaled)
        
        # Add cluster labels to dataframe
        df_result = df_selected.copy()
        df_result['Cluster'] = cluster_labels
        
        # Calculate metrics
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        intra_distances = intracluster_dist(df_scaled, optimal_k, cluster_labels, kmeans.cluster_centers_)
        avg_intra_distance = np.mean(intra_distances)
        
        inter_distances = [intercluster_dist(i, j, kmeans.cluster_centers_)
                          for i in range(optimal_k)
                          for j in range(i + 1, optimal_k)]
        avg_inter_distance = np.mean(inter_distances)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        col2.metric("Avg Intra-cluster Distance", f"{avg_intra_distance:.3f}")
        col3.metric("Avg Inter-cluster Distance", f"{avg_inter_distance:.3f}")
        
        # Cluster distribution
        st.subheader("Cluster Distribution")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Distribution Across Clusters')
        ax.set_xticklabels([f'Cluster {i}' for i in cluster_counts.index], rotation=0)
        st.pyplot(fig)
        
        # Cluster centers (in original scale)
        st.subheader("Cluster Centers (Original Scale)")
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers_original, columns=selected_features)
        centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]
        st.dataframe(centers_df, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        st.dataframe(df_result.groupby('Cluster').mean(), use_container_width=True)
    
    # Model Comparison Page
    elif page == "Model Comparison":
        st.header("⚖️ Custom vs Built-in K-means Comparison")
        
        st.markdown("""
        This section compares the custom K-means implementation (k_means_373) 
        with the built-in sklearn K-means algorithm.
        """)
        
        # Select features
        all_features = df.columns.tolist()
        default_features = ['Length of Membership', 'Yearly Amount Spent']
        selected_features = st.multiselect(
            "Select features:",
            all_features,
            default=default_features,
            key="compare_features"
        )
        
        if len(selected_features) < 2:
            st.error("Please select at least 2 features.")
            return
        
        df_selected = df[selected_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_selected)
        
        # Select K
        k_value = st.slider("Select K:", 2, 10, 5, key="compare_k")
        
        # Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Built-in K-means")
            
            # Built-in K-means
            kmeans_builtin = KMeans(n_clusters=k_value, n_init=1, random_state=42)
            labels_builtin = kmeans_builtin.fit_predict(df_scaled)
            
            sil_builtin = silhouette_score(df_scaled, labels_builtin)
            intra_builtin = np.mean(intracluster_dist(df_scaled, k_value, labels_builtin, kmeans_builtin.cluster_centers_))
            inter_builtin = np.mean([intercluster_dist(i, j, kmeans_builtin.cluster_centers_)
                                   for i in range(k_value) for j in range(i+1, k_value)])
            
            st.write(f"**Silhouette Score:** {sil_builtin:.3f}")
            st.write(f"**Avg Intra-cluster Distance:** {intra_builtin:.3f}")
            st.write(f"**Avg Inter-cluster Distance:** {inter_builtin:.3f}")
        
        with col2:
            st.subheader("Custom K-means (k_means_373)")
            
            # Custom K-means
            initial_centers = df_scaled.sample(n=k_value, random_state=42).values
            final_centers, labels_custom = k_means_373(df_scaled, k_value, initial_centers)
            
            sil_custom = silhouette_score(df_scaled, labels_custom)
            intra_custom = np.mean(intracluster_dist(df_scaled, k_value, labels_custom, final_centers))
            inter_custom = np.mean([intercluster_dist(i, j, final_centers)
                                  for i in range(k_value) for j in range(i+1, k_value)])
            
            st.write(f"**Silhouette Score:** {sil_custom:.3f}")
            st.write(f"**Avg Intra-cluster Distance:** {intra_custom:.3f}")
            st.write(f"**Avg Inter-cluster Distance:** {inter_custom:.3f}")
        
        # Summary
        st.subheader("Comparison Summary")
        comparison_df = pd.DataFrame({
            "Metric": ["Silhouette Score", "Avg Intra-cluster Distance", "Avg Inter-cluster Distance"],
            "Built-in K-means": [sil_builtin, intra_builtin, inter_builtin],
            "Custom K-means": [sil_custom, intra_custom, inter_custom]
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        if sil_builtin > sil_custom:
            st.success("Built-in K-means performs better in terms of Silhouette Score!")
        elif sil_custom > sil_builtin:
            st.success("Custom K-means performs better in terms of Silhouette Score!")
        else:
            st.info("Both methods perform equally!")
    
    # Visualizations Page
    elif page == "Visualizations":
        st.header("📈 Interactive Visualizations")
        
        # Select features and K
        all_features = df.columns.tolist()
        default_features = ['Length of Membership', 'Yearly Amount Spent']
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature:", all_features, index=3)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature:", all_features, index=4)
        
        k_value = st.slider("Select number of clusters:", 2, 10, 5, key="viz_k")
        
        # Perform clustering
        df_viz = df[[x_feature, y_feature]]
        scaler = StandardScaler()
        df_viz_scaled = scaler.fit_transform(df_viz)
        
        kmeans = KMeans(n_clusters=k_value, n_init=1, random_state=42)
        labels = kmeans.fit_predict(df_viz_scaled)
        
        # 2D Scatter plot
        st.subheader(f"2D Cluster Visualization: {x_feature} vs {y_feature}")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(df_viz[x_feature], df_viz[y_feature], c=labels, cmap='rainbow', 
                           alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, 
                 edgecolors='white', linewidth=2, label='Cluster Centers')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Customer Segments ({k_value} Clusters)')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        st.pyplot(fig)
        
        # 3D Visualization
        st.subheader("3D Cluster Visualization")
        
        # Select 3 features for 3D plot
        col1, col2, col3 = st.columns(3)
        
        with col1:
            z_feature = st.selectbox("Select 3rd feature for 3D:", all_features, index=0, key="z_feat")
        
        # Perform 3D clustering
        df_3d = df[[x_feature, y_feature, z_feature]]
        scaler_3d = StandardScaler()
        df_3d_scaled = scaler_3d.fit_transform(df_3d)
        
        kmeans_3d = KMeans(n_clusters=k_value, n_init=1, random_state=42)
        labels_3d = kmeans_3d.fit_predict(df_3d_scaled)
        
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df_3d[x_feature], df_3d[y_feature], df_3d[z_feature], 
                           c=labels_3d, cmap='rainbow', alpha=0.7)
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_zlabel(z_feature)
        ax.set_title(f'3D Customer Segments ({k_value} Clusters)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
