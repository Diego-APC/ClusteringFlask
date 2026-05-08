import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def scale_data(df, scaler=StandardScaler()):
    """Aplica escalado a las columnas numéricas."""
    scaler.fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    print("Datos escalados con StandardScaler.")
    return df_scaled, scaler

def plot_elbow_method(df_scaled, max_clusters=10, fig_path='figures/elbow_plot.png'):
    """Genera gráfica del método del codo."""
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    inertias = []
    K_range = range(1, max_clusters+1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia (suma de distancias cuadradas intra-cluster)')
    plt.title('Método del Codo para selección de k')
    plt.grid(True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Gráfica del codo guardada en {fig_path}")
    return inertias

def train_kmeans(df_scaled, n_clusters, random_state=42):
    """Entrena modelo K-Means con número fijo de clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    print(f"Modelo K-Means entrenado con k={n_clusters}")
    return kmeans, labels

def plot_clusters_pca(df_scaled, labels, fig_path='figures/clusters_pca.png'):
    """Reduce a 2D con PCA y visualiza los clusters."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    plt.figure(figsize=(10,7))
    scatter = plt.scatter(components[:,0], components[:,1], c=labels, cmap='viridis', alpha=0.7, edgecolors='w')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Visualización de Clusters (PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Gráfica de clusters guardada en {fig_path}")

def plot_centroids_heatmap(kmeans, scaler, original_columns, fig_path='figures/centroids_heatmap.png'):
    """Muestra los centroides en la escala original mediante heatmap."""
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    df_centroids = pd.DataFrame(centroids_original, columns=original_columns)
    df_centroids.index.name = 'Cluster'
    
    plt.figure(figsize=(10,6))
    sns.heatmap(df_centroids, annot=True, fmt='.1f', cmap='coolwarm', linewidths=0.5)
    plt.title('Centroides de los clusters (valores originales)')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Heatmap de centroides guardado en {fig_path}")
    return df_centroids

def cluster_profile(df_original, labels, output_table_path='figures/cluster_profiles.csv'):
    """Genera tabla con medias de las variables originales por cluster."""
    df_original['Cluster'] = labels
    profile = df_original.groupby('Cluster').mean()
    profile.to_csv(output_table_path)
    print(f"Perfiles medios por cluster guardados en {output_table_path}")
    return profile

def run_kmeans_pipeline(cleaned_data_path, n_clusters=None, max_k_for_elbow=10):
    """Ejecuta todo el pipeline de clustering."""
    # Cargar datos limpios
    df = pd.read_csv(cleaned_data_path)
    print("Datos limpios cargados para clustering.")
    
    # Escalar
    df_scaled, scaler = scale_data(df)
    
    # Determinar k si no se proporciona
    if n_clusters is None:
        inertias = plot_elbow_method(df_scaled, max_clusters=max_k_for_elbow)
        # El usuario debe elegir k según el codo.
        n_clusters = int(input("Observa la gráfica del codo. Introduce el número de clusters deseado: "))
        # Si se ejecuta sin interacción, se puede usar un valor por defecto: n_clusters=3
    # Entrenar
    kmeans, labels = train_kmeans(df_scaled, n_clusters)
    
    # Visualizaciones
    plot_clusters_pca(df_scaled, labels)
    df_centroids_original = plot_centroids_heatmap(kmeans, scaler, df.columns)
    profile = cluster_profile(df, labels)
    
    return kmeans, labels, profile, df_centroids_original