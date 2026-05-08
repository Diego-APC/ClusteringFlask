import io
import base64
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import data_preparation as dp
import kmeans_clustering as kc

app = Flask(__name__)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img


def get_cleaned_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'credit_card_customer_data.csv')
    df = dp.load_data(csv_path)
    df = dp.drop_non_predictive_columns(df)
    df = dp.handle_missing_values(df)
    df = dp.cap_outliers(df)
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/descripcion/")
def descripcion():
    return render_template("descripcion.html")


@app.route("/dataset/")
def dataset():
    return render_template("dataset.html")


@app.route("/conceptos/")
def conceptos():
    return render_template("conceptos.html")


@app.route("/codo/", methods=["GET", "POST"])
def codo():
    df = get_cleaned_data()
    df_scaled, scaler = kc.scale_data(df)

    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_range), inertias, 'bo-')
    ax.set_xlabel('Número de clusters (k)')
    ax.set_ylabel('Inercia')
    ax.set_title('Método del Codo para selección de k')
    ax.grid(True)
    elbow_img = fig_to_base64(fig)
    plt.close(fig)

    if request.method == "POST":
        k = int(request.form.get("k", 3))
        return redirect(url_for('ejecucion', k=k))

    return render_template("codo.html", elbow_img=elbow_img)


@app.route("/ejecucion/")
def ejecucion():
    k = int(request.args.get("k", 3))
    return render_template("ejecucion.html", k=k)


@app.route("/clusters/")
def clusters():
    k = int(request.args.get("k", 3))
    df = get_cleaned_data()
    df_scaled, scaler = kc.scale_data(df)
    kmeans_model, labels = kc.train_kmeans(df_scaled, k)

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        components[:, 0], components[:, 1],
        c=labels, cmap='viridis', alpha=0.7, edgecolors='w'
    )
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('Visualización de Clusters (PCA)')
    ax.grid(alpha=0.3)
    scatter_img = fig_to_base64(fig)
    plt.close(fig)

    return render_template("clusters.html", scatter_img=scatter_img, k=k)


@app.route("/centroides/")
def centroides():
    k = int(request.args.get("k", 3))
    df = get_cleaned_data()
    df_scaled, scaler = kc.scale_data(df)
    kmeans_model, labels = kc.train_kmeans(df_scaled, k)

    centroids_original = scaler.inverse_transform(kmeans_model.cluster_centers_)
    df_centroids = pd.DataFrame(centroids_original, columns=df.columns)
    df_centroids.index.name = 'Cluster'

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_centroids, annot=True, fmt='.1f', cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Centroides de los clusters (valores originales)')
    plt.tight_layout()
    heatmap_img = fig_to_base64(fig)
    plt.close(fig)

    return render_template("centroides.html", heatmap_img=heatmap_img, k=k)


@app.route("/interpretacion/")
def interpretacion():
    k = int(request.args.get("k", 3))
    df = get_cleaned_data()
    df_scaled, scaler = kc.scale_data(df)
    kmeans_model, labels = kc.train_kmeans(df_scaled, k)

    df_result = df.copy()
    df_result['Cluster'] = labels

    profile = df_result.groupby('Cluster').mean().round(2)
    profile_html = profile.to_html(classes='table table-bordered table-sm', border=0)

    return render_template(
        "interpretacion.html",
        profile_html=profile_html,
        table_data=df_result.head(50).to_dict('records'),
        total_records=len(df_result),
        columns=list(df.columns) + ['Cluster'],
        k=k
    )


if __name__ == "__main__":
    app.run(debug=True)
