import joblib
import pandas as pd
import numpy as np
import umap
import sklearn.cluster as cluster
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ELMoClustering:
    """
    Preparing ELMo embeddings and clustering them
    """

    def __init__(self, data_path, vecs_path, seed, umap_params, kmeans_params, out_path, ari_venv_path):
        """
        Parameters
        ----------
        data_path : str path to data
        vecs_path : str path to precomputed vectord
        seed : int state of random functions for reproducability
        umap_params : dict UMAP parameters
        kmeans_params : dict KMeans parameters
        out_path : str where to save the result
        ari_venv_path : path to venv to run the evaluation script
        """
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.vecs_path = vecs_path
        self.seed = seed
        self.umap_params = umap_params
        self.kmeans_params = kmeans_params
        self.out_path = out_path
        self.ari_venv_path = ari_venv_path

    def _make_embeddings(self):
        """
        Prepares ELMo context embeddings
        """
        vecs = joblib.load(self.vecs_path)
        self.data['vecs'] = vecs
        self.data['vecs'] = self.data.vecs.apply(lambda x: np.mean(x, axis=0))
        assert self.data.vecs[0].shape == (2560,)  # проверка, что вектор контекста правильной размерности
        emb = umap.UMAP(**self.umap_params).fit_transform(self.data['vecs'].to_list())
        self.data['embeddings'] = np.squeeze(emb).tolist()

    def fit_predict(self):
        """
        Fits KMeans for each target word and make predictions
        """
        self._make_embeddings()
        result = []
        groups = self.data.groupby('word')
        for group in groups:
            group = group[1]
            labels = cluster.KMeans(**self.kmeans_params).fit_predict(group['embeddings'].to_list())
            group.predict_sense_id = labels
            result.append(group)
        out = pd.concat(result)
        out.to_csv(self.out_path, sep='\t')

    def evaluate(self):
        """
        Runs evaluation script (tested on Linux; not sure if working on Windows)
        """
        command = f"{self.ari_venv_path}/bin/python evaluate.py {self.out_path}; exit 0"
        return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, ).decode('utf-8')
