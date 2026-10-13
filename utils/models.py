"""
utils/models.py
---------------
Cached model trainers shared by several pages. st.cache_resource means each
model is trained only once per session/server, so pages load instantly.
"""
import numpy as np
import pandas as pd
import streamlit as st

from algorithms.id3 import ID3Tree
from algorithms.kmeans import KMeans
from utils.preprocessing import TARGET, load_dataset
from utils.session import FEATURES


@st.cache_resource
def train_id3() -> tuple[ID3Tree, float, pd.DataFrame]:
    """
    Train the ID3 tree once. We also hold out 20% of the rows to measure
    accuracy honestly (the model used for predictions is trained on all rows).
    Returns (model, held_out_accuracy, dataframe).
    """
    df, _ = load_dataset()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    check = ID3Tree().fit(train, FEATURES, TARGET)
    accuracy = float(np.mean(np.array(check.predict(test)) == test[TARGET].to_numpy()))

    model = ID3Tree().fit(df, FEATURES, TARGET)
    return model, accuracy, df


@st.cache_resource
def train_kmeans(k: int) -> KMeans:
    """Train K-Means for a given k (cached per k)."""
    df, _ = load_dataset()
    return KMeans(k=k).fit(df[FEATURES].to_numpy(float))
