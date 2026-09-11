import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")
iris = load_iris()

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length", value=5.1)
sw = st.number_input("Sepal Width", value=3.5)
pl = st.number_input("Petal Length", value=1.4)
pw = st.number_input("Petal Width", value=0.2)

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    predicted_species = flowers[pred[0]]
    st.success(f"Predicted Species: {predicted_species}")

    # Scatter plot: Petal Length vs Petal Width, colored by species,
    # with the prediction highlighted (similar style to the manual's graph)
    fig, ax = plt.subplots()
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for i, species in enumerate(flowers):
        idx = iris.target == i
        ax.scatter(
            iris.data[idx, 2], iris.data[idx, 3],
            label=species, color=colors[i], alpha=0.6
        )

    ax.scatter(
        pl, pw, color="red", s=150, marker="o",
        edgecolor="black", zorder=5, label="Prediction"
    )
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Iris Dataset - Petal Length vs Petal Width")
    ax.legend()
    st.pyplot(fig)
