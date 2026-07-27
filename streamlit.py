import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("🎬 MovieIQ")
st.subheader("Movie Success Prediction Dashboard")

df = pd.read_csv("movies.csv")
model = joblib.load("model.pkl")

df = pd.read_csv("movies.csv")

# Sidebar Filters

st.sidebar.header("Movie Filter")

genre = st.sidebar.selectbox(
    "Select Genre",
    sorted(df["genres"].unique())
)

vote = st.sidebar.slider(
    "Minimum Vote Average",
    0.0,
    10.0,
    5.0
)

filtered_df = df[
    (df["genres"] == genre) &
    (df["vote_average"] >= vote)
]

# Show Dataset
st.header("Filtered Movies")

st.dataframe(filtered_df)

# Chart 1

st.header("Budget vs Revenue")

fig, ax = plt.subplots()

ax.scatter(
    filtered_df["budget"],
    filtered_df["revenue"]
)

ax.set_xlabel("Budget")
ax.set_ylabel("Revenue")

st.pyplot(fig)

# Chart 2

st.header("Top Genres")

fig, ax = plt.subplots()

df["genres"].value_counts().head(10).plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# Prediction
st.header("Movie Success Prediction")

budget = st.number_input("Budget", min_value=0)

popularity = st.number_input("Popularity", min_value=0.0)

runtime = st.number_input("Runtime", min_value=0)

vote_average = st.slider(
    "Vote Average",
    0.0,
    10.0,
    5.0
)

if st.button("Predict"):

    prediction = model.predict(
        [[budget, popularity, runtime, vote_average]]
    )

    if prediction[0] == 1:
        st.success("✅ Movie will be Successful")
    else:
        st.error("❌ Movie may not be Successful")
