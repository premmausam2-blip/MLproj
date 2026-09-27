import streamlit as st
import joblib
import re

# Load the trained model and TF-IDF vectorizer
model = joblib.load("cyberbullying_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)


# Title
st.title("🛡️ Cyberbullying Detection System")

st.write(
    "Enter a comment below and the machine learning model "
    "will classify it as Bullying or Non-Bullying."
)


# Comment input
comment = st.text_area(
    "Enter a comment:",
    placeholder="Type your comment here...",
    height=150
)


# Analyze button
if st.button("Analyze Comment"):

    if comment.strip() == "":
        st.warning("Please enter a comment first.")

    else:
        # Preprocess the comment
        clean_comment = preprocess_text(comment)

        # Convert text to TF-IDF
        comment_tfidf = tfidf.transform([clean_comment])

        # Make prediction
        prediction = model.predict(comment_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(comment_tfidf)[0]

        bullying_probability = probabilities[1]
        non_bullying_probability = probabilities[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ Cyberbullying Detected")
        else:
            st.success("✅ Non-Bullying Comment")

        # Display probabilities
        st.subheader("Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Non-Bullying",
                f"{non_bullying_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Bullying",
                f"{bullying_probability * 100:.2f}%"
            )