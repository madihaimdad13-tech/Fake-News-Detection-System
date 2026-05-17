import requests
from bs4 import BeautifulSoup
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import time
from newspaper import Article

# load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# page config
st.set_page_config(
    page_title="AI News Intelligence",
    page_icon="🤖",
    layout="wide"
)

# custom css
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00FFAA;
}

.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

textarea {
    background-color: #1E1E1E !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# sidebar
st.sidebar.title("🤖 AI Intelligence Panel")

st.sidebar.success("System Status: ONLINE")

st.sidebar.write("""
### Features
✔ AI Fake News Detection  
✔ Confidence Analysis  
✔ Prediction Dashboard  
✔ Smart AI Interface  
✔ NLP + Machine Learning  
""")

# title
st.markdown("""
<h1 style='text-align:center;'>
🧠 AI NEWS INTELLIGENCE SYSTEM
</h1>
""", unsafe_allow_html=True)

st.markdown("## 🔍 Advanced Fake News Detection using Artificial Intelligence")

# input
news = st.text_area(
    "📰 Enter News Content:",
    height=250,
    placeholder="Paste news article here..."
)
# URL input
url = st.text_input("🌍 Paste News URL")

if url:

    try:

        response = requests.get(url)

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")

        news = ""

        for p in paragraphs:
            news += p.get_text()

        st.success("✅ News fetched successfully!")

        st.text_area("📄 Extracted News", news, height=250)

    except:
        st.error("❌ Unable to fetch news from URL")

# analyze button
if st.button("🚀 ANALYZE NEWS"):

    if news.strip() == "":
        st.warning("⚠ AI requires news text input!")

    else:

        with st.spinner("🤖 AI is analyzing news patterns..."):
            time.sleep(2)

        # vectorize
        vec = vectorizer.transform([news])

        # prediction
        result = model.predict(vec)

        # confidence
        probability = model.predict_proba(vec)
        confidence = max(probability[0]) * 100

        st.markdown("---")

        col1, col2 = st.columns(2)

        # result section
        with col1:

            st.subheader("🧠 AI Prediction Result")

            if result[0] == 0:

                st.error("🔴 FAKE NEWS DETECTED")

                st.write("⚠ AI detected misleading patterns.")

                fake_prob = confidence
                real_prob = 100 - confidence

            else:

                st.success("🟢 VERIFIED REAL NEWS")

                st.write("✅ AI verification completed successfully.")

                real_prob = confidence
                fake_prob = 100 - confidence

            st.write(f"### 🎯 Confidence Score: {confidence:.2f}%")

        # chart section
        with col2:

            st.subheader("📊 AI Confidence Analysis")

            labels = ["Fake", "Real"]
            values = [fake_prob, real_prob]

            fig, ax = plt.subplots()

            ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%'
            )

            st.pyplot(fig)

        # statistics
        st.markdown("---")

        st.subheader("📈 Content Analysis")

        words = len(news.split())
        chars = len(news)

        col3, col4 = st.columns(2)

        with col3:
            st.info(f"📝 Word Count: {words}")

        with col4:
            st.info(f"🔡 Character Count: {chars}")

# footer
st.markdown("---")
st.markdown("""
<center>
🤖 Powered by NLP + Machine Learning + Streamlit
</center>
""", unsafe_allow_html=True)