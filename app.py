import streamlit as st
import nltk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.sentiment import SentimentIntensityAnalyzer

# Downloading and initializing sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Loading GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensuring tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_content(prompt, sentiment, max_retries=5, retries=0):
    
    if retries >= max_retries:
        return "Couldn't generate suitable content after multiple attempts."

    if sentiment == "positive":
        prompt = f"{prompt} This made me feel incredibly happy and grateful."
    elif sentiment == "negative":
        prompt = f"{prompt} It was an awful experience, and I felt deeply upset."
    elif sentiment == "neutral":
        prompt = f"{prompt} It was just another ordinary day."

    # Encode input and create attention mask
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Generate text with anti-repetition strategies
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  
        repetition_penalty=1.2,  
        no_repeat_ngram_size=3,  
        temperature=0.7,  
        top_k=50,
        top_p=0.95  
    )

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Analyze sentiment
    sentiment_score = sia.polarity_scores(generated_text)["compound"]

    # Define sentiment thresholds for better flexibility
    if sentiment == "positive" and sentiment_score < 0.2:
        return generate_content(prompt, sentiment, max_retries, retries + 1)
    elif sentiment == "negative" and sentiment_score > -0.2:
        return generate_content(prompt, sentiment, max_retries, retries + 1)
    elif sentiment == "neutral" and (sentiment_score > 0.2 or sentiment_score < -0.2):
        return generate_content(prompt, sentiment, max_retries, retries + 1)

    return generated_text

# ----------------- STREAMLIT APP -----------------
st.title("📝 Sentiment-Based Text Generator")

st.markdown("""
This app generates text based on your **prompt** and the selected **sentiment**.
It uses GPT-2 and NLP techniques to ensure the output aligns with the desired emotion.
""")

# User input fields
prompt = st.text_area("Enter a prompt:", "The day was going great until")
sentiment = st.selectbox("Choose sentiment:", ["positive", "negative", "neutral"])

# Generate button
if st.button("Generate Text"):
    with st.spinner("Generating... Please wait."):
        generated_text = generate_content(prompt, sentiment)
    st.subheader("Generated Text:")
    st.write(generated_text)

st.markdown("### 🔥 Built with GPT-2, NLTK, and Streamlit")
