from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

# Function to perform sentiment analysis on text
def score(x):
    blob1 = TextBlob(x)
    return blob1.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

st.header('Sentiment Analysis of Text Files')

with st.expander('Analyze Text'):
    text = st.text_input('Enter text here:')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text:')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))

with st.expander('Analyze Text File'):
    upl = st.file_uploader('Upload your text file', type=["txt"])

    if upl:
        # Read the uploaded text file
        file_content = upl.read().decode('utf-8')  # Read the file content as a string
        
        # Display the content of the file (first 500 characters to avoid overload)
        st.write("File content preview:")
        st.text(file_content[:500])  # Show first 500 characters as a preview

        # Process each line for sentiment analysis
        lines = file_content.split('\n')
        sentiments = []

        for line in lines:
            sentiment_score = score(line)
            sentiment_analysis = analyze(sentiment_score)
            sentiments.append({'Text': line, 'Polarity': sentiment_score, 'Sentiment': sentiment_analysis})

        # Convert the results to a DataFrame
        sentiment_df = pd.DataFrame(sentiments)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.write(sentiment_df)

        # Provide an option to download the results as CSV
        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(sentiment_df)

        st.download_button(
            label="Download sentiment analysis results as CSV",
            data=csv,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv',
        )
