import streamlit as st
import pickle

model = pickle.load(open('emotion_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def predict_emotion(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


st.title("Emotion Detector")
st.write("Analyze emotions in your messages!")

user_input = st.text_area("Enter your message:")

if st.button("Analyze"):
    if user_input:
        vec = vectorizer.transform([user_input])
        probas = model.predict_proba(vec)[0]
        
        #getting only top 3 as sequences can contain multiple emotions at the same time
        top_3_idx = probas.argsort()[-3:][::-1]
        
        st.success(f"### Primary Emotion: **{model.classes_[top_3_idx[0]]}**")
        st.write("**Other detected emotions:**")
        for idx in top_3_idx[1:]:
            emotion = model.classes_[idx]
            confidence = probas[idx]
            st.write(f"- {emotion} ({confidence:.1%})")
    else:
        st.warning("Please enter a message!")