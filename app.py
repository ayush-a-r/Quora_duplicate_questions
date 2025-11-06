import streamlit as st
import pickle
import helper

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

if st.button('Find'):
    if q1.strip() and q2.strip():
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]

        if result:
            st.success('✅ Duplicate')
        else:
            st.error('❌ Not Duplicate')
    else:
        st.warning('⚠️ Please enter both questions.')
