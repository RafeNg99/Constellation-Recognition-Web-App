import streamlit as st

st.title('Constellation Recognizer App')

uploaded_file = st.file_uploader("Browse an image", type=["png", "jpg", "jpeg"])

if st.button("Run"):
    if uploaded_file is None:
        with st.spinner("Processing image..."):



            
            st.success("Process complete!")

    else:
        st.warning("Please upload an image.")
