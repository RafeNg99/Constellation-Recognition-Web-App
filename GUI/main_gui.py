import streamlit as st
import base64

def get_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(layout="wide")
st.title('Constellation Recognizer App')

left_col, right_col = st.columns([1, 3])

bg_img_path = "Web App Background.jpg"
bg_img_b64 = get_base64(bg_img_path)

st.markdown(
    f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_img_b64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
<style>
""",
unsafe_allow_html=True
)

with left_col:
    uploaded_file = st.file_uploader("Browse an image", type=["png", "jpg", "jpeg"])
    run_btn = st.button("Run", width=75)

    if run_btn:
        if uploaded_file is not None:
            with st.spinner("Processing image..."):



                st.success("Process complete!")

        else:
            st.warning("Please upload an image.")


with right_col:
    if run_btn:
        # st.image(result_img, caption="")
        # st.code(result_txt, language="text")

        st.success("Results displayed!")
