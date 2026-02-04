import streamlit as st
import io
import requests
import base64
from PIL import Image

def get_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(layout="wide")
st.title('🔭Constellation Recognizer App')

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
    uploaded_file = st.file_uploader("Browse an image", accept_multiple_files=False)
    language = st.selectbox("Langauge", ["English", "Chinese", "Japanese", "Korean", "Malay"], index=0)
    valid_types = ["image/png", "image/jpeg"]

    # ---- Validation ----
    is_invalid_doc = uploaded_file is None or uploaded_file.type not in valid_types

    # ---- Warning immediately when invalid file is selected ----
    if uploaded_file is not None and uploaded_file.type not in valid_types:
        st.warning("⚠️ Invalid document type. Please upload a PNG or JPG image.")

    run_btn = st.button("Run", width=75, disabled=is_invalid_doc)

    if run_btn:
        if uploaded_file is not None and not is_invalid_doc:
            files = [("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))]
            with st.spinner("⏳Processing image..."):
                response = requests.post("http://127.0.0.1:9002/constellation_detector", files=files)

                if response.status_code == 200:
                    st.success("✅Process complete!")

        else:
            st.warning("⚠️Please upload an image.")


with right_col:
    if run_btn:
        if response.status_code == 200:
            json_result = response.json()
            img_bytes = base64.b64decode(json_result["yolo_img_result"])
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((640, 640), Image.LANCZOS)

            with st.container(horizontal_alignment='center'):
                st.image(img, caption="", output_format="PNG")

            constellation_list = json_result["yolo_class_result"]
            
            with left_col:
                with st.spinner("⏳Processing information..."):

                    response = requests.post("http://127.0.0.1:9002/constellation_explainer", 
                                             params={
                                                 "const_list": constellation_list if constellation_list else [""],  # can be []
                                                 "lang": language
                                                 })
                    
                    if response.status_code == 200:
                        st.success("✅Process complete!")
                    
                    else:
                        result_txt = response.content
                        st.success("⚠️Error occurred!")
            
            if response.status_code == 200:
                try:
                    json_txt = response.json()
                    result_txt = json_txt["llm_result"]
                except Exception as e:
                    result_txt = str(e)

            st.code(result_txt, language="text")
