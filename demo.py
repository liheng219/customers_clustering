


import streamlit as st

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Chọn trang", ["Home", "About"])

# Nội dung chính
st.title("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE K306")
st.image("data_science.jpg", caption="Recommender System")

# CSS hiển thị tên 2 người ở dưới cùng sidebar, căn giữa
st.markdown(
    """
    <style>
        [data-testid="stSidebar"]::after {
            content: "LÊ THỊ HẰNG\\A CHÂU HỮU NGHĨA";
            white-space: pre-line;   /* Cho phép xuống dòng */
            position: absolute;
            bottom: 10px;
            left: 0;
            width: 100%;
            text-align: center;      /* Căn giữa ngang */
            font-size: 14px;
            font-weight: bold;
            color: #444;
        }
    </style>
    """,
    unsafe_allow_html=True
)
