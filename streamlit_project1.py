import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import * 
from sklearn.preprocessing import RobustScaler
import squarify
import pickle
from streamlit_option_menu import option_menu
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Using menu
st.title("")
menu = ["Home", "TOPIC", "Business Objective", "Tổng quan về cửa hàng", "Phân loại khách hàng"]
choice = st.sidebar.selectbox('Menu', menu)
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

if choice == 'Home':    
    st.subheader("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE K306")
    st.image("data_science.jpg", width=700, caption="Customer Clustering")
elif choice == 'TOPIC':    
    st.subheader("Project_1: Customer Clustering- KMEANS")
    st.image("RFM.png", width=500, caption="Customer Clustering Kmeans")
elif choice == 'Business Objective':
    st.image("store.jpg", width=500, caption="Customer Clustering")
    st.subheader("About Store X")
    st.write("""
    Specializes in essential products: vegetables, fruits, meat, fish, eggs, milk, beverages, etc.

    Target customers: retail buyers, serving daily consumption needs.
    """)

    st.subheader("Project Objectives: Customer segmentation for easier management and advertising")
    st.write("""
    - Increase sales revenue by analyzing customer purchasing behavior.
    - Reach the right target audience and promote suitable products.
    - Enhance the shopping experience, improve customer care, and ensure satisfaction.
    """)
    
          
elif choice == 'Tổng quan về cửa hàng':
    
    st.image("store.jpg", width=400, caption="Customer Clustering")
    st.write("### Đọc dữ liệu của cửa hàng từ file csv")
    

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        selected = option_menu(
        menu_title=None,
        options=["Dữ liệu", "Biểu đồ", "Phân nhóm khách hàng"],
        icons=["file-earmark-text", "bar-chart-line", "image"],
        orientation="horizontal",
        styles={"nav-link": {
            "font-size": "14px",                                    # giảm chữ
            "padding": "4px 10px"                                   # giảm padding của tab
        },
                "nav-link-selected": {"background-color":"#a8dadc", "color": "black"}} )
        df = pd.read_csv(uploaded_file)
        if selected == "Dữ liệu":
            st.write("Dữ liệu đã nhập:")
            st.dataframe(df)
            st.write("#### Tổng quan về cửa hàng:")

            min_date = pd.to_datetime(df['Date']).min().date()
            max_date = pd.to_datetime(df['Date']).max().date()

            #  Tổng doanh thu
            total_sales = df['Gross_sales'].sum()

            #  Tổng số đơn hàng (OrderID duy nhất)
            total_orders = df['OrderID'].nunique()

            #  Tổng số khách hàng
            total_customers = df['Member_number'].nunique()

            #  Tổng số mặt hàng (productId duy nhất)
            total_products = df['productId'].nunique()

            #  Tổng số category
            total_categories = df['Category'].nunique()
            st.write(f"#### ⏳ Thời gian:  {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Tổng doanh thu", f"{total_sales:,.0f}")
                st.metric("🏷️ Tổng số Category", f"{total_categories}")

            with col2:
                st.metric("👥 Tổng số khách hàng", f"{total_customers:,.0f}")
                st.metric("🛒 Tổng số mặt hàng", f"{total_products}")

            with col3:
                st.metric("📦 Tổng số đơn hàng", f"{total_orders:,.0f}")

        elif selected == "Biểu đồ":
            st.write("#### 1. Biểu đồ doanh thu của cửa hàng và số đơn hàng:")
            df['Date'] = pd.to_datetime(df['Date'])

            # Lấy danh sách năm có trong dữ liệu
            years_available = sorted(df['Date'].dt.year.unique())

            # Người dùng chọn năm (multiselect cho tiện)
            selected_years = st.multiselect(
                "Chọn năm muốn xem",
                years_available,
                default=[years_available[0]]
            )

            # Vẽ biểu đồ
            fig = plot_revenue_orders(df, years=selected_years if selected_years else None)
            st.pyplot(fig)

            st.write("#### 2. Top 10 sản phẩm bán chạy và bán chậm nhất:")
            years = sorted(df['Date'].dt.year.unique())
            year_selected = st.selectbox("Chọn năm (hoặc Toàn bộ)", ["All"] + list(years))

            if year_selected == "All":
                year_filter = None
            else:
                year_filter = int(year_selected)

        
            mode_selected = st.radio(
                "Chọn loại sản phẩm", ["Bán chạy", "Bán chậm"], horizontal=True)

            # Lấy dữ liệu
            mode = "top" if mode_selected == "Bán chạy" else "slow"
            products = top_products(df, year=year_filter, top_n=10, mode=mode)

            
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x="productName", y="OrderID", data=products, ax=ax, palette="Blues_r" if mode=="top" else "Reds_r")

            ax.set_title(f"Top 10 sản phẩm {mode_selected} ({year_selected})")
            ax.set_xlabel("Sản phẩm")
            ax.set_ylabel("Số đơn hàng")
            ax.tick_params(axis='x', rotation=70)   
            st.pyplot(fig)
        
        elif selected == "Phân nhóm khách hàng":
            with open("kmeans_rfm_model.pkl", "rb") as f:
                scaler, model_kmeans = pickle.load(f)

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            max_date = df['Date'].max()

            Recency = lambda x: (max_date - x.max()).days
            Frequency  = lambda x: len(x.unique())
            Monetary = lambda x : round(sum(x), 2)

            data_RFM = df.groupby('Member_number').agg({'Date': Recency,
                                                    'OrderID': Frequency,
                                                    'Gross_sales': Monetary })
            
            data_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            df_log = data_RFM[["Recency","Frequency","Monetary"]].apply(lambda x: np.log1p(x))
            data_final = scaler.transform(df_log)
            preds = model_kmeans.predict(data_final)
            data_RFM["Cluster"] = preds
            data = summarize_clusters(data_RFM,'Cluster' )
            data['Segment'] = data['Cluster'].map(assign_segment)
            colors_dict2 = {
                'New customers':'yellow',
                'At risk':'royalblue',
                'Potential customers':'cyan',
                'Lost':'red',
                'Champions':'pink'
            }
            chart_type = st.radio("Chọn loại biểu đồ:", ["Scatter Plot", "Treemap"])
            if chart_type == "Treemap": 
                fig = plot_rfm_treemap(data, colors_dict=colors_dict2)
                st.pyplot(fig)
            elif chart_type == "Scatter Plot":
                fig1 = px.scatter(data, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Segment",
                hover_name="Segment", size_max=50)
                st.plotly_chart(fig1, use_container_width=True)


    
elif choice=='Phân loại khách hàng':
    st.image("RFM.png", width=500, caption="Customer Clustering Kmeans")
    with open("kmeans_rfm_model.pkl", "rb") as f:
     scaler, model_kmeans = pickle.load(f)

    # Nhập 1 khách hàng
    st.write("### Nhập 1 khách hàng")
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Chọn ngày mua cuối: ")
    with col2:
        fre = st.number_input("Nhập số đơn hàng:", min_value=1, step=1)
    with col3:
        mone = st.number_input("Nhập tổng chi tiêu:", min_value=0.0, step=10.0)

    # Tính Recency so với ngày hiện tại (hoặc max_date trong dataset gốc)
    max_date = pd.to_datetime(date)  
    today = pd.to_datetime("today")  
    recency = (today - max_date).days

    # Tạo DataFrame RFM cho 1 khách hàng
    data_ = pd.DataFrame([{
        "Recency": recency,
        "Frequency": fre,
        "Monetary": mone
    }])

    # Chuẩn hóa & dự đoán cluster
    df_ = data_[["Recency","Frequency","Monetary"]].apply(lambda x: np.log1p(x))
    _final = scaler.transform(df_)
    pred = model_kmeans.predict(_final)
    data_["Cluster"] = pred

    # Gắn nhãn phân khúc
    data_["Segment"] = data_["Cluster"].astype(str).map(assign_segment)

    if st.button("Phân loại khách hàng"):
        st.write(f'📌 Khách hàng thuộc nhóm: **{data_["Segment"].iloc[0]}**')

    # Trường hợp 2: Đọc dữ liệu từ file csv
    st.write("### Hoặc đọc dữ liệu từ file csv")
    st.write("#### Dữ liệu mẫu")
    temp = {
    "CustomerID": 0,
    "Date": "2025-06-01",
    "Frequency": 5,
    "Monetary": 200}

    # Chuyển thành DataFrame
    temp = pd.DataFrame(temp)
    st.dataframe(temp)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
        st.write("Dữ liệu đã nhập:")
        st.dataframe(df)
        if st.button("Phân loại khách hàng từ file"):
            # Hiển thị kết quả ra dataframe
            st.write("Kết quả phân loại khách hàng:")
            df['Date'] = pd.to_datetime(df['Date'])

            # Ngày tham chiếu = ngày mới nhất trong dữ liệu
            reference_date = df['Date'].max()
            df1 = df[["Date", "Frequency", "Monetary"]]
            df1['Recency'] = (reference_date - df['Date']).dt.days
            df1 = df1[["Recency","Frequency","Monetary"]].apply(lambda x: np.log1p(x))
            final = scaler.transform(df1)
            predicts = model_kmeans.predict(final)
            df["Cluster"] = predicts
            # Gắn nhãn phân khúc
            df["Segment"] = df["Cluster"].astype(str).map(assign_segment)
            st.dataframe(df)

    

# Done
    
    
    
        

        
        

    





