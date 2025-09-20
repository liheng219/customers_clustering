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

# Page title
st.title("📊 Customer Segmentation Dashboard")

# Sidebar menu
menu = ["🏠 Home", "📂 TOPIC", "🎯 Business Objective", "🏪 Tổng quan về cửa hàng", "🧩 Phân loại khách hàng"]
choice = st.sidebar.selectbox('📌 Menu', menu)

# Custom sidebar styling
st.markdown(
    """
        <style>
            [data-testid="stSidebar"] {
                position: relative;
                padding-bottom: 140px !important;
                background: 
                    repeating-linear-gradient(
                        45deg,
                        rgba(255, 255, 255, 0.03) 0px,
                        rgba(255, 255, 255, 0.03) 40px,
                        transparent 40px,
                        transparent 80px
                    ),
                    linear-gradient(180deg, #1a1a1a 0%, #2a2a2a 100%) !important; /* dark luxury */
                background-size: cover;
            }

    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: calc(20px + 12px + 8px);
        width: 72%;
        max-width: 220px;
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, #f7d9a7 0%, #ffe9c6 25%, #fffaf2 50%, #ffe9c6 75%, #f7d9a7 100%);
        background-size: 200% 100%;
        filter: blur(8px);
        opacity: 0.9;
        z-index: 9998;
        animation: slideGlow 3.5s linear infinite;
        pointer-events: none;
    }

    [data-testid="stSidebar"]::after {
        content: "👩‍💼 LÊ THỊ HẰNG \\A 👨‍💼 CHÂU HỮU NGHĨA";
        white-space: pre-line;
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 86%;
        max-width: 260px;
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: #7a4e14;  /* luxury bronze-gold */
        padding: 12px 14px;
        border-radius: 12px;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        background: linear-gradient(180deg, #fffdf7 0%, #fff0d6 100%);
        box-shadow: 0 8px 22px rgba(160,120,40,0.18);
        border: 1px solid rgba(160,120,40,0.12);
        z-index: 9999;
        pointer-events: none;
    }

    @keyframes slideGlow {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stSidebar"]::before {
            animation: none;
            filter: blur(6px);
            opacity: 0.7;
        }
    }

    @media (max-width: 600px) {
        [data-testid="stSidebar"]::after {
            max-width: 200px;
            font-size: 12px;
            padding: 10px 12px;
        }
        [data-testid="stSidebar"]::before {
            max-width: 180px;
            height: 5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)




if choice == '🏠 Home':    
    # Background style
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff; /* white background */
        }
        .title {
            text-align: center; 
            color: #2C3E50;
        }
        .subtitle {
            text-align: center; 
            font-size:16px; 
            color: gray;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Custom title
    st.markdown(
        """
        <h2 class="title">
            🎓 ĐỒ ÁN TỐT NGHIỆP <br> 
            <span style="color:#3498DB;">DATA SCIENCE K306</span>
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    
    # Show main image
    st.image(
        "data_science.jpg",
        use_container_width=True,
        caption="📊 Customer Clustering"
    )

    # Intro
    st.markdown(
        """<p class="subtitle">
        Khám phá sức mạnh của dữ liệu qua phân tích & mô hình học máy
        </p>""",
        unsafe_allow_html=True
    )

    # Add extra images in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/906/906175.png", caption="📈 Machine Learning", use_container_width=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4149/4149643.png", caption="🤖 AI Models", use_container_width=True)
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2721/2721291.png", caption="📊 Data Visualization", use_container_width=True)

    # Divider
    st.markdown("---")

    # Closing message
    st.markdown(
        """<p class="subtitle">
        🌟 Chào mừng bạn đến với đồ án tốt nghiệp K306 🌟
        </p>""",
        unsafe_allow_html=True
    )

elif choice == '📂 TOPIC':    
    st.subheader("📊 Project_1: Customer Clustering - KMEANS")

    # Section title with icon
    st.markdown("### 🔎 RFM Analysis Overview")

    # Add explanation with inline icons
    st.write(
        "The **RFM Model** helps to segment customers based on their behavior "
        "before applying **K-Means clustering 🤖**. "
        "\n\n"
        "- 🕒 **Recency**: How recently a customer made a purchase\n"
        "- 🔁 **Frequency**: How often they purchase\n"
        "- 💰 **Monetary**: How much they spend"
    )

    # Show multiple RFM related pictures in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://storage.googleapis.com/anfin.vn/public/cms/ngay_giao_dich_khong_huong_quyen_0e439e8aca/ngay_giao_dich_khong_huong_quyen_0e439e8aca.webp", 
                 use_container_width=True, caption="Recency Distribution")
    with col2:
        st.image("https://cdn.accgroup.vn/wp-content/uploads/2022/11/coopmart-ban-cua-moi-nha-source-dantri.jpg", 
                 use_container_width=True, caption="Frequency Distribution")
    with col3:
        st.image("https://www.shutterstock.com/image-illustration/concept-unequal-distribution-income-3d-600nw-2120566865.jpg", 
                 use_container_width=True, caption="Monetary Distribution")

    st.markdown("---")

    # Show clustering result
    st.markdown("### Customer Clustering Result")
    st.image("https://projectgurukul.org/wp-content/uploads/2022/01/k-means-clustering-customer-segmentation-output.webp", 
             width=800, caption="Customer Clustering with KMeans")

    st.info("💡 Tip: Each cluster represents a customer group with similar buying behaviors.")

elif choice == '🎯 Business Objective':
# Main RFM clustering image
    st.image(
        "https://www.moengage.com/wp-content/uploads/2020/07/predictive-segments-using-rfm-moengage.jpg",
        width=800,
        caption="📊 Customer Clustering with RFM"
    )

    # About section with store icon
    st.subheader("🏪 About Store X")
    st.write("""
    **Specializes in essential products:**
    🥦 Vegetables, 🍎 Fruits, 🍖 Meat, 🐟 Fish, 🥚 Eggs, 🥛 Milk, 🥤 Beverages, etc.  

    **Target customers:** Retail buyers 👥, serving daily consumption needs.
    """)

    # Objectives section with dart icon
    st.subheader("🎯 Project Objectives: Customer Segmentation for Management & Advertising")
    st.write("""
    - 📈 **Increase sales revenue** by analyzing customer purchasing behavior.  
    - 🎯 **Reach the right target audience** and promote suitable products.  
    - 🤝 **Enhance shopping experience**: improve customer care and ensure satisfaction.  
    """)

    # Optional: Highlight box
    st.success("💡 With RFM + KMeans, Store X can better understand customers and create targeted marketing campaigns.")

        
          
elif choice == '🏪 Tổng quan về cửa hàng':
    
    st.image("store.jpg", width=800, caption="Customer Clustering")
    st.write("### Đọc dữ liệu của cửa hàng từ file csv")
    st.write("* Dữ liệu mẫu:")
    temp_ = {
    "Member": [1808],
    "Date": ["2015-07-21"],
    "productId": [1],
    "items": [3],
    "productName": ["tropical fruit"],
    "price": [7.8],
    "Category": ["Fresh Food"],
    "Gross_sales": [23.4],
    "OrderID": [2988]
    }

    temp_ = pd.DataFrame(temp_)
    st.dataframe(temp_)

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
            st.write( '''
                     * Champions: Chi tiêu nhiều, thường xuyên: Nên duy trì quan hệ, chăm sóc đặc biệt (ưu đãi, chương trình VIP).
                     * Potential customers: Chi tiêu vừa, khá thường xuyên: Có khả năng trở thành khách hàng trung thành nếu được chăm sóc tốt.
                     * New customers: Mua gần đây: Cần nuôi dưỡng để biến thành khách hàng trung thành.
                     * At risk: Lâu không mua, sức chi vừa: Nên có chiến dịch khuyến mãi/nhắc nhở để giữ chân.
                     * Lost: Rất lâu không ghé, chi tiêu ít: Khả năng quay lại thấp, không nên đầu tư nhiều tài nguyên.

            ''')

    
elif choice=='🧩 Phân loại khách hàng':
    st.image("RFM.png", width=500, caption="Customer Clustering Kmeans")
    st.write(''' Có 5 nhóm khách hàng: 
            * Champions.
            * Potential customers.
            * New customers.
            * At risk.
            * Lost.
             
             ''')
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
        segment = data_["Segment"].iloc[0]
        #st.write(f'📌 Khách hàng thuộc nhóm: **{segment}**')
        st.success(f"📌 Khách hàng thuộc nhóm: **{segment}**")

        if segment == 'Champions':
            st.write('Champions: Chi tiêu nhiều, thường xuyên: Nên duy trì quan hệ, chăm sóc đặc biệt (ưu đãi, chương trình VIP).')
        elif segment == 'Potential customers':
            st.write('* Potential customers: Chi tiêu vừa, khá thường xuyên: Có khả năng trở thành khách hàng trung thành nếu được chăm sóc tốt.')
        elif segment == 'New customers':
            st.write('New customers: Mua gần đây: Cần nuôi dưỡng để biến thành khách hàng trung thành.')
        elif segment == 'At risk':
            st.write('At risk: Lâu không mua, sức chi vừa: Nên có chiến dịch khuyến mãi/nhắc nhở để giữ chân.')
        elif segment == 'Lost':
            st.write('Lost: Rất lâu không ghé, chi tiêu ít: Khả năng quay lại thấp, không nên đầu tư nhiều tài nguyên.')


    # Trường hợp 2: Đọc dữ liệu từ file csv
    st.write("### Hoặc đọc dữ liệu từ file csv")
    st.write("* Dữ liệu mẫu")
    temp = {
    "CustomerID": 0,
    "Date": "2025-06-01",
    "Frequency": 5,
    "Monetary": 200}

    # Chuyển thành DataFrame
    temp = pd.DataFrame([temp])
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
    
    
    
        

        
        

    








