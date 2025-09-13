
import matplotlib.pyplot as plt

def plot_revenue_orders(data, years=None):
    data['YearMonth'] = data['Date'].dt.to_period('M')
    data['Year'] = data['Date'].dt.year

    if years is not None:
        data = data[data['Year'].isin(years)]

    monthly_data = data.groupby('YearMonth').agg({
        'Gross_sales': 'sum',
        'OrderID': 'nunique'
    }).reset_index()

    monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)

    # Tạo figure và axes
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Biểu đồ cột
    ax1.bar(monthly_data['YearMonth'], monthly_data['Gross_sales'],
            color='skyblue', label='Doanh thu')
    ax1.set_xlabel('Tháng')
    ax1.set_ylabel('Doanh thu', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Biểu đồ đường
    ax2 = ax1.twinx()
    ax2.plot(monthly_data['YearMonth'], monthly_data['OrderID'],
             color='orange', marker='o', label='Số đơn hàng')
    ax2.set_ylabel('Số đơn hàng', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Tiêu đề
    if years is None:
        ax1.set_title('Doanh thu và Số đơn hàng theo tháng (tất cả các năm)')
    else:
        ax1.set_title(f"Doanh thu và Số đơn hàng theo tháng ({', '.join(map(str, years))})")

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig

def top_products(data, year=None, top_n=10, mode="top"):
    """
    Lấy top sản phẩm bán chạy hoặc bán chậm theo số đơn hàng.

    Parameters
    ----------
    data : DataFrame
        Phải có ['Date', 'productName', 'OrderID', 'Gross_sales' ].
    year : int hoặc None
        Năm cần lọc. Nếu None thì lấy toàn bộ.
    top_n : int
        Số sản phẩm muốn lấy.
    mode : str
        "top"  -> sản phẩm bán chạy nhất
        "slow" -> sản phẩm bán chậm nhất
    """
    df = data.copy()
    df['Year'] = df['Date'].dt.year
    if year is not None:
        df = df[df['Year'] == year]

    product_stats = df.groupby('productName').agg({
        'OrderID': 'nunique',     
        'Gross_sales': 'sum'      
    }).reset_index()

    if mode == "top":
        return product_stats.sort_values('OrderID', ascending=False).head(top_n)
    else:
        return product_stats.sort_values('OrderID', ascending=True).head(top_n)
    



    

def create_rfm(data, customer_col='Member_number', 
               date_col='Date', order_col='OrderID', sales_col='Gross_sales'):
    """
    Tạo bảng RFM từ dữ liệu giao dịch.

    Parameters
    ----------
    data : DataFrame
        DataFrame chứa dữ liệu giao dịch, yêu cầu có ít nhất 4 cột:
        - customer_col : mã khách hàng
        - date_col     : ngày giao dịch (kiểu datetime)
        - order_col    : mã đơn hàng
        - sales_col    : doanh thu

    customer_col : str
        Tên cột mã khách hàng (mặc định: 'Member_number').

    date_col : str
        Tên cột ngày giao dịch (mặc định: 'Date').

    order_col : str
        Tên cột mã đơn hàng (mặc định: 'OrderID').

    sales_col : str
        Tên cột doanh thu (mặc định: 'Gross_sales').

    Returns
    -------
    DataFrame
        Bảng RFM gồm 3 cột:
        - Recency  : số ngày kể từ lần mua gần nhất
        - Frequency: số lượng đơn hàng
        - Monetary : tổng doanh thu
    """
    max_date = data[date_col].max().date()

    Recency = lambda x: (max_date - x.max().date()).days
    Frequency = lambda x: len(x.unique())
    Monetary = lambda x: round(sum(x), 2)

    data_RFM = data.groupby(customer_col).agg({
        date_col: Recency,
        order_col: Frequency,
        sales_col: Monetary
    })

    data_RFM.columns = ['Recency', 'Frequency', 'Monetary']

    return data_RFM

def summarize_clusters(data, cluster_col):
    """
    Tính toán đặc trưng trung bình RFM cho từng cụm khách hàng.

    Parameters
    ----------
    data : DataFrame
        Dữ liệu RFM đã có cột 'Cluster'.
    cluster_col : str, optional
        Tên cột cụm (mặc định là 'Cluster').

    Returns
    -------
    DataFrame
        Bảng tóm tắt RFM cho từng cụm với RecencyMean, FrequencyMean, MonetaryMean,
        số lượng khách hàng (Count) và tỷ lệ phần trăm (Percent).
    """
    rfm_agg = data.groupby(cluster_col).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)

    # Làm phẳng MultiIndex columns
    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']

    # Tính phần trăm
    rfm_agg['Percent'] = round((rfm_agg['Count'] / rfm_agg['Count'].sum()) * 100, 2)

    # Reset index và đổi tên cột Cluster
    rfm_agg = rfm_agg.reset_index()
    rfm_agg['Cluster'] = rfm_agg[cluster_col].astype(str)

    return rfm_agg

def assign_segment(cluster):
    """
    Trả về tên nhóm khách hàng dựa vào nhãn cluster.
    """
    if cluster == '0':
        return 'New customers'
    elif cluster == '1':
        return 'At risk'
    elif cluster == '2':
        return 'Potential customers'
    elif cluster == '3':
        return 'Lost'
    elif cluster == '4':
        return 'Champions'
    else:
        return 'Unknown'

import matplotlib.pyplot as plt
import squarify

def plot_rfm_treemap(rfm_agg, colors_dict=None, figsize=(12, 6), fontsize=10):
    """
    Vẽ treemap cho RFM segmentation
    
    Parameters
    ----------
    rfm_agg : DataFrame
        DataFrame phải có các cột:
        ['Segment', 'RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count', 'Percent']
        
    colors_dict : dict hoặc None
        Dictionary ánh xạ Segment -> màu. Nếu None sẽ tự sinh màu.
        Ví dụ: {'New customers':'yellow', 'At risk':'royalblue', ...}
        
    figsize : tuple
        Kích thước figure (mặc định (12,6))
        
    fontsize : int
        Cỡ chữ label trên treemap
    """
    
    # Khởi tạo figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    
    # Nếu không truyền màu thì sinh tự động
    if colors_dict is None:
        import seaborn as sns
        unique_segments = rfm_agg['Segment'].unique()
        palette = sns.color_palette("tab10", len(unique_segments))
        colors_dict = {seg: palette[i] for i, seg in enumerate(unique_segments)}
    
    # Tạo labels
    labels = [
        '{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({:.2f}%)'.format(
            rfm_agg.iloc[i]['Segment'],
            rfm_agg.iloc[i]['RecencyMean'],
            rfm_agg.iloc[i]['FrequencyMean'],
            rfm_agg.iloc[i]['MonetaryMean'],
            rfm_agg.iloc[i]['Count'],
            rfm_agg.iloc[i]['Percent']
        )
        for i in range(len(rfm_agg))
    ]
    
    # Vẽ treemap
    squarify.plot(
        sizes=rfm_agg['Count'],
        label=labels,
        color=[colors_dict[seg] for seg in rfm_agg['Segment']],
        text_kwargs={'fontsize': fontsize, 'weight':'bold', 'fontname':"sans serif"},
        alpha=0.8
    )
    
    plt.title("Customers Segments", fontsize=22, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    return fig
