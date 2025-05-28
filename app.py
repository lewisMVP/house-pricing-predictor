import streamlit as st
import pandas as pd
import numpy as np
import joblib  # hoặc pickle
from sklearn.preprocessing import LabelEncoder

# ---------- PAGE CONFIG & THEME ----------
st.set_page_config(
    page_title="House-Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette pastel nhẹ nhàng
st.markdown("""
<style>
    /* Bảo đảm light-mode cho toàn site */
    :root{color-scheme:light;}

    body, .stApp {background:#fafafa;color:#212529;}
    /* Header */
    .wizard-header   {
        background:#e3f6f5;
        padding:16px 24px;
        border-radius:8px;
        margin-bottom:20px;

        /* NEW – center the text */
        display:flex;
        justify-content:center;
        align-items:center;
    }
    .wizard-header h1{
        color:#006d77;
        font-size:28px;
        margin:0;
        text-align:center;               /* bảo đảm căn giữa khi wrap dòng */
    }
    /* Tabs */
    .stTabs [role=tab]{padding:8px 18px;font-size:15px;color:#006d77;}
    .stTabs [aria-selected="true"]{background:#e3f6f5;color:#006d77;font-weight:700;}
    /* Input & label luôn nền sáng / chữ tối */
    label, .stMarkdown, .stText {color:#212529 !important;}
    input, textarea, select{
        background:#ffffff !important;
        color:#212529 !important;
        border:1px solid #ced4da !important;
    }
    /* NumberInput & Selectbox wrapper */
    div[data-baseweb="input"] input{background:#ffffff !important;color:#212529 !important;}
    div[data-baseweb="select"]{background:#ffffff !important;color:#212529 !important;}
    /* Metric, alert … giữ nguyên */

    /* ------------ BỔ SUNG: ép màu cho metric, alert, result box ------------ */
    div[data-testid="stMetricValue"]  {color:#006d77 !important; font-weight:700 !important; font-size:28px;}
    div[data-testid="stMetricLabel"]  {color:#495057 !important; font-weight:600 !important;}

    /* Kết quả dự đoán trong .result-box */
    .result-box, .result-box h2      {color:#006d77 !important;}

    /* Hộp cảnh báo / thông báo -------------------------------------------------- */
    .stAlertSuccess, .stAlertSuccess *{
        background:#d1e7dd !important;      /* same tone nền */
        color:#0f5132  !important;          /* chữ xanh đậm */
        font-weight:600 !important;
    }

    /* WARNING (st.warning) – nền vàng nhạt, chữ vàng đậm */
    .stAlertWarning, .stAlertWarning *{
        background:#fff3cd !important;
        color:#664d03 !important;
        font-weight:600 !important;
    }

    /* INFO  & ERROR nếu có  */
    .stAlertInfo,    .stAlertInfo    *{color:#0c5460 !important;}
    .stAlertError,   .stAlertError   *{color:#721c24 !important;}

    /* Đảm bảo icon (svg) của alert cũng đổi màu cho đồng bộ */
    .stAlertSuccess svg, .stAlertWarning svg,
    .stAlertInfo svg,    .stAlertError svg{
        fill:currentColor !important;
    }

    /* caption / delta (ví dụ -88.4%)  */
    .stCaption, .caption, span[data-testid="stMetricDelta"]{color:#6c757d !important;}

    /* title & subheader  */
    h1,h2,h3,h4,h5,h6 {color:#212529 !important;}

    /* ===== ÉP LẠI TOÀN BỘ ST.ALERT (success / warning / info / error) ===== */
    /* khung bao ngoài */
    div[data-testid="stAlert"]{
        padding:10px 14px !important;
        border-radius:6px !important;
        font-weight:600 !important;
    }

    /* phần text nằm trong <div role="alert"> */
    div[data-testid="stAlert"] > div[role="alert"],
    div[data-testid="stAlert"] *{
        color:#212529 !important;             /* chữ đen */
        fill:#212529 !important;              /* màu biểu-tượng SVG */
        opacity:1 !important;                 /* huỷ mờ phần con */
        font-weight:600 !important;
    }

    /* SUCCESS – nền xanh nhạt, chữ xanh đậm   */
    div[data-testid="stAlert"][class*="Success"],
    div[data-testid="stAlert"][class*="Success"] > div[role="alert"],
    div[data-testid="stAlert"][class*="Success"] svg{
        background:#d1e7dd !important;
        color:#0f5132   !important;
        fill:#0f5132    !important;
    }

    /* WARNING – nền vàng nhạt, chữ vàng đậm   */
    div[data-testid="stAlert"][class*="Warning"],
    div[data-testid="stAlert"][class*="Warning"] > div[role="alert"],
    div[data-testid="stAlert"][class*="Warning"] svg{
        background:#fff3cd !important;
        color:#664d03   !important;
        fill:#664d03    !important;
    }

    /* INFO + ERROR nếu cần giữ lại             */
    div[data-testid="stAlert"][class*="Info"],
    div[data-testid="stAlert"][class*="Info"] > div[role="alert"],
    div[data-testid="stAlert"][class*="Info"] svg{
        color:#055160 !important; fill:#055160 !important;
    }
    div[data-testid="stAlert"][class*="Error"],
    div[data-testid="stAlert"][class*="Error"] > div[role="alert"],
    div[data-testid="stAlert"][class*="Error"] svg{
        color:#842029 !important; fill:#842029 !important;
    }

    /* ==== ép TẤT CẢ chữ & icon trong stAlert thành đen rõ ==== */
    div[data-testid="stAlert"] {opacity:1 !important;}                  /* khung */
    div[data-testid="stAlert"] * {color:#000 !important; fill:#000 !important; opacity:1 !important;}
    div[data-testid="stAlert"] p,           /* đoạn Markdown trong alert  */
    div[data-testid="stAlert"] span{color:#000 !important;}

    /* ——— ÉP chữ trong mọi st.success / st.warning thành đen đậm ——— */
    div[data-testid="stAlert"] > div[role="alert"]{
        color:#000 !important;        /* chữ đen */
        opacity:1 !important;         /* huỷ mờ */
    }
    div[data-testid="stAlert"] svg{fill:#000 !important;}   /* icon cũng đen */

    /* ===== Thiết kế dịu & bo góc cho bảng ===== */
    div[data-testid="stTable"]{
        background:#f5f7fa !important;           /* nền xám rất nhạt */
        border:1px solid #d0d7de !important;     /* viền xám nhẹ */
        border-radius:8px !important;            /* bo góc */
        overflow:hidden !important;              /* giữ bo góc */
    }

    /* Header */
    div[data-testid="stTable"] thead th{
        background:#e3f6f5 !important;           /* xanh pastel header */
        color:#006d77 !important;
        font-weight:600 !important;
        border:1px solid #d0d7de !important;
        padding:6px 10px !important;
    }

    /* Body: cả ô dữ liệu & cột chỉ số (th) */
    div[data-testid="stTable"] tbody td,
    div[data-testid="stTable"] tbody th{
        background:#ffffff !important;
        color:#212529 !important;                /* chữ đen/xám đậm */
        border:1px solid #d0d7de !important;     /* viền xám nhạt */
        padding:6px 10px !important;
    }

    /* ===== st.dataframe – bảng phân tích theo khu vực (EDA) ===== */
    div[data-testid="stDataFrame"]{
        background:#f5f7fa !important;             /* nền xám nhạt dịu */
        border:1px solid #d0d7de !important;       /* viền ngoài nhạt */
        border-radius:8px !important;              /* bo góc */
        overflow:hidden !important;                /* giữ bo góc */
    }

    /* header của bảng */
    div[data-testid="stDataFrame"] thead tr th{
        background:#e3f6f5 !important;             /* xanh pastel nhẹ */
        color:#006d77 !important;                  /* chữ xanh đậm */
        font-weight:600 !important;
        border:1px solid #d0d7de !important;
        padding:6px 10px !important;
    }

    /* ô dữ liệu & cột chỉ số (index) */
    div[data-testid="stDataFrame"] tbody tr th,
    div[data-testid="stDataFrame"] tbody tr td{
        background:#ffffff !important;
        color:#212529 !important;                  /* chữ đen/xám đậm */
        border:1px solid #d0d7de !important;       /* viền ô nhạt */
        padding:6px 10px !important;
    }

    /* loại bỏ hiệu ứng hover tối của theme gốc */
    div[data-testid="stDataFrame"] tbody tr:hover td,
    div[data-testid="stDataFrame"] tbody tr:hover th{
        background:#f1f3f5 !important;
    }

    /* ===== Rounded card for every st.expander (EDA blocks) ===== */
    div[data-testid="stExpander"]{
        background:#ffffff !important;        /* card background        */
        border:1px solid #d0d7de !important;  /* soft grey border       */
        border-radius:8px !important;         /* rounded corners        */
        overflow:hidden !important;           /* keep radius on header  */
        margin-bottom:14px !important;        /* spacing between cards  */
    }
    /* header bar (clickable) */
    .streamlit-expanderHeader{
        background:#e3f6f5 !important;        /* pastel header colour   */
        color:#006d77 !important;
        padding:10px 14px !important;
        font-weight:600 !important;
        border-bottom:1px solid #d0d7de !important;
    }
    /* inside content area */
    .streamlit-expanderContent{
        background:#ffffff !important;
        padding:12px 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="wizard-header"><h1>Enter Prediction Inputs</h1></div>', unsafe_allow_html=True)

# ADD: Lazy load plotting libraries
@st.cache_resource
def load_plotting_libs():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns

# ADD: Optimize data loading
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("merged_output.csv")
    
    # Quick data type optimization
    for col in df.select_dtypes(include=[np.number]):
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

model = load_model()
df = load_data()

# ---------- INPUT: 3 TABS ----------
tab_coban, tab_vitri, tab_taichinh = st.tabs(
    ["🧱 Basics", "📍 Location", "💰 Finance"]
)

with tab_coban:
    area        = st.number_input("Area (m²)",          10, 1000, 120)
    rooms       = st.slider("Number of Rooms",          1, 10, 3)
    year_build  = st.number_input("Year Built",         1900, 2025, 2000)
    sqm_price   = st.number_input("Price per m²",       0, value=0)

with tab_vitri:
    zipcode     = st.selectbox("Zip Code",           df['zipcode'].unique())
    city        = st.selectbox("City",               df['city'].unique())
    region      = st.selectbox("Region",             df['region'].unique())
    house_type  = st.selectbox("House Type",         df['house_type'].unique())
    sales_type  = st.selectbox("Sale Type",          df['sales_type'].unique())

with tab_taichinh:
    nom_interest_rate            = st.number_input("Mortgage Rate (%)",         value=0.0)
    dk_ann_infl_rate             = st.number_input("Expected Inflation (%)",    value=0.0)
    yield_on_mortgage_credit_bonds = st.number_input("Mortgage-Bond Yield (%)", value=0.0)

# --- Dự đoán ---  
input_data = pd.DataFrame({
    # Convert datetime to numeric features
    'date': [pd.Timestamp.now().year * 12 + pd.Timestamp.now().month],  # Convert to months since epoch
    'quarter': [pd.Timestamp.now().quarter],
    'house_id': [0],
    'house_type': [house_type],
    'sales_type': [sales_type],
    'year_build': [year_build],
    '%_change_between_offer_and_purchase': [0],
    'no_rooms': [rooms],
    'sqm': [area],
    'sqm_price': [sqm_price],
    'address': [0],  # Convert to numeric placeholder
    'zipcode': [zipcode],
    'city': [city],
    'area': [area],
    'region': [region],
    'nom_interest_rate%': [nom_interest_rate],
    'dk_ann_infl_rate%': [dk_ann_infl_rate],
    'yield_on_mortgage_credit_bonds%': [yield_on_mortgage_credit_bonds],
}).astype({
    'date': 'int32',
    'quarter': 'int32',
    'house_id': 'int32',
    'year_build': 'int32',
    '%_change_between_offer_and_purchase': 'float32',
    'no_rooms': 'int32',
    'sqm': 'float32',
    'sqm_price': 'float32',
    'address': 'int32',
    'zipcode': 'int32',
    'area': 'float32',
    'nom_interest_rate%': 'float32',
    'dk_ann_infl_rate%': 'float32',
    'yield_on_mortgage_credit_bonds%': 'float32'
})

# --- Mã hóa các cột chuỗi bằng LabelEncoder ---
# Khởi tạo LabelEncoder
le_house_type = LabelEncoder()
le_sales_type = LabelEncoder()
le_city = LabelEncoder()
le_region = LabelEncoder()

# Fit các LabelEncoder với dữ liệu từ `df` (hoặc dữ liệu tương tự mà bạn có)
le_house_type.fit(df['house_type'])
le_sales_type.fit(df['sales_type'])
le_city.fit(df['city'])
le_region.fit(df['region'])

# Mã hóa các cột chuỗi trong input_data
input_data['house_type'] = le_house_type.transform(input_data['house_type'])
input_data['sales_type'] = le_sales_type.transform(input_data['sales_type'])
input_data['city'] = le_city.transform(input_data['city'])
input_data['region'] = le_region.transform(input_data['region'])

# --- Dự đoán giá nhà ---
predicted_price = model.predict(input_data)[0]

# --- Hiển thị kết quả --- 
st.title("🏠 House-Price Prediction")
st.subheader("Predicted Price:")
st.success(f"💰 {predicted_price:,.0f} DKK")

# --- Thêm phần validation và gợi ý giá ---
st.subheader("💡 Price Analysis & Suggestions")

# Tạo tabs để hiển thị thông tin
tab1, tab2 = st.tabs(["Predicted-Price Review", "Detailed Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Tìm giá trung bình của khu vực
        area_data = df[df['zipcode'] == zipcode]
        if not area_data.empty and 'purchaseprice' in area_data.columns:
            area_avg_price = area_data['purchaseprice'].mean()
            area_min_price = area_data['purchaseprice'].min()
            area_max_price = area_data['purchaseprice'].max()
            area_count = len(area_data)
            
            st.metric(
                label=f"Average Price in {zipcode}", 
                value=f"{area_avg_price:,.0f} DKK",
                delta=f"{((predicted_price - area_avg_price) / area_avg_price * 100):.1f}%" 
            )
            
            # Hiện thị khoảng giá
            st.caption(f"Based on {area_count} transactions in the area")
            st.write(f"Price Range: {area_min_price:,.0f} - {area_max_price:,.0f} DKK")
            
            # Đánh giá mức giá
            if predicted_price < area_min_price:
                st.warning(f"⚠️ Predicted price is below the local minimum ({area_min_price:,.0f} DKK)")
            elif predicted_price > area_max_price:
                st.warning(f"⚠️ Predicted price is above the local maximum ({area_max_price:,.0f} DKK)")
            else:
                price_percentile = len(area_data[area_data['purchaseprice'] <= predicted_price]) / len(area_data) * 100
                st.info(f"✓ Predicted price is within the local price range at {price_percentile:.1f}%")
        else:
            st.info("Not enough transactions in this area for comparison")
    
    with col2:
        # Tìm giá trung bình của nhà tương tự
        similar_houses = df[
            (df['house_type'] == house_type) & 
            (abs(df['sqm'] - area) <= 20) & 
            (abs(df['no_rooms'] - rooms) <= 1)
        ]
        
        if not similar_houses.empty and len(similar_houses) >= 5:
            similar_avg_price = similar_houses['purchaseprice'].mean()
            similar_count = len(similar_houses)
            
            st.metric(
                label=f"Average Price of Similar Houses", 
                value=f"{similar_avg_price:,.0f} DKK",
                delta=f"{((predicted_price - similar_avg_price) / similar_avg_price * 100):.1f}%"
            )
            
            st.caption(f"Based on {similar_count} similar houses")
            
            # Tính toán giá đề xuất
            suggested_price = None
            confidence = "Average"
            
            if len(similar_houses) >= 20 and len(area_data) >= 20:
                # Kết hợp giữa dự đoán, nhà tương tự và giá khu vực
                suggested_price = predicted_price * 0.5 + similar_avg_price * 0.3 + area_avg_price * 0.2
                confidence = "High"
            elif len(similar_houses) >= 10:
                # Kết hợp giữa dự đoán và nhà tương tự
                suggested_price = predicted_price * 0.6 + similar_avg_price * 0.4
                confidence = "High"
            else:
                # Chủ yếu dựa vào dự đoán
                suggested_price = predicted_price * 0.8 + similar_avg_price * 0.2
            
            # Hiển thị khoảng giá đề xuất (±5%)
            st.write("### Suggested Selling Price:")
            min_suggested = suggested_price * 0.95
            max_suggested = suggested_price * 1.05
            
            st.success(f"💰 {min_suggested:,.0f} - {max_suggested:,.0f} DKK")
            st.caption(f"Confidence: {confidence}")
        else:
            st.info("Not enough similar houses for comparison")
    
    # Hiển thị biểu đồ so sánh
    if 'area_avg_price' in locals() and 'similar_avg_price' in locals():
        st.subheader("Price Comparison")
        comparison_data = {
            'Type': ['Predicted Price', 'Average Area Price', 'Similar Houses'],
            'Price (DKK)': [predicted_price, area_avg_price, similar_avg_price]
        }
        
        if 'suggested_price' in locals():
            comparison_data['Type'].append('Suggested Price')
            comparison_data['Price (DKK)'].append(suggested_price)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # ADD: Load plotting libs when needed
        plt, sns = load_plotting_libs()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = sns.barplot(x='Type', y='Price (DKK)', data=comparison_df, palette='viridis', ax=ax)
        
        # Thêm giá trị lên các thanh
        for i, bar in enumerate(bars.patches):
            bars.annotate(f'{comparison_df["Price (DKK)"][i]:,.0f}',
                         (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                         ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.title('Comparing Price Levels')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # ADD: Free memory

with tab2:
    st.write("### Detailed Analysis")
    
    # Tạo bảng đánh giá chi tiết
    eval_data = []
    
    # Đánh giá diện tích
    area_percentile = np.percentile(df['area'], [25, 50, 75])
    if area <= area_percentile[0]:
        area_eval = ("Small", "Area is below 25% of properties")
    elif area <= area_percentile[1]:
        area_eval = ("Low Average", "Area is between 25-50% of properties")
    elif area <= area_percentile[2]:
        area_eval = ("High Average", "Area is between 50-75% of properties")
    else:
        area_eval = ("Large", "Area is above 75% of properties")
    
    eval_data.append(["Area", f"{area} m²", area_eval[0], area_eval[1]])
    
    # Đánh giá số phòng
    room_counts = df['no_rooms'].value_counts()
    if rooms in room_counts.index:
        room_pct = room_counts[rooms] / room_counts.sum() * 100
        room_eval = (f"Common ({room_pct:.1f}%)", f"{room_pct:.1f}% of properties have {rooms} rooms")
    else:
        room_eval = ("Uncommon", f"Very few properties have {rooms} rooms")
    
    eval_data.append(["Number of Rooms", f"{rooms}", room_eval[0], room_eval[1]])
    
    # Đánh giá khu vực (zipcode)
    zipcode_counts = df['zipcode'].value_counts()
    if zipcode in zipcode_counts.index:
        zipcode_pct = zipcode_counts[zipcode] / zipcode_counts.sum() * 100
        zipcode_eval = (f"Data Available ({zipcode_pct:.1f}%)", f"{zipcode_counts[zipcode]} transactions ({zipcode_pct:.1f}%)")
    else:
        zipcode_eval = ("Data Missing", "No transactions in this area")
    
    eval_data.append(["Zip Code", f"{zipcode}", zipcode_eval[0], zipcode_eval[1]])
    
    # Đánh giá loại nhà
    house_type_counts = df['house_type'].value_counts()
    if house_type in house_type_counts.index:
        house_type_pct = house_type_counts[house_type] / house_type_counts.sum() * 100
        house_eval = (f"Common ({house_type_pct:.1f}%)", f"{house_type_pct:.1f}% of this house type")
    else:
        house_eval = ("Uncommon", "Very few data about this house type")
    
    eval_data.append(["House Type", f"{house_type}", house_eval[0], house_eval[1]])
    
    # Đánh giá năm xây dựng
    year_percentile = np.percentile(df['year_build'], [25, 50, 75])
    if year_build <= year_percentile[0]:
        year_eval = ("Old", "Oldest 25% of properties")
    elif year_build <= year_percentile[1]:
        year_eval = ("Average Old", "Properties between 25-50% of age")
    elif year_build <= year_percentile[2]:
        year_eval = ("Average New", "Properties between 50-75% of age")
    else:
        year_eval = ("New", "Newest 25% of properties")
    
    eval_data.append(["Year Built", f"{year_build}", year_eval[0], year_eval[1]])
    
    # Hiển thị bảng đánh giá
    eval_df = pd.DataFrame(eval_data, columns=["Parameter", "Value", "Rating", "Details"])
    st.table(eval_df)
    
    # Tính điểm độ tin cậy của dự đoán
    reliability_score = 0
    max_score = 5  # Tổng điểm tối đa
    
    # Điểm cho dữ liệu khu vực
    if 'area_data' in locals() and len(area_data) >= 20:
        reliability_score += 1
    elif 'area_data' in locals() and len(area_data) >= 5:
        reliability_score += 0.5
    
    # Điểm cho dữ liệu nhà tương tự
    if 'similar_houses' in locals() and len(similar_houses) >= 20:
        reliability_score += 1
    elif 'similar_houses' in locals() and len(similar_houses) >= 5:
        reliability_score += 0.5
    
    # Điểm cho tham số diện tích
    if area_percentile[0] <= area <= area_percentile[2]:
        reliability_score += 1  # Diện tích trong khoảng phổ biến
    
    # Điểm cho số phòng
    common_rooms = room_counts.nlargest(3).index.tolist()
    if rooms in common_rooms:
        reliability_score += 1  # Số phòng phổ biến
    
    # Điểm cho năm xây dựng
    current_year = pd.Timestamp.now().year
    if 1950 <= year_build <= current_year:
        reliability_score += 1  # Năm xây dựng hợp lý
    
    # Hiển thị điểm độ tin cậy
    reliability_percentage = (reliability_score / max_score) * 100
    st.subheader("Prediction Reliability")
    st.progress(reliability_score / max_score)
    
    if reliability_percentage >= 80:
        st.success(f"✅ High Confidence: {reliability_percentage:.0f}%")
    elif reliability_percentage >= 50:
        st.warning(f"⚠️ Average Confidence: {reliability_percentage:.0f}%")
    else:
        st.error(f"❌ Low Confidence: {reliability_percentage:.0f}%")
    
    # Lý do đánh giá độ tin cậy
    st.write("**Factors Affecting Reliability:**")
    reasons = []
    
    if 'area_data' in locals():
        reasons.append(f"- Number of transactions in the area: {len(area_data)}")
    else:
        reasons.append("- No transaction data in the area")
    
    if 'similar_houses' in locals():
        reasons.append(f"- Number of similar houses: {len(similar_houses)}")
    else:
        reasons.append("- No data about similar houses")
    
    if area_percentile[0] <= area <= area_percentile[2]:
        reasons.append("- Area is within the common range")
    else:
        reasons.append("- Area is outside the common range")
    
    if rooms in common_rooms:
        reasons.append(f"- Number of rooms ({rooms}) is common")
    else:
        reasons.append(f"- Number of rooms ({rooms}) is uncommon")
    
    for reason in reasons:
        st.write(reason)

# --- Biểu đồ EDA ---
st.subheader("Exploratory Data Analysis (EDA)")

with st.expander("Correlation Heatmap"):
    # ADD: Load plotting libs only when needed
    plt, sns = load_plotting_libs()
    
    corr = df.corr(numeric_only=True)
    if 'purchaseprice' in corr.columns:
        # OPTIMIZE: Limit to top 10 instead of all
        top_corr = corr['purchaseprice'].abs().sort_values(ascending=False)[1:11].index
        selected_corr = corr.loc[top_corr, top_corr]
    else:
        selected_corr = corr
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(selected_corr, annot=True, cmap="coolwarm", linewidths=0.5, square=True, fmt=".2f")
    ax.set_title("Heatmap of Correlated Variables")
    st.pyplot(fig)
    plt.close(fig)  # ADD: Free memory

with st.expander("Area Distribution"):
    plt, sns = load_plotting_libs()  # ADD
    fig2, ax2 = plt.subplots()
    # OPTIMIZE: Sample data if too large
    if len(df) > 5000:
        sample_data = df['area'].sample(5000, random_state=42)
        sns.histplot(sample_data, bins=30, kde=True, ax=ax2)
        ax2.set_title("Area Distribution (5000 samples)")
    else:
        sns.histplot(df['area'], bins=30, kde=True, ax=ax2)
        ax2.set_title("Area Distribution")
    st.pyplot(fig2)
    plt.close(fig2)  # ADD: Free memory

with st.expander("Price Analysis by Zip Code"):
    # OPTIMIZE: Limit to top 20 zip codes
    top_zipcodes = df['zipcode'].value_counts().head(20).index
    analysis_data = df[df['zipcode'].isin(top_zipcodes)]
    
    price_analysis = analysis_data.groupby('zipcode').agg({
        'purchaseprice': ['mean', 'count']
    }).round(2)
    
    price_analysis.columns = ['Average Price', 'Transaction Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 20 Zip Codes Analysis:")
        st.dataframe(price_analysis)
    
    with col2:
        st.write("Average Price by Zip Code:")
        
        # Use matplotlib for reliable chart display
        plt, sns = load_plotting_libs()
        
        # Prepare data
        chart_data = price_analysis.reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(chart_data)), chart_data['Average Price'], 
                marker='o', linewidth=2, markersize=6, color='#1f77b4')
        
        # Customize chart
        ax.set_xticks(range(len(chart_data)))
        ax.set_xticklabels(chart_data['zipcode'], rotation=45)
        ax.set_title("Average Price by Zip Code", fontsize=14, pad=20)
        ax.set_xlabel("Zip Code")
        ax.set_ylabel("Average Price (DKK)")
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show prices nicely
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------- USER GUIDE -----------------------------------------------------
st.markdown("""
### 📋 User Guide:
1. **Enter the information** above to predict a house price  
2. **See the predicted result** and detailed analysis  
3. **Review** the charts and data below  
4. **Compare** the prediction with the local market  

*Note: Predictions are for reference only and depend on input data quality.*
""")