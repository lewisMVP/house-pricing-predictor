import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import shap 

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
    
    # ADD: Price Range Budget
    st.write("---")
    st.write("#### 💰 Budget & Price Range")
    
    # Get price range from data for reasonable defaults
    min_price = int(df['purchaseprice'].min()) if 'purchaseprice' in df.columns else 500000
    max_price = int(df['purchaseprice'].max()) if 'purchaseprice' in df.columns else 10000000
    median_price = int(df['purchaseprice'].median()) if 'purchaseprice' in df.columns else 2000000
    
    # Price range slider
    price_range = st.slider(
        "Expected Price Range (DKK)",
        min_value=min_price,
        max_value=max_price,
        value=(median_price - 500000, median_price + 500000),
        step=50000,
        format="%d",
        help="Set your budget range to compare with prediction and find similar houses"
    )
    
    st.caption(f"💰 **Your Budget**: {price_range[0]:,} - {price_range[1]:,} DKK")

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

# --- GIẢI PHÁP ĐƠN GIẢN - KHÔNG DÙNG SESSION STATE ---
# Mã hóa các cột chuỗi trong input_data
input_data['house_type'] = le_house_type.transform(input_data['house_type'])
input_data['sales_type'] = le_sales_type.transform(input_data['sales_type'])
input_data['city'] = le_city.transform(input_data['city'])
input_data['region'] = le_region.transform(input_data['region'])

# TÍNH TOÁN PREDICTED_PRICE MỘT LẦN DUY NHẤT
predicted_price = float(model.predict(input_data)[0])
price_per_sqm = predicted_price / area

st.markdown('<div class="result-box">', unsafe_allow_html=True)
st.subheader("🏠 House Price Prediction")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.metric(
        label="Predicted House Price",
        value=f"{predicted_price:,.0f} DKK",
        help="Machine learning prediction based on your inputs"
    )

with col2:
    st.metric(
        label="Price per m²",
        value=f"{price_per_sqm:,.0f} DKK/m²"
    )

with col3:
    if area > 0 and rooms > 0:
        st.metric(
            label="Prediction Quality",
            value="Good" if 50 <= area <= 300 and 1 <= rooms <= 8 else "Fair"
        )

st.markdown('</div>', unsafe_allow_html=True)

# THÊM: Quick SHAP explanation
with st.expander("🔍 Why this price? (Quick Explanation)"):
    @st.cache_data
    def get_quick_shap_explanation():
        try:
            # Create explainer for current prediction
            explainer = shap.TreeExplainer(model)
            
            # Get SHAP values for current input
            shap_values = explainer.shap_values(input_data)
            
            # Get feature names
            feature_names = input_data.columns.tolist()
            
            # Create quick explanation
            explanation_data = []
            for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values[0])):
                if abs(shap_val) > 1000:  # Only show significant impacts
                    explanation_data.append({
                        'Feature': feature,
                        'Impact': shap_val,
                        'Direction': 'Increases Price' if shap_val > 0 else 'Decreases Price'
                    })
            
            # Sort by absolute impact
            explanation_data.sort(key=lambda x: abs(x['Impact']), reverse=True)
            
            return explanation_data[:5], explainer.expected_value  # Top 5 features
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    quick_explanation, base_value = get_quick_shap_explanation()
    
    if quick_explanation:
        st.write(f"**Base price (market average)**: {base_value:,.0f} DKK")
        st.write("**Key factors affecting your prediction**:")
        
        for item in quick_explanation:
            impact_color = "🟢" if item['Impact'] > 0 else "🔴"
            st.write(f"{impact_color} **{item['Feature']}**: {item['Impact']:+,.0f} DKK ({item['Direction']})")
        
        st.caption("💡 See 'Model Evaluation' section for detailed SHAP analysis")
    else:
        st.error(f"Could not generate explanation: {base_value}")

# --- Price analysis and suggestions ---
st.subheader("💡 Price Analysis & Suggestions")

budget_min, budget_max = price_range

if budget_min <= predicted_price <= budget_max:
    st.success(f"✅ **Within Budget!** Predicted price ({predicted_price:,.0f} DKK) fits your budget range ({budget_min:,.0f} - {budget_max:,.0f} DKK)")
elif predicted_price < budget_min:
    diff = budget_min - predicted_price
    st.info(f"💰 **Below Budget!** Predicted price is {diff:,.0f} DKK below your minimum budget. You might find better options or save money!")
else:
    diff = predicted_price - budget_max
    st.warning(f"⚠️ **Over Budget!** Predicted price exceeds your budget by {diff:,.0f} DKK. Consider adjusting expectations or increasing budget.")

# Tạo tabs để hiển thị thông tin
tab1, tab2 = st.tabs(["Predicted-Price Review", "Detailed Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Tìm giá trung bình của khu vực
        area_data = df[df['zipcode'] == zipcode]
        # ADD: Filter by price range
        area_data_in_budget = area_data[
            (area_data['purchaseprice'] >= budget_min) & 
            (area_data['purchaseprice'] <= budget_max)
        ] if 'purchaseprice' in area_data.columns else area_data
        
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
            
            # ADD: Budget-filtered info
            if len(area_data_in_budget) > 0:
                st.caption(f"📊 Based on {area_count} total transactions")
                budget_avg = area_data_in_budget['purchaseprice'].mean()
                budget_count = len(area_data_in_budget)
                st.success(f"🎯 **{budget_count} houses in your budget** (avg: {budget_avg:,.0f} DKK)")
            else:
                st.caption(f"📊 Based on {area_count} transactions")
                st.warning("⚠️ **No houses in your budget range** in this area")
            
            # Hiện thị khoảng giá
            st.write(f"**Price Range**: {area_min_price:,.0f} - {area_max_price:,.0f} DKK")
            
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
        # Tìm giá trung bình của nhà tương tự - WITH BUDGET FILTER
        similar_houses = df[
            (df['house_type'] == house_type) & 
            (abs(df['sqm'] - area) <= 20) & 
            (abs(df['no_rooms'] - rooms) <= 1)
        ]
        
        # ADD: Budget-filtered similar houses
        similar_houses_in_budget = similar_houses[
            (similar_houses['purchaseprice'] >= budget_min) & 
            (similar_houses['purchaseprice'] <= budget_max)
        ] if 'purchaseprice' in similar_houses.columns else similar_houses
        
        if not similar_houses.empty and len(similar_houses) >= 5:
            similar_avg_price = similar_houses['purchaseprice'].mean()
            similar_count = len(similar_houses)
            
            st.metric(
                label=f"Average Price of Similar Houses", 
                value=f"{similar_avg_price:,.0f} DKK",
                delta=f"{((predicted_price - similar_avg_price) / similar_avg_price * 100):.1f}%"
            )
            
            st.caption(f"📊 Based on {similar_count} similar houses")
            
            # ADD: Budget info for similar houses
            if len(similar_houses_in_budget) > 0:
                budget_similar_avg = similar_houses_in_budget['purchaseprice'].mean()
                budget_similar_count = len(similar_houses_in_budget)
                st.success(f"🎯 **{budget_similar_count} similar houses in budget** (avg: {budget_similar_avg:,.0f} DKK)")
            else:
                st.warning("⚠️ **No similar houses in your budget range**")
            
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
            
            # ADD: Budget comparison for suggested price
            if budget_min <= suggested_price <= budget_max:
                st.success(f"💰 {min_suggested:,.0f} - {max_suggested:,.0f} DKK ✅")
                st.caption(f"Confidence: {confidence} | **Fits your budget!**")
            else:
                st.success(f"💰 {min_suggested:,.0f} - {max_suggested:,.0f} DKK")
                st.caption(f"Confidence: {confidence} | ⚠️ **Outside your budget range**")
        else:
            st.info("Not enough similar houses for comparison")
    
    # Hiển thị biểu đồ so sánh - WITH BUDGET VISUALIZATION
    if 'area_avg_price' in locals() and 'similar_avg_price' in locals():
        st.subheader("Price Comparison")
        comparison_data = {
            'Type': ['Predicted Price', 'Average Area Price', 'Similar Houses'],
            'Price (DKK)': [predicted_price, area_avg_price, similar_avg_price]
        }
        
        if 'suggested_price' in locals():
            comparison_data['Type'].append('Suggested Price')
            comparison_data['Price (DKK)'].append(suggested_price)
        
        # ADD: Budget range to comparison
        comparison_data['Type'].extend(['Budget Min', 'Budget Max'])
        comparison_data['Price (DKK)'].extend([budget_min, budget_max])
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # ADD: Load plotting libs when needed
        plt, sns = load_plotting_libs()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create colors: budget range in different color
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax.bar(comparison_df['Type'], comparison_df['Price (DKK)'], color=colors[:len(comparison_df)])
        
        # Highlight budget range
        for i, bar in enumerate(bars):
            if comparison_df['Type'].iloc[i] in ['Budget Min', 'Budget Max']:
                bar.set_alpha(0.5)
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
        
        # Thêm giá trị lên các thanh
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}',
                       (bar.get_x() + bar.get_width()/2, height), 
                       ha='center', va='bottom', fontsize=9, rotation=0)
        
        ax.set_title('Price Comparison with Budget Range', fontsize=14)
        ax.set_ylabel('Price (DKK)')
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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

# KEEP: Original EDA features first
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

# NEW: Price Range Market Analysis - ADD AFTER existing EDA features
with st.expander("🏷️ Price Range Market Analysis"):
    st.write("### Explore Market by Price Range")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price range for analysis (can be different from budget)
        analysis_price_range = st.slider(
            "Analysis Price Range (DKK)",
            min_value=min_price,
            max_value=max_price,
            value=(price_range[0], price_range[1]),  # Default to budget range
            step=100000,
            key="analysis_price_range",
            help="Filter market analysis by price range"
        )
    
    with col2:
        st.write("**Quick Filters:**")
        if st.button("Use My Budget", help="Set to your budget range"):
            analysis_price_range = price_range
        if st.button("Market Overview", help="Set to full market range"):
            analysis_price_range = (min_price, max_price)
    
    # Filter data by price range
    filtered_df = df[
        (df['purchaseprice'] >= analysis_price_range[0]) & 
        (df['purchaseprice'] <= analysis_price_range[1])
    ] if 'purchaseprice' in df.columns else df
    
    if len(filtered_df) > 0:
        # Market overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Houses in Range", f"{len(filtered_df):,}")
            
        with col2:
            avg_price_in_range = filtered_df['purchaseprice'].mean()
            market_avg = df['purchaseprice'].mean()
            delta_pct = ((avg_price_in_range - market_avg) / market_avg) * 100
            st.metric("Average Price", f"{avg_price_in_range:,.0f} DKK", 
                     delta=f"{delta_pct:+.1f}% vs market")
            
        with col3:
            avg_sqm_in_range = filtered_df['sqm'].mean()
            st.metric("Average Area", f"{avg_sqm_in_range:.0f} m²")
            
        with col4:
            avg_price_per_sqm = avg_price_in_range / avg_sqm_in_range
            st.metric("Price per m²", f"{avg_price_per_sqm:,.0f} DKK")
        
        # Market insights
        st.write("### 📊 Market Insights in Your Price Range")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # Top locations in price range
            st.write("**🏘️ Top Locations:**")
            location_analysis = filtered_df.groupby('city').agg({
                'purchaseprice': ['count', 'mean']
            }).round(0)
            location_analysis.columns = ['Count', 'Avg Price']
            location_analysis = location_analysis.sort_values('Count', ascending=False).head(5)
            st.dataframe(location_analysis)
        
        with insight_col2:
            # House types distribution
            st.write("**🏠 House Types Distribution:**")
            house_type_dist = filtered_df['house_type'].value_counts().head(5)
            st.bar_chart(house_type_dist)
        
        # Visual analysis
        st.write("### 📈 Visual Analysis")
        
        plt, sns = load_plotting_libs()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Price distribution
        sns.histplot(filtered_df['purchaseprice'], bins=30, ax=ax1, color='skyblue')
        ax1.axvline(predicted_price, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {predicted_price:,.0f}')
        ax1.set_title("Price Distribution in Range")
        ax1.set_xlabel("Price (DKK)")
        ax1.legend()
        
        # 2. Area vs Price scatter (sample for performance)
        sample_size = min(1000, len(filtered_df))
        sample_data = filtered_df.sample(sample_size, random_state=42)
        
        scatter = ax2.scatter(sample_data['sqm'], sample_data['purchaseprice'], 
                             alpha=0.6, c=sample_data['no_rooms'], cmap='viridis')
        ax2.scatter(area, predicted_price, color='red', s=100, marker='*', 
                   label=f'Your House ({area}m², {predicted_price:,.0f} DKK)')
        ax2.set_xlabel("Area (m²)")
        ax2.set_ylabel("Price (DKK)")
        ax2.set_title("Area vs Price (colored by rooms)")
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Number of Rooms')
        
        # 3. Price by year built
        year_price = filtered_df.groupby('year_build')['purchaseprice'].mean()
        year_price.plot(ax=ax3, marker='o', color='green')
        ax3.axvline(year_build, color='red', linestyle='--', alpha=0.7, label=f'Your House: {year_build}')
        ax3.set_title("Price by Year Built")
        ax3.set_xlabel("Year Built")
        ax3.set_ylabel("Average Price (DKK)")
        ax3.legend()
        
        # 4. Rooms distribution
        rooms_dist = filtered_df['no_rooms'].value_counts().sort_index()
        bars = ax4.bar(rooms_dist.index, rooms_dist.values, color='orange', alpha=0.7)
        
        # Highlight user's room count
        if rooms in rooms_dist.index:
            idx = list(rooms_dist.index).index(rooms)
            bars[idx].set_color('red')
            bars[idx].set_alpha(1.0)
        
        ax4.set_title("Number of Rooms Distribution")
        ax4.set_xlabel("Number of Rooms")
        ax4.set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Market recommendation
        st.write("### 💡 Market Recommendations")
        
        # Compare user house with market
        user_price_per_sqm = predicted_price / area
        market_price_per_sqm = avg_price_per_sqm
        
        if user_price_per_sqm > market_price_per_sqm * 1.1:
            st.warning(f"⚠️ **Above Market**: Your price per m² ({user_price_per_sqm:,.0f}) is {((user_price_per_sqm/market_price_per_sqm-1)*100):.1f}% above market average.")
        elif user_price_per_sqm < market_price_per_sqm * 0.9:
            st.success(f"💰 **Below Market**: Your price per m² ({user_price_per_sqm:,.0f}) is {((1-user_price_per_sqm/market_price_per_sqm)*100):.1f}% below market average. Good value!")
        else:
            st.info(f"✅ **Market Rate**: Your price per m² ({user_price_per_sqm:,.0f}) is aligned with market average ({market_price_per_sqm:,.0f}).")
            
    else:
        st.warning("🔍 **No houses found** in the selected price range. Try expanding the range.")
        
        # Suggest nearby ranges
        st.write("**💡 Suggestions:**")
        nearby_low = analysis_price_range[0] - 500000
        nearby_high = analysis_price_range[1] + 500000
        
        if nearby_low > 0:
            nearby_count_low = len(df[df['purchaseprice'] < analysis_price_range[0]])
            st.write(f"- **Lower range** ({nearby_low:,} - {analysis_price_range[0]:,.0f}): {nearby_count_low} houses")
            
        nearby_count_high = len(df[df['purchaseprice'] > analysis_price_range[1]])
        st.write(f"- **Higher range** ({analysis_price_range[1]:,.0f} - {nearby_high:,}): {nearby_count_high} houses")

# ---------- MODEL EVALUATION & TUNING ----------
# MOVE: Di chuyển phần này XUỐNG sau EDA
st.subheader("🔬 Model Performance & Tuning")

# Tạo tabs cho Model Evaluation và Hyperparameter Tuning
eval_tab, tuning_tab = st.tabs(["📊 Model Evaluation", "⚙️ Hyperparameter Tuning"])

with eval_tab:
    st.write("### Model Evaluation Metrics")
    
    @st.cache_data
    def evaluate_model():
        """Evaluate model performance on test data"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # Prepare data for evaluation
        feature_columns = [
            'date', 'quarter', 'house_id', 'house_type', 'sales_type', 
            'year_build', '%_change_between_offer_and_purchase', 'no_rooms', 
            'sqm', 'sqm_price', 'address', 'zipcode', 'city', 'area', 'region',
            'nom_interest_rate%', 'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%'
        ]
        
        # Check if we have target column
        if 'purchaseprice' not in df.columns:
            return None, "No target variable (purchaseprice) found in dataset"
        
        # Prepare features
        eval_df = df.copy()
        
        # Convert datetime features
        eval_df['date'] = pd.to_datetime(eval_df['date'], errors='coerce')
        eval_df['date'] = eval_df['date'].dt.year * 12 + eval_df['date'].dt.month
        eval_df['quarter'] = pd.to_datetime(eval_df['date'], errors='coerce').dt.quarter
        
        # Fill missing values
        eval_df = eval_df.fillna(0)
        
        # Encode categorical variables
        le_house_type_eval = LabelEncoder()
        le_sales_type_eval = LabelEncoder()
        le_city_eval = LabelEncoder()
        le_region_eval = LabelEncoder()
        
        eval_df['house_type'] = le_house_type_eval.fit_transform(eval_df['house_type'].astype(str))
        eval_df['sales_type'] = le_sales_type_eval.fit_transform(eval_df['sales_type'].astype(str))
        eval_df['city'] = le_city_eval.fit_transform(eval_df['city'].astype(str))
        eval_df['region'] = le_region_eval.fit_transform(eval_df['region'].astype(str))
        
        # Select features and target
        available_features = [col for col in feature_columns if col in eval_df.columns]
        X = eval_df[available_features]
        y = eval_df['purchaseprice']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Test_Size': len(X_test),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return metrics, None
    
    # Evaluate model
    with st.spinner("Evaluating model performance..."):
        metrics, error = evaluate_model()
    
    if error:
        st.error(f"Error evaluating model: {error}")
    else:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"{metrics['MAE']:,.0f} DKK",
                help="Average absolute difference between predicted and actual prices"
            )
        
        with col2:
            st.metric(
                label="RMSE (Root Mean Square Error)",
                value=f"{metrics['RMSE']:,.0f} DKK",
                help="Root mean square error - penalizes larger errors more"
            )
        
        with col3:
            st.metric(
                label="R² (R-squared)",
                value=f"{metrics['R²']:.3f}",
                help="Coefficient of determination - how well the model explains variance"
            )
        
        with col4:
            st.metric(
                label="MAPE (%)",
                value=f"{metrics['MAPE']:.1f}%",
                help="Mean Absolute Percentage Error"
            )
        
        # Performance interpretation
        st.write("### Performance Interpretation")
        
        if metrics['R²'] >= 0.8:
            st.success("🎯 **Excellent Performance**: R² ≥ 0.8 indicates the model explains most of the variance in house prices.")
        elif metrics['R²'] >= 0.6:
            st.info("👍 **Good Performance**: R² ≥ 0.6 indicates the model performs reasonably well.")
        elif metrics['R²'] >= 0.4:
            st.warning("⚠️ **Average Performance**: R² ≥ 0.4 indicates moderate predictive power.")
        else:
            st.error("❌ **Poor Performance**: R² < 0.4 indicates the model needs improvement.")
        
        # Prediction vs Actual scatter plot
        st.write("### Prediction vs Actual Prices")
        
        plt, sns = load_plotting_libs()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        sample_size = min(1000, len(metrics['y_test']))
        indices = np.random.choice(len(metrics['y_test']), sample_size, replace=False)
        
        y_test_sample = metrics['y_test'].iloc[indices]
        y_pred_sample = metrics['y_pred'][indices]
        
        ax1.scatter(y_test_sample, y_pred_sample, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_test_sample.min(), y_pred_sample.min())
        max_val = max(y_test_sample.max(), y_pred_sample.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Price (DKK)')
        ax1.set_ylabel('Predicted Price (DKK)')
        ax1.set_title(f'Predicted vs Actual Prices (R² = {metrics["R²"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_test_sample - y_pred_sample
        ax2.scatter(y_pred_sample, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Price (DKK)')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ========== THÊM PHẦN SHAP ANALYSIS ==========
        st.write("### 🔍 SHAP Model Explainability")
        st.info("SHAP (SHapley Additive exPlanations) helps understand which features most influence the model's predictions.")
        
        # Create tabs for different SHAP visualizations
        shap_tab1, shap_tab2, shap_tab3, shap_tab4 = st.tabs([
            "🎯 Current Prediction", 
            "📊 Feature Importance", 
            "🌊 Summary Plot", 
            "🔬 Dependence Plot"
        ])
        
        @st.cache_data
        def calculate_shap_values():
            """Calculate SHAP values for model interpretation"""
            try:
                # Prepare data for SHAP (same as evaluation)
                feature_columns = [
                    'date', 'quarter', 'house_id', 'house_type', 'sales_type', 
                    'year_build', '%_change_between_offer_and_purchase', 'no_rooms', 
                    'sqm', 'sqm_price', 'address', 'zipcode', 'city', 'area', 'region',
                    'nom_interest_rate%', 'dk_ann_infl_rate%', 'yield_on_mortgage_credit_bonds%'
                ]
                
                eval_df = df.copy()
                
                # Convert datetime features
                eval_df['date'] = pd.to_datetime(eval_df['date'], errors='coerce')
                eval_df['date'] = eval_df['date'].dt.year * 12 + eval_df['date'].dt.month
                eval_df['quarter'] = eval_df['date'] % 4 + 1
                
                # Fill missing values
                eval_df = eval_df.fillna(0)
                
                # Encode categorical variables
                le_house_type_shap = LabelEncoder()
                le_sales_type_shap = LabelEncoder()
                le_city_shap = LabelEncoder()
                le_region_shap = LabelEncoder()
                
                eval_df['house_type'] = le_house_type_shap.fit_transform(eval_df['house_type'].astype(str))
                eval_df['sales_type'] = le_sales_type_shap.fit_transform(eval_df['sales_type'].astype(str))
                eval_df['city'] = le_city_shap.fit_transform(eval_df['city'].astype(str))
                eval_df['region'] = le_region_shap.fit_transform(eval_df['region'].astype(str))
                
                # Select features
                available_features = [col for col in feature_columns if col in eval_df.columns]
                X_shap = eval_df[available_features]
                
                # Sample data for performance (SHAP can be slow on large datasets)
                if len(X_shap) > 1000:
                    X_sample = X_shap.sample(1000, random_state=42)
                else:
                    X_sample = X_shap
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values for sample
                shap_values = explainer.shap_values(X_sample)
                
                # Get current prediction SHAP values
                current_shap_values = explainer.shap_values(input_data[available_features])
                
                return {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'X_sample': X_sample,
                    'current_shap': current_shap_values,
                    'current_input': input_data[available_features],
                    'feature_names': available_features,
                    'base_value': explainer.expected_value
                }
                
            except Exception as e:
                return {'error': f"SHAP calculation failed: {str(e)}"}
        
        # Calculate SHAP values
        with st.spinner("Calculating SHAP values... This may take a moment."):
            shap_data = calculate_shap_values()
        
        if 'error' in shap_data:
            st.error(shap_data['error'])
            st.info("💡 SHAP analysis requires the model to be compatible and properly configured.")
        else:
            
            # Tab 1: Current Prediction Explanation
            with shap_tab1:
                st.write("#### Explanation for Your Current Prediction")
                st.write(f"**Predicted Price**: {predicted_price:,.0f} DKK")
                st.write(f"**Base Value (Average)**: {shap_data['base_value']:,.0f} DKK")
                
                # Current prediction SHAP values
                current_shap = shap_data['current_shap'][0]
                feature_names = shap_data['feature_names']
                
                # Create explanation dataframe
                explanation_data = []
                for i, (feature, shap_val) in enumerate(zip(feature_names, current_shap)):
                    feature_value = shap_data['current_input'].iloc[0, i]
                    explanation_data.append({
                        'Feature': feature,
                        'Value': feature_value,
                        'SHAP Impact': shap_val,
                        'Impact (DKK)': shap_val,
                        'Contribution': 'Increases Price' if shap_val > 0 else 'Decreases Price'
                    })
                
                explanation_df = pd.DataFrame(explanation_data)
                explanation_df = explanation_df.reindex(
                    explanation_df['SHAP Impact'].abs().sort_values(ascending=False).index
                )
                
                # Display top contributing features
                st.write("**Top 10 Most Influential Features:**")
                top_features = explanation_df.head(10)
                
                # Create a horizontal bar chart for SHAP values
                plt, sns = load_plotting_libs()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = ['green' if x > 0 else 'red' for x in top_features['SHAP Impact']]
                bars = ax.barh(range(len(top_features)), top_features['SHAP Impact'], color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['Feature'])
                ax.set_xlabel('SHAP Impact on Price (DKK)')
                ax.set_title('Feature Impact on Your Prediction')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, top_features['SHAP Impact'])):
                    ax.text(val + (max(top_features['SHAP Impact']) * 0.01), bar.get_y() + bar.get_height()/2, 
                           f'{val:,.0f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display detailed table
                st.write("**Detailed Feature Contributions:**")
                display_df = top_features[['Feature', 'Value', 'Impact (DKK)', 'Contribution']].copy()
                display_df['Impact (DKK)'] = display_df['Impact (DKK)'].apply(lambda x: f"{x:+,.0f}")
                st.dataframe(display_df, use_container_width=True)
                
                # Calculation verification
                total_shap = current_shap.sum()
                calculated_prediction = shap_data['base_value'] + total_shap
                st.write("**Verification:**")
                st.write(f"Base Value: {shap_data['base_value']:,.0f} DKK")
                st.write(f"+ Total Feature Contributions: {total_shap:+,.0f} DKK")
                st.write(f"= Calculated Prediction: {calculated_prediction:,.0f} DKK")
                st.write(f"Model Prediction: {predicted_price:,.0f} DKK")
                
                if abs(calculated_prediction - predicted_price) < 1:
                    st.success("✅ SHAP explanation matches model prediction perfectly!")
                else:
                    st.info(f"ℹ️ Small difference ({abs(calculated_prediction - predicted_price):,.0f} DKK) due to rounding.")
            
            # Tab 2: Global Feature Importance
            with shap_tab2:
                st.write("#### Global Feature Importance")
                st.write("Shows the average impact of each feature across all predictions in the dataset.")
                
                # Calculate mean absolute SHAP values for feature importance
                feature_importance = np.abs(shap_data['shap_values']).mean(axis=0)
                importance_df = pd.DataFrame({
                    'Feature': shap_data['feature_names'],
                    'Mean |SHAP Value|': feature_importance
                }).sort_values('Mean |SHAP Value|', ascending=False)
                
                # Plot feature importance
                plt, sns = load_plotting_libs()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                top_15 = importance_df.head(15)
                bars = ax.barh(range(len(top_15)), top_15['Mean |SHAP Value|'], color='skyblue', alpha=0.8)
                
                ax.set_yticks(range(len(top_15)))
                ax.set_yticklabels(top_15['Feature'])
                ax.set_xlabel('Mean |SHAP Value| (Average Feature Impact)')
                ax.set_title('Global Feature Importance (Top 15 Features)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_15['Mean |SHAP Value|'])):
                    ax.text(val + max(top_15['Mean |SHAP Value|']) * 0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{val:.0f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display importance table
                st.write("**Feature Importance Rankings:**")
                display_importance = importance_df.copy()
                display_importance['Mean |SHAP Value|'] = display_importance['Mean |SHAP Value|'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(display_importance, use_container_width=True)
                
                # Insights
                top_3_features = importance_df.head(3)['Feature'].tolist()
                st.info(f"🔍 **Key Insight**: The three most important features globally are: **{', '.join(top_3_features)}**")
            
            # Tab 3: SHAP Summary Plot
            with shap_tab3:
                st.write("#### SHAP Summary Plot")
                st.write("Shows the distribution of SHAP values for each feature. Each dot represents one prediction.")
                
                try:
                    # Create SHAP summary plot
                    plt, sns = load_plotting_libs()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Use SHAP's built-in summary plot
                    shap.summary_plot(
                        shap_data['shap_values'], 
                        shap_data['X_sample'], 
                        feature_names=shap_data['feature_names'],
                        max_display=15,
                        show=False
                    )
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.write("""
                    **How to read this plot:**
                    - Each row represents a feature
                    - Each dot represents one house prediction
                    - Color indicates feature value (red = high, blue = low)
                    - X-axis shows SHAP impact on price
                    - Features are sorted by importance
                    """)
                    
                except Exception as e:
                    st.error(f"Could not generate SHAP summary plot: {str(e)}")
                    st.info("Try using a smaller dataset or different visualization.")
            
            # Tab 4: Dependence Plot
            with shap_tab4:
                st.write("#### SHAP Dependence Plot")
                st.write("Shows how a feature's value affects the prediction, and interactions with other features.")
                
                # Let user select feature for dependence plot
                important_features = importance_df.head(10)['Feature'].tolist()
                selected_feature = st.selectbox(
                    "Select feature for dependence analysis:",
                    important_features,
                    help="Choose a feature to see how its value affects predictions"
                )
                
                if selected_feature in shap_data['feature_names']:
                    feature_idx = shap_data['feature_names'].index(selected_feature)
                    
                    try:
                        plt, sns = load_plotting_libs()
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Create dependence plot
                        shap.dependence_plot(
                            feature_idx,
                            shap_data['shap_values'],
                            shap_data['X_sample'],
                            feature_names=shap_data['feature_names'],
                            show=False,
                            ax=ax
                        )
                        
                        ax.set_title(f'SHAP Dependence Plot: {selected_feature}')
                        
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Show current prediction point
                        current_feature_value = shap_data['current_input'][selected_feature].iloc[0]
                        current_shap_value = current_shap[feature_idx]
                        
                        st.info(f"""
                        **Your house's {selected_feature}**: {current_feature_value}  
                        **SHAP impact**: {current_shap_value:+,.0f} DKK
                        """)
                        
                        st.write(f"""
                        **How to read this plot:**
                        - X-axis: {selected_feature} values
                        - Y-axis: SHAP impact on price prediction
                        - Each dot: one house in the dataset
                        - Color: interaction effect with another feature
                        - Your house value is shown above
                        """)
                        
                    except Exception as e:
                        st.error(f"Could not generate dependence plot: {str(e)}")
                        
                        # Fallback: simple scatter plot
                        st.write("Showing simplified relationship:")
                        
                        plt, sns = load_plotting_libs()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        x_values = shap_data['X_sample'].iloc[:, feature_idx]
                        y_values = shap_data['shap_values'][:, feature_idx]
                        
                        ax.scatter(x_values, y_values, alpha=0.6, color='blue')
                        ax.scatter(current_feature_value, current_shap_value, 
                                 color='red', s=100, marker='*', label='Your House')
                        
                        ax.set_xlabel(f'{selected_feature} Value')
                        ax.set_ylabel('SHAP Impact on Price')
                        ax.set_title(f'Feature Impact: {selected_feature}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close(fig)
            
            # Additional SHAP insights
            st.write("### 💡 SHAP Insights Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**For Your Prediction:**")
                positive_impact = current_shap[current_shap > 0].sum()
                negative_impact = current_shap[current_shap < 0].sum()
                
                st.metric("Positive Contributions", f"+{positive_impact:,.0f} DKK")
                st.metric("Negative Contributions", f"{negative_impact:,.0f} DKK")
                st.metric("Net Effect", f"{positive_impact + negative_impact:+,.0f} DKK")
            
            with col2:
                st.write("**Most Influential Features:**")
                top_positive = explanation_df[explanation_df['SHAP Impact'] > 0].head(1)
                top_negative = explanation_df[explanation_df['SHAP Impact'] < 0].head(1)
                
                if not top_positive.empty:
                    feature = top_positive.iloc[0]
                    st.success(f"🔺 **{feature['Feature']}**: +{feature['SHAP Impact']:,.0f} DKK")
                
                if not top_negative.empty:
                    feature = top_negative.iloc[0]
                    st.error(f"🔻 **{feature['Feature']}**: {feature['SHAP Impact']:,.0f} DKK")
                
                # Overall prediction confidence
                total_magnitude = np.abs(current_shap).sum()
                relative_uncertainty = total_magnitude / predicted_price * 100
                
                if relative_uncertainty < 10:
                    st.info("🎯 **High Confidence**: Low feature uncertainty")
                elif relative_uncertainty < 20:
                    st.warning("⚠️ **Medium Confidence**: Moderate uncertainty")
                else:
                    st.error("❌ **Low Confidence**: High feature uncertainty")