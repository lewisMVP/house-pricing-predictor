import streamlit as st
import pandas as pd
import numpy as np
import joblib  # hoặc pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.joblib")  # model đã train
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("merged_output.csv")  # dữ liệu đã xử lý
    return df

model = load_model()
df = load_data()

# --- Sidebar: User Input ---  
st.sidebar.title("🔍 Nhập thông tin dự đoán")  
area = st.sidebar.number_input("Diện tích (m²)", min_value=10, max_value=1000, value=120)  
rooms = st.sidebar.slider("Số phòng", 1, 10, 3)  
zipcode = st.sidebar.selectbox("Mã vùng (zipcode)", df['zipcode'].unique())  
house_type = st.sidebar.selectbox("Loại nhà", df['house_type'].unique())  
sales_type = st.sidebar.selectbox("Loại bán", df['sales_type'].unique())  
year_build = st.sidebar.number_input("Năm xây dựng", min_value=1900, max_value=2025, value=2000)  
sqm_price = st.sidebar.number_input("Giá/m²", min_value=0, value=0)  
city = st.sidebar.selectbox("Thành phố", df['city'].unique())  
region = st.sidebar.selectbox("Vùng", df['region'].unique())  
nom_interest_rate = st.sidebar.number_input("Lãi suất cho vay (%)", value=0.0)  
dk_ann_infl_rate = st.sidebar.number_input("Lạm phát dự kiến (%)", value=0.0)  
yield_on_mortgage_credit_bonds = st.sidebar.number_input("Lợi suất trái phiếu tín dụng thế chấp (%)", value=0.0)  

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
st.title("🏠 Dự đoán giá nhà ở Đan Mạch")
st.subheader("Giá nhà dự đoán:")
st.success(f"💰 {predicted_price:,.0f} DKK")

# --- Thêm phần validation và gợi ý giá ---
st.subheader("💡 Phân tích giá và gợi ý")

# Tạo tabs để hiển thị thông tin
tab1, tab2 = st.tabs(["Đánh giá giá dự đoán", "Chi tiết phân tích"])

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
                label=f"Giá trung bình khu vực {zipcode}", 
                value=f"{area_avg_price:,.0f} DKK",
                delta=f"{((predicted_price - area_avg_price) / area_avg_price * 100):.1f}%" 
            )
            
            # Hiện thị khoảng giá
            st.caption(f"Dựa trên {area_count} giao dịch trong khu vực")
            st.write(f"Khoảng giá: {area_min_price:,.0f} - {area_max_price:,.0f} DKK")
            
            # Đánh giá mức giá
            if predicted_price < area_min_price:
                st.warning(f"⚠️ Giá dự đoán thấp hơn mức thấp nhất trong khu vực ({area_min_price:,.0f} DKK)")
            elif predicted_price > area_max_price:
                st.warning(f"⚠️ Giá dự đoán cao hơn mức cao nhất trong khu vực ({area_max_price:,.0f} DKK)")
            else:
                price_percentile = len(area_data[area_data['purchaseprice'] <= predicted_price]) / len(area_data) * 100
                st.info(f"✓ Giá dự đoán nằm ở mức {price_percentile:.1f}% so với các giao dịch trong khu vực")
        else:
            st.info("Không có đủ dữ liệu cho khu vực này để so sánh")
    
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
                label=f"Giá trung bình nhà tương tự", 
                value=f"{similar_avg_price:,.0f} DKK",
                delta=f"{((predicted_price - similar_avg_price) / similar_avg_price * 100):.1f}%"
            )
            
            st.caption(f"Dựa trên {similar_count} nhà có đặc điểm tương tự")
            
            # Tính toán giá đề xuất
            suggested_price = None
            confidence = "Trung bình"
            
            if len(similar_houses) >= 20 and len(area_data) >= 20:
                # Kết hợp giữa dự đoán, nhà tương tự và giá khu vực
                suggested_price = predicted_price * 0.5 + similar_avg_price * 0.3 + area_avg_price * 0.2
                confidence = "Cao"
            elif len(similar_houses) >= 10:
                # Kết hợp giữa dự đoán và nhà tương tự
                suggested_price = predicted_price * 0.6 + similar_avg_price * 0.4
                confidence = "Khá cao"
            else:
                # Chủ yếu dựa vào dự đoán
                suggested_price = predicted_price * 0.8 + similar_avg_price * 0.2
            
            # Hiển thị khoảng giá đề xuất (±5%)
            st.write("### Gợi ý giá bán:")
            min_suggested = suggested_price * 0.95
            max_suggested = suggested_price * 1.05
            
            st.success(f"💰 {min_suggested:,.0f} - {max_suggested:,.0f} DKK")
            st.caption(f"Độ tin cậy: {confidence}")
        else:
            st.info("Không đủ dữ liệu về nhà có đặc điểm tương tự để so sánh")
    
    # Hiển thị biểu đồ so sánh
    if 'area_avg_price' in locals() and 'similar_avg_price' in locals():
        st.subheader("So sánh giá")
        comparison_data = {
            'Loại': ['Giá dự đoán', 'Trung bình khu vực', 'Nhà tương tự'],
            'Giá (DKK)': [predicted_price, area_avg_price, similar_avg_price]
        }
        
        if 'suggested_price' in locals():
            comparison_data['Loại'].append('Giá đề xuất')
            comparison_data['Giá (DKK)'].append(suggested_price)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = sns.barplot(x='Loại', y='Giá (DKK)', data=comparison_df, palette='viridis', ax=ax)
        
        # Thêm giá trị lên các thanh
        for i, bar in enumerate(bars.patches):
            bars.annotate(f'{comparison_df["Giá (DKK)"][i]:,.0f}',
                         (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                         ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.title('So sánh các mức giá')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.write("### Chi tiết đánh giá")
    
    # Tạo bảng đánh giá chi tiết
    eval_data = []
    
    # Đánh giá diện tích
    area_percentile = np.percentile(df['area'], [25, 50, 75])
    if area <= area_percentile[0]:
        area_eval = ("Nhỏ", "Diện tích nhỏ hơn 25% các bất động sản")
    elif area <= area_percentile[1]:
        area_eval = ("Trung bình thấp", "Diện tích nằm giữa 25-50% các bất động sản")
    elif area <= area_percentile[2]:
        area_eval = ("Trung bình cao", "Diện tích nằm giữa 50-75% các bất động sản")
    else:
        area_eval = ("Lớn", "Diện tích lớn hơn 75% các bất động sản")
    
    eval_data.append(["Diện tích", f"{area} m²", area_eval[0], area_eval[1]])
    
    # Đánh giá số phòng
    room_counts = df['no_rooms'].value_counts()
    if rooms in room_counts.index:
        room_pct = room_counts[rooms] / room_counts.sum() * 100
        room_eval = (f"Phổ biến ({room_pct:.1f}%)", f"{room_pct:.1f}% các bất động sản có {rooms} phòng")
    else:
        room_eval = ("Không phổ biến", f"Rất ít bất động sản có {rooms} phòng")
    
    eval_data.append(["Số phòng", f"{rooms}", room_eval[0], room_eval[1]])
    
    # Đánh giá khu vực (zipcode)
    zipcode_counts = df['zipcode'].value_counts()
    if zipcode in zipcode_counts.index:
        zipcode_pct = zipcode_counts[zipcode] / zipcode_counts.sum() * 100
        zipcode_eval = (f"Có dữ liệu ({zipcode_pct:.1f}%)", f"{zipcode_counts[zipcode]} giao dịch ({zipcode_pct:.1f}%)")
    else:
        zipcode_eval = ("Thiếu dữ liệu", "Không có giao dịch trong khu vực này")
    
    eval_data.append(["Mã khu vực", f"{zipcode}", zipcode_eval[0], zipcode_eval[1]])
    
    # Đánh giá loại nhà
    house_type_counts = df['house_type'].value_counts()
    if house_type in house_type_counts.index:
        house_type_pct = house_type_counts[house_type] / house_type_counts.sum() * 100
        house_eval = (f"Phổ biến ({house_type_pct:.1f}%)", f"{house_type_pct:.1f}% là loại nhà này")
    else:
        house_eval = ("Không phổ biến", "Rất ít dữ liệu về loại nhà này")
    
    eval_data.append(["Loại nhà", f"{house_type}", house_eval[0], house_eval[1]])
    
    # Đánh giá năm xây dựng
    year_percentile = np.percentile(df['year_build'], [25, 50, 75])
    if year_build <= year_percentile[0]:
        year_eval = ("Cũ", "Thuộc 25% các bất động sản cũ nhất")
    elif year_build <= year_percentile[1]:
        year_eval = ("Trung bình cũ", "Thuộc 25-50% các bất động sản theo độ tuổi")
    elif year_build <= year_percentile[2]:
        year_eval = ("Trung bình mới", "Thuộc 50-75% các bất động sản theo độ tuổi")
    else:
        year_eval = ("Mới", "Thuộc 25% các bất động sản mới nhất")
    
    eval_data.append(["Năm xây dựng", f"{year_build}", year_eval[0], year_eval[1]])
    
    # Hiển thị bảng đánh giá
    eval_df = pd.DataFrame(eval_data, columns=["Thông số", "Giá trị", "Đánh giá", "Chi tiết"])
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
    st.subheader("Độ tin cậy của dự đoán")
    st.progress(reliability_score / max_score)
    
    if reliability_percentage >= 80:
        st.success(f"✅ Độ tin cậy cao: {reliability_percentage:.0f}%")
    elif reliability_percentage >= 50:
        st.warning(f"⚠️ Độ tin cậy trung bình: {reliability_percentage:.0f}%")
    else:
        st.error(f"❌ Độ tin cậy thấp: {reliability_percentage:.0f}%")
    
    # Lý do đánh giá độ tin cậy
    st.write("**Các yếu tố ảnh hưởng đến độ tin cậy:**")
    reasons = []
    
    if 'area_data' in locals():
        reasons.append(f"- Số giao dịch trong khu vực: {len(area_data)}")
    else:
        reasons.append("- Không có dữ liệu giao dịch trong khu vực")
    
    if 'similar_houses' in locals():
        reasons.append(f"- Số nhà có đặc điểm tương tự: {len(similar_houses)}")
    else:
        reasons.append("- Không có dữ liệu về nhà có đặc điểm tương tự")
    
    if area_percentile[0] <= area <= area_percentile[2]:
        reasons.append("- Diện tích nằm trong khoảng phổ biến")
    else:
        reasons.append("- Diện tích nằm ngoài khoảng phổ biến")
    
    if rooms in common_rooms:
        reasons.append(f"- Số phòng ({rooms}) là phổ biến")
    else:
        reasons.append(f"- Số phòng ({rooms}) không phổ biến")
    
    for reason in reasons:
        st.write(reason)

# --- Biểu đồ EDA ---
st.subheader("Phân tích dữ liệu (EDA)")

with st.expander("Heatmap Tương Quan"):
    corr = df.corr(numeric_only=True)

    # Kiểm tra nếu có cột 'purchaseprice'
    if 'purchaseprice' in corr.columns:
        # Lấy top 10 biến tương quan mạnh nhất với 'purchaseprice'
        top_corr = corr['purchaseprice'].abs().sort_values(ascending=False)[1:11].index
        selected_corr = corr.loc[top_corr, top_corr]
    else:
        # fallback: dùng toàn bộ nếu không có cột 'purchaseprice'
        selected_corr = corr

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(selected_corr, annot=True, cmap="coolwarm", linewidths=0.5, square=True, fmt=".2f")
    ax.set_title("Heatmap các biến tương quan")
    st.pyplot(fig)

with st.expander("Biểu đồ phân bố diện tích"):
    fig2, ax2 = plt.subplots()
    sns.histplot(df['area'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

with st.expander("Phân tích giá theo khu vực"):
    # Tính giá trung bình và số lượng giao dịch theo zipcode
    price_analysis = df.groupby('zipcode').agg({
        'purchaseprice': ['mean', 'count']
    }).round(2)
    
    # Đặt tên cột
    price_analysis.columns = ['Giá trung bình', 'Số giao dịch']
    
    # Hiển thị bảng và biểu đồ
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Bảng phân tích theo khu vực:")
        st.dataframe(price_analysis)
    
    with col2:
        st.write("Biểu đồ giá trung bình theo khu vực:")
        st.line_chart(price_analysis['Giá trung bình'])

# --- User Guide ---
st.sidebar.markdown("---")
st.sidebar.markdown(" **Hướng dẫn:** Nhập thông tin ở trên để dự đoán giá nhà. Xem biểu đồ và phân tích bên dưới.")
