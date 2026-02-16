# Dashboard Analisis Data: E-Commerce Public Dataset

## Import Semua Packages/Library yang Digunakan

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set_theme(style="whitegrid", palette="pastel")

## Menyiapkan DataFrame

### Membuat Helper Function create_monthly_delivery_review_trend_df

def create_monthly_delivery_composition_df(
    df,
    date_col="order_delivered_customer_date",
    min_order=75,
):
    df = df.copy()
    df["month"] = df[date_col].dt.to_period("M")

    monthly = (
        df.groupby(["month", "delivery_status"])
          .agg(order_count=("order_id", "nunique"))
          .reset_index()
    )

    # Total order per bulan
    monthly["total_orders"] = (
        monthly.groupby("month")["order_count"]
               .transform("sum")
    )

    # Filter
    monthly = monthly[monthly["total_orders"] >= min_order]

    # Hitung proporsi
    monthly["delivery_pct"] = (
        monthly["order_count"] / monthly["total_orders"]
    )

    monthly_delivery_composition = (
        monthly
        .pivot(
            index="month",
            columns="delivery_status",
            values="delivery_pct"
        )
        .sort_index()
    )

    # Pastikan ketiga status ini ada di kolom, kalau tidak ada isi dengan 0
    expected_status = ["Lebih Cepat", "Tepat Waktu", "Terlambat"]
    monthly_delivery_composition = monthly_delivery_composition.reindex(
        columns=expected_status, 
        fill_value=0
    )

    # Ubah tipe data month (index) menjadi string
    monthly_delivery_composition.index = (
        monthly_delivery_composition.index.astype(str)
    )
    
    return monthly_delivery_composition

### Membuat Helper Function create_review_by_delivery_status_df

def create_review_by_delivery_status_df(df):
    review_by_delivery_status = (
        df.groupby("delivery_status")
          .agg(
              avg_review_score=("review_score", "mean"),
              order_count=("order_id", "nunique")
          )
          .reset_index()
    )
    return review_by_delivery_status

### Membuat Helper Function create_review_by_delivery_status_and_category_df

def create_review_by_delivery_status_and_category_df(
    df,
    top_n=5,
    category_col="product_category_name_english",
):
    # Ambil 5 kategori berdasarkan pesanan terbanyak
    top_categories = (
        df.groupby(category_col)
          .order_id
          .nunique()
          .sort_values(ascending=False)
          .head(top_n)
          .index
    )

    # Kelompokkan rata-rata skor review pada tiap delivery status untuk masing-masing kategori
    category_review = (
        df.groupby([category_col, "delivery_status"])
          .agg(
              avg_review_score=("review_score", "mean"),
              order_count=("order_id", "nunique")
          )
          .reset_index()
    )

    # Filter top categories
    category_review = category_review[
        category_review[category_col].isin(top_categories)
    ]

    # Pivot for plotting
    review_by_delivery_status_and_category = (
        category_review
        .pivot(
            index=category_col,
            columns="delivery_status",
            values="avg_review_score"
        )
    )

    return review_by_delivery_status_and_category

### Membuat Helper Function create_freight_ratio_satisfaction_df

def create_freight_ratio_satisfaction_df(
    df,
    exclude_bin=">100%",
    ratio_col="freight_ratio_bin",
):
    # Filter valid ratio range
    df_valid = df[df[ratio_col] != exclude_bin].copy()

    # Hapus kategori yang tidak terpakai
    if isinstance(df_valid[ratio_col].dtype, pd.CategoricalDtype):
        df_valid [ratio_col] = (
            df_valid [ratio_col]
            .cat.remove_unused_categories()
        )

    # Kelompokkan rata-rata skor review pada tiap freight_ratio_bin
    freight_ratio_satisfaction = (
        df_valid
        .groupby(ratio_col, observed=False)
        .agg(
            avg_review_score=("review_score", "mean"),
            order_count=("order_id", "nunique")
        )
        .reset_index()
    )

    return freight_ratio_satisfaction

### Membuat Helper Function create_customer_delay_by_state_df

def create_customer_delay_by_state_df(
    df,
    customers_df,
):
    # Gabungkan (join) df dan customer state
    orders_df_cust = df.merge(
        customers_df[["customer_id", "customer_state"]],
        on="customer_id",
        how="left"
    )

    # Filter data dengan delivery status terlambat
    orders_df_cust = orders_df_cust.assign(
        is_delayed=lambda x: x["delivery_status"] == "Terlambat"
    )

    # Kelompokkan berdasarkan customer state
    # Hitung delayed rate
    customer_delay_by_state = (
        orders_df_cust.groupby("customer_state")
          .agg(
              total_orders=("order_id", "nunique"),
              delayed_orders=("is_delayed", "sum"),
              avg_review=("review_score", "mean")
          )
          .assign(
              delayed_rate=lambda x: x["delayed_orders"] / x["total_orders"]
          )
          .reset_index()
    )

    return customer_delay_by_state

### Membuat Helper Function create_seller_density_by_state_df

def create_seller_density_by_state_df(sellers_df):
    seller_density_by_state = (
        sellers_df
        .groupby("seller_state")
        .agg(
            seller_count=("seller_id", "nunique")
        )
        .reset_index()
    )

    return seller_density_by_state

### Membuat Helper Function attach_geo_state_data

def attach_geo_state_data(
    geo_df,
    data_df,
    geo_key="sigla",
    data_key="customer_state",
):
    geo_df = geo_df.copy()
    geo_df[geo_key] = geo_df[geo_key].str.upper()

    merged_geo = geo_df.merge(
        data_df,
        left_on=geo_key,
        right_on=data_key,
        how="left"
    )

    return merged_geo

### Membuat Helper Function get_top_n_states

def get_top_n_states(df, metric_col, n=3):
    return (
        df.sort_values(metric_col, ascending=False)
          .head(n)
    )

### Load Data

df = pd.read_csv("orders_df_master.csv")
df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"],
    errors="coerce"
)
customers_df = pd.read_csv("customers.csv")
sellers_df = pd.read_csv("sellers.csv")
brazil_states = gpd.read_file(
    'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
)

### Memanggil Helper Function

monthly_delivery_composition = (
    create_monthly_delivery_composition_df(
        df,
        date_col="order_delivered_customer_date",
        min_order=75,
    )
)
review_by_delivery_status = (
    create_review_by_delivery_status_df(df)
)
review_by_delivery_status_and_category = (
    create_review_by_delivery_status_and_category_df(
        df,
        top_n=5,
        category_col="product_category_name_english"
    )
)
freight_ratio_satisfaction = (
    create_freight_ratio_satisfaction_df(
        df,
        exclude_bin=">100%",
        ratio_col="freight_ratio_bin",
    )
)
customer_delay_by_state = (
    create_customer_delay_by_state_df(
        df,
        customers_df,
    )
)
seller_density_by_state = (
    create_seller_density_by_state_df(sellers_df)
)
geo_delayed = (
    attach_geo_state_data(
        brazil_states,
        customer_delay_by_state,
        geo_key="sigla",
        data_key="customer_state"
    )
)
geo_seller = (
    attach_geo_state_data(
        brazil_states,
        seller_density_by_state,
        geo_key="sigla",
        data_key="seller_state"
    )
)
top_delayed_states = (
    get_top_n_states(
        geo_delayed,  
        metric_col="delayed_rate",
        n=3
    )
)
top_seller_states = (
    get_top_n_states(
        geo_seller,     
        metric_col="seller_count",
        n=3
    )
)

## Melengkapi Dashboard dengan Berbagai Visualisasi Data

### Menambahkan Header

# Tambahkan header
st.header('Logistics Performance Analysis: Delivery Timeliness, Cost, and Customer Satisfaction')

### Menambahkan Informasi Terkait Overall Health

# Tampilkan informasi total delivered orders, delayed percentage, dan average review score
st.subheader('Overall Health')

col1, col2, col3 = st.columns(3)

with col1:
    total_delivered_orders = df["order_id"].nunique()
    st.metric(
        label="Total Delivered Orders",
        value=f"{total_delivered_orders:,}"
    )

with col2:
    delayed_percentage = (
        (df["delivery_status"] == "Terlambat").mean()
    )
    st.metric(
        label="Delayed Orders (%)",
        value=f"{delayed_percentage:.1%}"
    )

with col3:
    avg_review_score = df["review_score"].mean()
    st.metric(
        label="Average Review Score",
        value=f"{avg_review_score:.2f}"
    )

### Menambahkan Informasi Terkait Operational Reability

# Tampilkan informasi tentang persentase performa pengiriman dan trennya seiring waktu
st.subheader('Operational Reability')

col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Delivery Mix")

    delivery_status_map = {
        "Terlambat": "Delayed",
        "Lebih Cepat": "Early",
        "Tepat Waktu": "On-Time"
    }

    status_order = ["Early", "On-Time", "Delayed"]
    colors = ["#90CAF9", "#A5D6A7", "#EF9A9A"]

    delivery_perf = (
        df["delivery_status"]
          .map(delivery_status_map)
          .value_counts(normalize=True)
          .reindex(status_order)
          .reset_index()
    )

    delivery_perf.columns = ["delivery_status", "percentage"]

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        delivery_perf["percentage"],
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white")  # donut effect
    )

    # Center text
    ax.text(
        0, 0,
        "92%\nEarly",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_title("Delivery Status Distribution", pad=8)
    ax.axis("equal")
    st.pyplot(fig, width='stretch')

with col2:
    st.markdown("### Monthly Delivery Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    for col, label, color in [
        ("Lebih Cepat", "Early", "#81C784"),
        ("Tepat Waktu", "On-Time", "#64B5F6"),
        ("Terlambat", "Delayed", "#E57373"),
    ]:
        ax.plot(
            monthly_delivery_composition.index,
            monthly_delivery_composition[col],
            marker="o",
            linewidth=2.5,
            label=label,
            color=color
        )

    ax.set_title(
        "Monthly Delivery Performance Trend",
        fontsize=14,
        pad=12
    )
    ax.set_ylabel("Percentage of Orders")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

    ax.yaxis.set_major_formatter(
        lambda x, _: f"{x:.0%}"
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.autofmt_xdate() # Otomatis merapikan tanggal yang miring di sumbu X
    plt.tight_layout()  # Memastikan tidak ada label yang terpotong frame
    
    st.pyplot(fig, width='stretch')
    st.markdown(
        """
        <p style="font-size: 10px; color: gray; line-height: 1;">
        Note : Data only includes monthly order volumes above 75 to ensure sample validity.
        </p>
        """, 
        unsafe_allow_html=True
    )

### Menambahkan Informasi Terkait Customer Experience Impact

# Tampilkan informasi tentang pengaruh ketepatan pengiriman dan rasio ongkos kirim terhadap kepuasan pelanggan
st.subheader('Customer Experience Impact')

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Impact of Delivery Timeliness")

    delivery_order = ["Lebih Cepat", "Tepat Waktu", "Terlambat"]
    review_by_delivery_status["delivery_status"] = pd.Categorical(
        review_by_delivery_status["delivery_status"],
        categories=delivery_order,
        ordered=True
    )
    review_by_delivery_status = review_by_delivery_status.sort_values("delivery_status")


    fig, ax = plt.subplots(figsize=(6, 5))

    sns.barplot(
        x="delivery_status",
        y="avg_review_score",
        data=review_by_delivery_status,
        hue="delivery_status",
        errorbar=None,
        palette='Set1',
        legend=False,
        ax=ax
    )

    # Anotasi
    for i, row in review_by_delivery_status.iterrows():
        ax.text(
            i,
            row["avg_review_score"] + 0.05,
            f"{row['avg_review_score']:.2f}\n(n={row['order_count']:,})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylabel("Average Review Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 5)
    ax.set_xticks(
        ticks=[0, 1, 2],
        labels=['Early', 'On-Time', 'Delayed']
    )
    ax.set_title("Customer Satisfaction by Delivery Status")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig, width='stretch')


with col2:
    st.markdown("### Impact of Freight-to-Price Ratio")

    fig, ax = plt.subplots(figsize=(6, 5.5))

    sns.barplot(
        data=freight_ratio_satisfaction,
        x='freight_ratio_bin',
        y='avg_review_score',
        hue="freight_ratio_bin",
        errorbar=None,
        palette='Blues_r',
        legend=False,
        ax=ax
    )

    # Anotasi
    for i, row in freight_ratio_satisfaction.iterrows():
        ax.text(
            i,
            row["avg_review_score"] + 0.02,
            f"{row['avg_review_score']:.2f}\n(n={row['order_count']:,})",
            ha="center",
            va="bottom",
            fontsize=6
        )

    # Benchmark line
    overall_avg = freight_ratio_satisfaction["avg_review_score"].mean()
    ax.axhline(
        overall_avg,
        linestyle="--",
        linewidth=1,
        color='red',
        label=f"Overall Avg ({overall_avg:.2f})"
    )

    ax.set_ylabel("Average Review Score")
    ax.set_xlabel("Freight-to-Price Ratio")
    ax.set_ylim(3.8, 4.3)
    ax.tick_params(axis='x', rotation=30, labelsize=10)

    ax.legend()
    ax.set_title("The Effect of Freight-to-Price Ratio on Customer Satisfaction (0-100%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig, width='stretch')
    st.markdown(
        """
        <p style="font-size: 10px; color: gray; line-height: 1;">
        Note: Orders with shipping cost ratio >100% are excluded from the visualization
        as they are extreme cases and the amount is relatively small.
        </p>
        """, 
        unsafe_allow_html=True
    )

# Tampilkan informasi tentang hubungan skor review dengan kategori produk dan status pengiriman
heatmap_data = review_by_delivery_status_and_category.copy()

delivery_status_map = {
   "Terlambat": "Delayed",
    "Lebih Cepat": "Early",
    "Tepat Waktu": "On-Time"
}

heatmap_data = heatmap_data.rename(columns=delivery_status_map)

heatmap_data = heatmap_data[["Early", "On-Time", "Delayed"]]

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=1,
    vmax=5,
    linewidths=0.5,
    linecolor="white",
    ax=ax
)

ax.set_title(
    "Customer Review Score by Product Category and Delivery Status",
    fontsize=14,
    pad=12
)
ax.set_xlabel("Delivery Status", fontsize=10)
ax.set_ylabel("Product Category", fontsize=10)

plt.tight_layout()
st.pyplot(fig, width='stretch')

st.caption(
    ""
    "Across all top product categories, delayed deliveries consistently lead to lower customer review scores, "
    "notwithstanding that most orders actually arrive early. This negative impact is universal,"
    "regardless of product type. While higher freight costs are tolerated up to a certain threshold, "
    "delivery timeliness has a significantly stronger influence on overall customer satisfaction."
)

### Menambahkan Informasi Terkait Geographical Operational Risk

st.subheader("Geographical Operational Risk")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    geo_delayed.plot(
        column="delayed_rate",
        cmap="Reds",
        linewidth=0.8,
        ax=ax1,
        edgecolor="0.7",
        legend=True
    )

    ax1.set_title("Delayed Delivery Rate by Customer Region", fontsize=14)
    ax1.axis("off")

    # Anotasi top 3 delayed states
    for _, row in top_delayed_states.iterrows():
        centroid = row.geometry.centroid

        ax1.annotate(
            text=row["name"],
            xy=(centroid.x, centroid.y),
            xytext=(centroid.x + 3, centroid.y - 2.5),
            arrowprops=dict(arrowstyle="->", lw=1, color='black'),
            fontsize=10,
            fontweight="bold",
            ha="left"
        )

    st.pyplot(fig1, width='stretch')

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    geo_seller.plot(
        column="seller_count",
        cmap="Blues",
        linewidth=0.8,
        ax=ax2,
        edgecolor="0.7",
        legend=True
    )

    ax2.set_title("Seller Density by Region", fontsize=14)
    ax2.axis("off")

    # Anotasi top 3 seller density states
    for _, row in top_seller_states.iterrows():
        centroid = row.geometry.centroid

        ax2.annotate(
            text=row["name"],
            xy=(centroid.x, centroid.y),
            xytext=(centroid.x - 8, centroid.y - 5),
            arrowprops=dict(arrowstyle="->", lw=1, color='black'),
            fontsize=10,
            fontweight="bold",
            ha="right"
        )

    st.pyplot(fig2, width='stretch')

st.caption(
    "Regions with higher delayed delivery rates tend to have lower seller density, "
    "indicating a structural logistics imbalance. Improving seller distribution or "
    "last-mile efficiency in these regions could significantly reduce delivery delays."
)