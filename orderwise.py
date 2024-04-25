import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

st.set_page_config(page_title="Sales Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide"
)

df = pd.read_excel(
    io="Coffee_sales.xlsx",
    engine="openpyxl",
    sheet_name="Sheet1",
    skiprows=0,
    usecols="B:L",
    #nrows=1000,
)


#MAINPAGE

st.title(":coffee: Sales Dashboard")
st.markdown("##")


st.subheader("Full Dataset:")
st.dataframe(df)

#Sidebar

periods = st.sidebar.text_input('Periods', value='10')
periods = int(periods)

st.sidebar.header("Please filter here:")

product = st.sidebar.multiselect(
    "Select the product",
    options=df["Product"].unique(),
    #default=df["Product"].unique()
)

size = st.sidebar.multiselect(
    "Select the size",
    options=df["Size"].unique(),
    #default=df["Size"].unique()
)

sales_type = st.sidebar.multiselect(
    "Select the sales type id",
    options=df["Sales_type_Id"].unique(),
    #default=df["Sales_type_Id"].unique()
)

category = st.sidebar.multiselect(
    "Select the categories",
    options=df["Category"].unique(),
    #default=df["Category"].unique()
)

df_selection = df.query(
    "Product == @product & Size == @size & Sales_type_Id == @sales_type & Category == @category"
)

st.subheader("Filtered Dataset:")
st.dataframe(df_selection)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.


#Top KPIs

total_sales = df_selection["Total_units"].sum()
total_retail = round(df_selection["Total_Retail"].sum(), 2)
total_cost = round(df_selection["Total_Cost"].sum(), 2)
#star_rating = ":star:" * int(round(average_retail, 0))
#average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"{total_sales:,}")
with middle_column:
    st.subheader("Average Total Retail:")
    st.subheader(f"US $ {total_retail}")
with right_column:
    st.subheader("Average Total Cost:")
    st.subheader(f"US $ {total_cost}")

st.markdown("""---""")

if len(product) > 1:

    # SALES BY PRODUCTS [BAR CHART]
    sales_by_product = df_selection.groupby(by=["Product"])[["Total_units"]].sum().sort_values(by="Total_units")
    fig_product_sales = px.bar(
        sales_by_product,
        x="Total_units",
        y=sales_by_product.index,
        orientation="h",
        title="<b>Sales by Product</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_product),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # SALES BY PRODUCT RETAIL PRICE [BAR CHART]
    product_retail = df_selection.groupby(by=["Product"])[["Total_Retail"]].sum().sort_values(by="Total_Retail")
    fig_retail = px.bar(
        product_retail,
        x="Total_Retail",
        y=product_retail.index,
        orientation="h",
        title="<b>Product Retail</b>",
        color_discrete_sequence=["#0083B8"] * len(product_retail),
        template="plotly_white",
    )
    fig_retail.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # SALES BY HOUR [BAR CHART]
    #sales_by_hour = df_selection.groupby(by=["hour"])[["Total"]].sum()
    #fig_hourly_sales = px.bar(
    #    sales_by_hour,
    #    x=sales_by_hour.index,
    #    y="Total",
    #    title="<b>Sales by hour</b>",
    #    color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
    #    template="plotly_white",
    #)
    #fig_hourly_sales.update_layout(
    #    xaxis=dict(tickmode="linear"),
    #    plot_bgcolor="rgba(0,0,0,0)",
    #    yaxis=(dict(showgrid=False)),
    #)


    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_retail, use_container_width=True) 
    right_column.plotly_chart(fig_product_sales, use_container_width=True)


if len(product) == 1:

    #Prediction

    df_pred = df_selection[["Item_Segment", "Total_units"]]
    df_pred = df_pred.rename(columns={'Item_Segment': 'ds', 'Total_units': 'y'})

    #df_pred.set_index('ds', drop=True, inplace=True)

    st.subheader("Prediction Dataset:")
    df_temp = df_pred
    df_temp = df_temp.rename(columns={'ds': 'Date', 'y': 'Total_units'})
    st.dataframe(df_temp)
    del df_temp
    #st.line_chart(df_pred['y'])

    m = Prophet()
    m.fit(df_pred)

    future = m.make_future_dataframe(periods=periods)
    #st.dataframe(future)

    forecast = m.predict(future)
    st.subheader("Forecasting Dataset:")
    #st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    df_temp = forecast
    df_temp = df_temp.rename(columns={'ds': 'Date', 'yhat': 'Predicted Total_units', 'yhat_lower': 'Predicted Total_units Lower', 'yhat_upper': 'Predicted Total_units Upper'})
    pred_max_sum_total_no_units = round(df_temp[['Predicted Total_units Upper']].tail(periods).sum()).astype(float)
    pred_sum_total_no_units = round(df_temp[['Predicted Total_units']].tail(periods).sum()).astype(float)
    st.dataframe(df_temp[['Date', 'Predicted Total_units', 'Predicted Total_units Lower', 'Predicted Total_units Upper']].tail(periods)) 

    del df_temp

    #st.dataframe(forecast)

    st.subheader("Forecasting Plot:")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Seasonal Components Plots:")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader(f"INFERENCE FROM PREDICTIONS:")
    st.markdown(f"Projected Total Sales for :blue[{product[0]}] of size :blue[{size[0]}]:")
    st.markdown(f"Max: :green[{pred_max_sum_total_no_units[0]}]")
    st.markdown(f"Exact: :green[{pred_sum_total_no_units[0]}]")


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

