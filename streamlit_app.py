import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP Dashboard',
    page_icon=':earth_americas:',
    layout='wide',
    initial_sidebar_state='auto',
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file."""
    DATA_FILENAME = Path(__file__).parent / 'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Theme Selection
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .stTextInput > div > div > input {
            background-color: #333333;
            color: #ffffff;
        }
        .stSelectbox > div > div > div > input {
            background-color: #333333;
            color: #ffffff;
        }
        .stSlider > div > div > div > div > div > div > div {
            background-color: #333333;
            color: #ffffff;
        }
        .stMetricLabel {
            color: #ffffff;
        }
        .stMetricValue {
            color: #ffffff;
        }
        .stMetricDelta {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Draw the actual page

st.title(':earth_americas: GDP Dashboard')
st.sidebar.header('Filter Options')

# Sidebar filters
min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.sidebar.slider(
    'Select Year Range:',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value]
)

countries = gdp_df['Country Code'].unique()
selected_countries = st.sidebar.multiselect(
    'Select Countries:',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN']
)

if not selected_countries:
    st.warning("Please select at least one country.")

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries)) &
    (gdp_df['Year'] <= to_year) &
    (from_year <= gdp_df['Year'])
]

# Plotting the GDP over time
st.header('GDP Over Time')
plot_type = st.selectbox("Select Plot Type:", ["Line Chart", "Bar Chart"])

if plot_type == "Line Chart":
    fig = px.line(
        filtered_gdp_df,
        x='Year',
        y='GDP',
        color='Country Code',
        title='GDP Trends by Country',
        labels={'GDP': 'GDP in USD'},
        hover_data={'Country Code': True, 'Year': True, 'GDP': True},
    )
else:
    fig = px.bar(
        filtered_gdp_df,
        x='Year',
        y='GDP',
        color='Country Code',
        title='GDP Trends by Country',
        labels={'GDP': 'GDP in USD'},
        hover_data={'Country Code': True, 'Year': True, 'GDP': True},
    )

fig.update_layout(
    hovermode='x unified',
    xaxis_title='Year',
    yaxis_title='GDP in USD',
    legend_title='Country Code',
    plot_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
    paper_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
)

st.plotly_chart(fig)

# Displaying GDP metrics for the selected years
st.header(f'GDP Metrics for {from_year} to {to_year}')
cols = st.columns(len(selected_countries))

for i, country in enumerate(selected_countries):
    col = cols[i]
    with col:
        first_gdp = filtered_gdp_df[filtered_gdp_df['Year'] == from_year][filtered_gdp_df['Country Code'] == country]['GDP'].iat[0] / 1_000_000_000
        last_gdp = filtered_gdp_df[filtered_gdp_df['Year'] == to_year][filtered_gdp_df['Country Code'] == country]['GDP'].iat[0] / 1_000_000_000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'${last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )

# Additional statistics
if len(selected_countries) > 0:
    avg_growth = (filtered_gdp_df.groupby('Country Code')['GDP'].mean() / filtered_gdp_df.groupby('Country Code')['GDP'].mean().shift(1)).dropna()
    st.write("Average GDP Growth Rate:")
    st.write(avg_growth)

# Data Download Option
st.header('Download Data')
if st.button('Download Filtered Data'):
    csv = filtered_gdp_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='filtered_gdp_data.csv',
        mime='text/csv',
    )

# Interactive Map
st.header('GDP by Country Map')
map_fig = px.choropleth(
    gdp_df,
    locations='Country Code',
    locationmode='ISO-3',
    color='GDP',
    hover_name='Country Code',
    animation_frame='Year',
    color_continuous_scale=px.colors.sequential.Plasma,
    title='GDP by Country Over Time',
    labels={'GDP': 'GDP in USD'},
)

map_fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
    paper_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
)

st.plotly_chart(map_fig)

# Forecasting
st.header('GDP Forecasting')
if len(selected_countries) == 1:
    country_code = selected_countries[0]
    country_data = gdp_df[gdp_df['Country Code'] == country_code]
    country_data = country_data[['Year', 'GDP']].dropna()

    X = country_data[['Year']]
    y = country_data['GDP']

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.array(range(to_year + 1, to_year + 11)).reshape(-1, 1)
    future_gdp = model.predict(future_years)

    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'GDP': future_gdp
    })

    forecast_fig = px.line(
        forecast_df,
        x='Year',
        y='GDP',
        title=f'GDP Forecast for {country_code}',
        labels={'GDP': 'GDP in USD'},
        hover_data={'Year': True, 'GDP': True},
    )

    forecast_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
        paper_bgcolor='rgba(0,0,0,0)' if theme == "Dark" else 'rgba(255,255,255,1)',
    )

    st.plotly_chart(forecast_fig)
else:
    st.warning("Please select only one country for forecasting.")

# Expander for additional insights
with st.expander("Additional Insights", expanded=False):
    st.write("Explore more about GDP trends and economic indicators.")
    st.write("You can analyze the impact of global events on GDP growth.")
