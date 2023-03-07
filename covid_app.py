# covid_app.py
# Importing relevant modules
import streamlit as st
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from google.cloud import storage
from io import BytesIO
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime, date
from matplotlib.pyplot import subplot2grid
from matplotlib.lines import Line2D
import time

# Authenticating connection to cloud data source
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)
st.title("Sars-CoV-2 Data Exploration Project")
st.markdown("This is a Streamlit dashboard to explore global Covid-19 cases between Jan 2020 and Jan 2022")
# Loading Dataset

blob = client.get_bucket('edelweis').get_blob('dataset_cleaned.csv')
blobBytes = blob.download_as_bytes()
info = BytesIO(blobBytes)

@st.cache(persist=True, allow_output_mutation=True)
def load_data(info):
    dataset = pd.read_csv(info)
    return pd.DataFrame(dataset)


dataset = load_data(info)
pd.set_option('mode.chained_assignment',
              None
              )

st.header('Exploratory Data Analysis')

#st.markdown("Daily global report of confirmed cases, death and recovery from Sars-CoV-2 Virus between 01-01-2020 and 01-01-2022")

with st.spinner("data loading..."):
    time.sleep(5)

view_raw_data = st.checkbox('View Dataset', False)
if view_raw_data:
    st.caption("The data above has been preprocessed for the following exploratory tasks")
    st.dataframe(dataset)

# Create a mapping dictionary of country and continent name
country_continent = {'United Kingdom': 'Europe',
                     'Korea, North': 'Asia',
                     'Summer Olympics 2020': 'Asia',
                     'Tonga': 'Oceania',
                     'Winter Olympics 2022': 'Asia',
                     'Canada': 'North America'}

# Map dictionary to fill NaN values in continent name column
dataset['continent_name'].fillna(dataset.country.map(country_continent),
                                 inplace=True
                                 )

# Create mapping dictionary of continent name and continent code
a = ['North America', 'Asia', 'Oceania',
     'Europe', 'South America', 'Africa', 'nan']
b = ['NA', 'AS', 'OC', 'EU', 'SA', 'AF', 'NAN']
name_code = dict(zip(a, b))

# Map dictionary to fill NaN values in continent code column
dataset['continent_code'].fillna(dataset.continent_name.map(name_code),
                                 inplace=True
                                 )

# Drop continent names that have inconsistent data that cannot be reliably mapped with the information available
dataset.dropna(subset=['continent_name'],
               inplace=True
               )

# Forward-fill NaN values in population column
dataset.population.fillna(method='ffill', inplace=True)

# Set column data types
dataset = dataset.astype({'date': 'datetime64[ns]',
                          'country': 'object',
                         'type': 'category',
                          'cases': 'int',
                          'population': 'int',
                          'continent_name': 'category',
                          'continent_code': 'category'}
                         )

# Drop values less that zero in cases column
dataset=dataset[dataset.cases >= 0]

# Set date column as index and drop original column
dataset.index=dataset['date']
dataset.drop(columns='date',
             inplace=True
             )

#filtering for dates between jan 2020 and jan 2022
dataset=dataset[dataset.index < '2022-01-01']
original_dataset = dataset

view_data_info = st.checkbox("View info of cleaned dataset", False)
if view_data_info:
    st.write(original_dataset.info(verbose=True))
fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.barplot(data=original_dataset,
                x='continent_code',
                y='population'
                 )
ax.set_title('Population per Continent')
ax.set_xlabel('continent code')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ylabel = ("continent name")
plt.tight_layout()
#bar = plt.savefig("population.jpg")
#st.caption("The data above has been preprocessed for the following exploratory tasks")

st.pyplot(fig)
st.caption("Barplot showing the total population of each continent")
# st.bar_chart(dataset.continent_name.value_counts(dropna=False))

#Group the data by month, continent_name and type and calculate the mean number of cases for each group
monthly_continent_grouping=pd.DataFrame(original_dataset.groupby([pd.Grouper(level=0,
                                                                    freq='M'),
                                                         'continent_name',
                                                         'type']).mean()
                                                         )

# Plotting categories of monthly cases per continent
fig, cont_bars = plt.subplots(figsize=(10,5))
cont_bars=sns.barplot(data=monthly_continent_grouping,
            x=monthly_continent_grouping.index.get_level_values(1),
            y=monthly_continent_grouping.values[:,0],
            hue=monthly_continent_grouping.index.get_level_values(2),
            ci=None)
cont_bars.set_title('Cases per Continent (Monthly)')
cont_bars.set_xlabel('continent')
cont_bars.set_ylabel('cases')
cont_bars.spines['right'].set_visible(True)
cont_bars.spines['top'].set_visible(False)
cont_bars.legend(ncol=3,frameon=False)
sns.despine(left=True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

continents=['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
select_continent = st.selectbox("Continent", continents)

@st.cache(persist=True, allow_output_mutation=True)
def create_continent_pivot_table(select_continent):

    """ Create a pivot table for each continent containing the number of confirmed cases, deaths,
    and recoveries for each month in the pivot table's continent.
    It also calculates the death and recovery rates per confirmation for each month."""

    # Filter data for the given continent
    monthly_continent = monthly_continent_grouping.query(f'continent_name == "{select_continent}"')

    # Create pivot table using the filtered data
    monthly_continent_pivot = monthly_continent.pivot_table(index='date',
                                                           columns='type',
                                                           values='cases'
                                                           )
    
    # Calculate 'death_per_confirmation' rate and add it to pivot table
    monthly_continent_pivot['death_per_confirmation'] = monthly_continent_pivot.death/monthly_continent_pivot.confirmed

    # Calculate 'recovery_per_confirmation' rate and add it to pivot table
    monthly_continent_pivot['recovery_per_confirmation'] = monthly_continent_pivot.recovery/monthly_continent_pivot.confirmed
    
    # Set the name of pivot table to the given continent name
    monthly_continent_pivot.name = select_continent
    monthly_continent_pivot.date = monthly_continent_pivot.index.to_series
    return monthly_continent_pivot

table =create_continent_pivot_table(select_continent)
table=pd.DataFrame(table)
st.dataframe(table)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1=sns.lineplot(data=table,
                x=table.index,
                y=table.recovery_per_confirmation,
                label=select_continent,
                color='green',
                linestyle='-.'
                )

ax=ax1.axes.twinx()

ax=sns.lineplot(data=table,
                x=table.index,
                y=table.death_per_confirmation,
                color='red',
                )
ax1.legend(handles=
            [Line2D([],
                [],
                marker='_',
                color="red", label="recovery per confirmation"),
            Line2D([],
                [],
                marker="_",
                color='green',
                label='death per confirmation')
                ],
            frameon=False,
            bbox_to_anchor=(1.01,1.03),
            loc='upper right'
        )
ax1.set_title('Cases per Continent (Monthly)', loc='left')
ax1.tick_params(axis='y',
                color='green',
                labelcolor='green')
ax1.set_ylabel('recovery per confirmation', color='green')
ax1.set_xlabel(f'{table.index.name}')
ax1.tick_params(axis='x',
                rotation=45
                )

ax.tick_params(axis='y',
                color='red',
                labelcolor='red')
ax.set_ylabel('death per confirmation', color='red')

plt.tight_layout()
st.pyplot(fig)

# Table of countries indexed by month
monthly_country_grouping=pd.DataFrame(dataset.groupby([pd.Grouper(level=0,
                                                                    freq='M'),
                                                         'country',
                                                         'type']).mean()
                                                         )
st.dataframe(monthly_country_grouping)
country = pd.unique(original_dataset.country)
country = sorted(country)

select_country = st.selectbox("Country", country)
fig, ax = plt.subplots(figsize=(10,5))
table1 = monthly_country_grouping.loc[(monthly_country_grouping.index.get_level_values(1)==f"{select_country}")&(monthly_country_grouping.index.get_level_values(2)=="confirmed")]
confirmed, = ax.plot(table1.index.get_level_values(0),
          'cases',
          data=table1,
          color='mediumslateblue',
          label=f'{country}',
          linestyle='-.'
          )
table2 = monthly_country_grouping.loc[(monthly_country_grouping.index.get_level_values(1)==f"{select_country}")&(monthly_country_grouping.index.get_level_values(2)=="death")]  
death, = ax.plot(table2.index.get_level_values(0),
          'cases',
          data=table2,
          color='r',
          label=f'{country}',
          linestyle='-.'
          )
table3 = monthly_country_grouping.loc[(monthly_country_grouping.index.get_level_values(1)==f"{select_country}")&(monthly_country_grouping.index.get_level_values(2)=="recovery")]
recovery, = ax.plot(table3.index.get_level_values(0),
          'cases',
          data=table3,
          color='g',
          label=f'{country}',
          linestyle='-.'
          )
ax.set_xlabel(f'{monthly_country_grouping.index.get_level_values(0).name}')
ax.set_ylabel('cases')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(handles=
            [Line2D([],
                [],
                marker='_',
                color="blue", label="confirmed"),
            Line2D([],
                [],
                marker="_",
                color='red',
                label='death'),
            Line2D([],
                [],
                marker="_",
                color='green',
                label='recovery')
                ],
                frameon=False,
                loc='lower right'
        )
sns.despine(left=True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.tight_layout()
st.pyplot(fig)