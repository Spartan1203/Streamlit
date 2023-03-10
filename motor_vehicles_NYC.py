import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from google.oauth2 import service_account
from google.cloud import storage
from IO import BytesIO

DATA_URL = ("https://storage.cloud.google.com/edelweis/Motor_Vehicle_Collisions_-_Crashes.csv")
#Authenticating connection to cloud data source
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)
st.title("Sars-CoV-2 Data Exploration Project")
# Loading Dataset

st.title('Motor Vehicle Collision in New York City')
st.markdown("This is a streamlit dashboard to monitor vehicle collision in NYC")

blob = client.get_bucket('edelweis').get_blob('Motor_Vehicle_Collisions_-_Crashes.csv')
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
st.header('Where are the most injured people in NYC?')
injured_people=st.slider("Number of persons injured in vehicle collisions", 0, 19)
st.map(data.query("injured_persons >= @injured_people")[['latitude','longitude']].dropna(how='any'))

st.header("How many collisions happen during a given time of day?")
hour = st.slider("Hour to look at", 0,23)
data=data[data['date/time'].dt.hour==hour]

st.markdown("vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) %24 ))
midpoint=(np.average(data['latitude']),np.average(data['longitude']))
st.write(pdk.Deck(
	map_style="mapbox://styles/mapbox/light-v9",
	initial_view_state={
		"latitude":midpoint[0],
		"longitude":midpoint[1],
		"zoom":11,
		"pitch":50,
	},
	layers=[pdk.Layer(
	"HexagonLayer",
	data=data[['date/time','latitude','longitude']],
	get_position=['longitude','latitude'],
	radius=100,
	extruded=True,
	pickable=True,
	elevation_scale=4,
	elevation_range=[0,1000],
	)],
))

st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour+1) %24))
filtered = data[
	(data['date/time'].dt.hour >= hour) & (data['date/time'].dt.hour < (hour+1))
]
hist=np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0,60))[0]
chart_data=pd.DataFrame({'minute':range(60), 'crashes':hist})
fig=px.bar(chart_data,x='minute', y='crashes',hover_data=['minute','crashes'],height=400)
st.write(fig)

st.header("Top 5 dangerous streets by affected type")

select=st.selectbox("Affected type of people", ['Pedestrian','Cyclist','Motorist'])

if select=='Pedestrian':
	st.write(original_data.query("injured_pedestrians>=1")[['on_street_name','injured_pedestrians']].sort_values(by=['injured_pedestrians'],ascending=False).dropna(how='any')[:5])

elif select=='Cyclist':
	st.write(original_data.query("injured_cyclists>=1")[['on_street_name','injured_cyclists']].sort_values(by=['injured_cyclists'],ascending=False).dropna(how='any')[:5])

else:
	st.write(original_data.query("injured_motorists>=1")[['on_street_name','injured_motorists']].sort_values(by=['injured_motorists'],ascending=False).dropna(how='any')[:5])

if st.checkbox("Show Raw Data", False):
	st.subheader('Raw Data')
	st.write(data)
