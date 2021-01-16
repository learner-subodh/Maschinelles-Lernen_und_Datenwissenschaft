import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import io
import requests
from PIL import Image

st.title('KKBox Music Recommendation!')

image = Image.open('kkbox-unsplash.jpg')
st.image(image, use_column_width=True)

st.header('1. Business Problem')

st.subheader('1.1 Problem Desciption')
st.markdown("The 11th ACM International Conference on Web Search and Data Mining (WSDM 2018) challenged to build a better music recommendation system using a donated dataset from KKBOX. WSDM (pronounced 'wisdom') is one of the premier conferences on web inspired research involving search and data mining.")
st.markdown("The glory days of Radio DJs have passed, and musical gatekeepers have been replaced with personalizing algorithms and unlimited streaming services. With easy access to various kinds of music across the globe, public is now listening to all kinds of music. Existing algorithms, however, struggle in key areas. Without enough historical data, how would an algorithm know if listeners will like a new song or a new artist? And how would it know what songs to recommend brand new users?")
st.markdown("The dataset is from KKBOX, Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. The input contains text data only, and no audio features. They currently use a collaborative filtering based algorithm with matrix factorization and word embedding in their recommendation system but believe new techniques could lead to better results.")
st.markdown("In this case study, we will be looking towards some good techniques to recommend music to brand new as well as existing users. By building this system, we aim to provide a better user experience for the app users.")

st.subheader('1.2 Problem Statement')
st.markdown("We are asked to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. The objective is to make prediction on whether a user will re-listen to a song or not. Broadly, it is a music recommendation problem (ROC-AUC Score as per Kaggle Challenge Evaluation Metric).")

st.subheader('1.3 Sources/References')
st.markdown("1. Paper: KKBox’s Music Recommendation, Yunru Huang, Mengyu Li, Yun Wu, Stanford University: http://cs229.stanford.edu/proj2019spr/report/4.pdf")
st.markdown("2. Paper: KKBOX’s Music Recommendation Challenge Solution with Feature Engineering, Jianyu Zhang, Françoise Fogelman-Soulié, School of Computer Software, Tianjin University: https://wsdm-cup-2018.kkbox.events/pdf/WSDM_KKboxs_Music_Recommendation_Challenge_6th_Solution.pdf")
st.markdown("3. Kaggle Notebook: Recommendation System with 83 percent accuracy lgbm: https://www.kaggle.com/rohandx1996/recommendation-system-with-83-accuracy-lgbm")
st.markdown("4. Kaggle Notebook: Introduction to Boosting using LGBM (LB: 0.68357): https://www.kaggle.com/vinnsvinay/introduction-to-boosting-using-lgbm-lb-0-68357")
st.markdown("5. Blog: WSDM — KKBox’s Music Recommendation Challenge: https://medium.com/@anjar.aquil123/wsdm-kkboxs-music-recommendation-challenge-87ca72c41593")

st.header('2. Machine Learning Problem')
st.subheader('2.1 Data Overview')
st.markdown("Get the data from: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data")
st.markdown("Data Files:")
st.markdown("1. members.csv")
st.markdown("2. sample_submission.csv")
st.markdown("3. song_extra_info.csv")
st.markdown("4. songs.csv")
st.markdown("5. test.csv")
st.markdown("6. train.csv")

st.subheader('2.2 Mapping the real world problem to a Machine Learning Problem')
st.markdown("2.2.1 Type of Machine Learning Problem")
st.markdown("We are asked to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. The objective is to make prediction on whether a user will re-listen to a song or not. Broadly, it is a music recommendation problem.")
st.markdown("This can be also thought of as classification problem, that is, whether user will or will not listen to the recommended song.")
st.markdown('2.2.2 Performance metric')
st.markdown("Area under the ROC curve between the predicted probability and the observed target:")
st.markdown("https://en.wikipedia.org/wiki/Receiver_operating_characteristic")
st.markdown("https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/")
st.markdown('2.2.3 Machine Learning Objective and Constraints')
st.markdown("1. Maximize Area Under Curve")
st.markdown("2. Try to provide some interpretability")

st.header("3. Exploratory Data Analysis")

# Method 1
#url = requests.get('https://drive.google.com/file/d/14yS4Q4kRGAN4D65V-2UBzKU4RvU-Yj16/view?usp=sharing')
#DATA_URL = StringIO(url.text)

# Method 2
#orig_url = 'https://drive.google.com/file/d/1-G4oszK_85iH7XYl0NrgQqbmOARphRye/view?usp=sharing'
#file_id = orig_url.split('/')[-2]
#dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
#url = requests.get(dwn_url).text
#DATA_URL = StringIO(url)

# Method 3
#url = 'https://drive.google.com/file/d/1-G4oszK_85iH7XYl0NrgQqbmOARphRye/view?usp=sharing'
#DATA_URL = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

# Method 4
#fileDownloaded = drive.CreateFile({'id':'14yS4Q4kRGAN4D65V-2UBzKU4RvU-Yj16'})
#fileDownloaded.GetContentFile('train_data_merged.csv')

#url = "https://drive.google.com/file/d/1-G4oszK_85iH7XYl0NrgQqbmOARphRye/view?usp=sharing"
#s = requests.get(url).content

@st.cache
def load_data(nrows):
    #data = pd.read_csv(io.StringIO(s.decode('utf-8')))
    data = pd.read_csv('train_data_merged.csv', nrows=nrows)
    return data

data_load_state = st.text('Loading data...')
data = load_data(100000)
data_load_state.text("Done!")

if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(data)

st.subheader('Distribution of target')
image = Image.open('cnt_plot_tar.png')
st.image(image, use_column_width=True)
st.markdown("We can observe that the dataset is **_balanced_** around the **'target'** attribute.",
             unsafe_allow_html=False)

#st.subheader('Age Distribution of users')
#fig, ax = plt.subplots()
#ax.hist(data['bd'], bins=20)
#st.pyplot(fig)
st.subheader('Age Distribution of users')
image = Image.open('cnt_plot_bd.png')
st.image(image, use_column_width=True)
st.markdown("Most users are less than **_60 years_** of age. However, some users have age less than or equal to **_0 years_** while there are a few with age greater than **_100 years_**, some even with age greater than **_1000 years_**, which is practically impossible. This can be further understood by looking at the boxplot below.", 
            unsafe_allow_html=False)

st.subheader('Boxplot for Age of users')
image = Image.open('bd_boxplot.png')
st.image(image, use_column_width=True)

st.subheader('PDF of Age of users')
image = Image.open('bd_pdf.png')
st.image(image, use_column_width=True)
st.markdown("Around **_35 percent_** of the age values are less than or equal to zero, while around **_66 percent_** of them lie between 0 & 54 years of age, both inclusive.", 
            unsafe_allow_html=False)

st.subheader('CDF of Age of users')
image = Image.open('bd_cdf.png')
st.image(image, use_column_width=True)
st.markdown("More than **_99 percent_** of the users have their age registered between 0 & 54 years", 
            unsafe_allow_html=False)

st.subheader('Source System Tab vs target')
image = Image.open('sst_tar.png')
st.image(image, use_column_width=True)
st.markdown("Max number of listening events occur at **_'my library'_** & **_'discover'_** tabs, which is kind of logical as the prior one tends to have songs which a user listens quite frequently while the later acts as a platform to explore more songs which a user might in turn like. **_60 percent_** of the events generated through the 'my library' tab have target value of 1 while 'discover' tab has events with target as 0 **slightly** more than that with target as 1.", 
            unsafe_allow_html=False)

st.subheader('Source Type vs target')
image = Image.open('sstype_tar.png')
st.image(image, use_column_width=True)
st.markdown("The source type **_'local library'_** generates the most number of events out of which around two third of the events have target value as 1. It is followed by source types **_'online playlist'_** and **_'local playlist'_** which have more contributions towards target = 0 and target = 1 respectively.", 
            unsafe_allow_html=False)

st.subheader('Source Screen Name vs target')
image = Image.open('ssn_tar.png')
st.image(image, use_column_width=True)
st.markdown("Out of the total events trigerred through given source screens, more than **_half_** of the events are triggered at the **_'local playlist more_** screen. And out of the total events generated from 'local playlist more' screen, around two third of events have target as 1. The **_'online playlist more'_** screen also has a good enough contribution towards target = 1 events.", 
            unsafe_allow_html=False)

st.markdown("Thus, it is quite evident that there is some **_overlap_** in the **tabs** or **screens** provided under _'Source System Tab'_, _'Source Type'_ & _'Source Screen Name'_.", 
            unsafe_allow_html=False)

st.subheader('Distribution of Genders')
image = Image.open('cnt_plt_gender.png')
st.image(image, use_column_width=True)
st.markdown("The dataset is **_balanced_** around the **'gender'** attribute.", 
            unsafe_allow_html=False)

st.subheader('Gender vs target')
image = Image.open('gen_tar.png')
st.image(image, use_column_width=True)
st.markdown("Both genders seem to be **_evenly_** poised towards both target values.", 
            unsafe_allow_html=False)

st.subheader('Song Duration vs target')
image = Image.open('songdur_tar.png')
st.image(image, use_column_width=True)
st.markdown("Most songs, around **_50 percent_** to be precise, have a duration of **4 minutes**, and around**_ 95 percent_** of the songs have a duration of either **3, 4 or 5 minutes**. Also, most users prefer listeing to songs with **_not so long_** duration, that is somewhat close to or around **3 to 5 minutes** of length. Both target values are in **_equal_** number for all songs with different durations.", 
            unsafe_allow_html=False)

st.subheader('Distribution of Listening events')
image = Image.open('dist_listening.png')
st.image(image, use_column_width=True)
st.markdown("As can be observed, a lot of songs have been listened to very few number of times and very few songs have been listened to a lot of time. Thus, these songs which are listeed to many times might be very **_popular_** ones either in the area where the data is collected or even globally.", 
            unsafe_allow_html=False)

st.subheader('City vs target')
image = Image.open('city_tar.png')
st.image(image, use_column_width=True)
st.markdown("Most of the users in the given dataset belong to city having id as **_1_** or the data that is provided was collected largely in cities with id as **_1_** followed by id **_13_** & **_5_**. Thus, we can expect some local trends in the type of songs generally liked by the people living in these particular cities. Both the target values are almost in equal numbers in all cities.", 
            unsafe_allow_html=False)

st.subheader('Language vs target')
image = Image.open('lang_tar.png')
st.image(image, use_column_width=True)
st.markdown("Most of the users tend to prefer the language having id as **_3_** followed by id **_52_**. There isn't any significant difference in the target values they contribute to. But after having a look at the previous plot of cities data, we can infer that users/people living in cities **_1_** & **_13_** prefer languages **_3_** & **_52_** respectively.", 
            unsafe_allow_html=False)

st.subheader('Registration via vs target')
image = Image.open('regvia_tar.png')
st.image(image, use_column_width=True)
st.markdown("A ot of users prefer registering via modes **_9_** & **_7_**, which may be easy ways to register probably using existing google accounts or some social media account, etc. Both these modes along with others have somewhat same number of events trigerred for both the target values.", 
            unsafe_allow_html=False)

st.subheader('Heatmap for missing numbers')
image = Image.open('heatmap_feat.png')
st.image(image, use_column_width=True)
st.markdown("We can see there is a strong **_correlation_** in the trends of **missing values** for some features.", 
            unsafe_allow_html=False)

st.subheader('Dendrogram for missing numbers')
image = Image.open('msno_dendro.png')
st.image(image, use_column_width=True)
st.markdown("A missing value dendrogram **_clusters_** features which show a strong **_correlation_** in trens=ds of missing values. Features clustered initially have a strong correlation while those joined/clustered later don't have that strong correlation.", 
            unsafe_allow_html=False)