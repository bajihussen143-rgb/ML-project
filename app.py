from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from titanic_pipeline import (TARGET, clean_text, drop_useless, encode_categoricals,
    handle_missing, load_data, scale_features, split_data, validate_data)

st.set_page_config(page_title="Titanic Data Studio", page_icon="◈", layout="wide")
st.markdown(r'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#080b18;--panel:#11162b;--line:rgba(165,149,255,.20);--muted:#9ba6c7;--purple:#a99aff;--cyan:#67e8f9}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} .stApp{background:radial-gradient(circle at 85% 0%,rgba(112,77,220,.18),transparent 32%),radial-gradient(circle at 4% 18%,rgba(40,130,190,.08),transparent 25%),var(--bg);color:#f4f5ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1125,#090d1c);border-right:1px solid var(--line)}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.17em;font-size:.72rem;font-weight:700;margin-bottom:.5rem}
.hero{font-family:'Space Grotesk',sans-serif;font-size:2.8rem;line-height:1.08;font-weight:700;letter-spacing:-.06em;color:#fff}
.grad{background:linear-gradient(90deg,#b7aaff,#75e9f4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--muted);font-size:1rem;max-width:800px;margin-top:.8rem}
.metric{background:linear-gradient(145deg,rgba(25,31,61,.98),rgba(14,19,38,.98));border:1px solid var(--line);border-radius:16px;padding:18px;min-height:112px}
.mlabel{font-size:.78rem;color:var(--muted);margin-bottom:8px}.mvalue{font-family:'Space Grotesk',sans-serif;font-size:1.85rem;font-weight:700;color:#fff}.mnote{font-size:.76rem;color:var(--purple);margin-top:7px}
.panel{background:rgba(17,22,43,.7);border:1px solid var(--line);border-radius:16px;padding:18px}.kicker{color:#b7aaff;text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:700}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid rgba(169,154,255,.4);background:linear-gradient(100deg,#7160df,#5d55bb);color:white;font-weight:600;min-height:2.6rem}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#c3b9ff;color:white;box-shadow:0 0 18px rgba(126,106,245,.22)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}hr{border-color:var(--line)}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading and preparing the Titanic dataset…")
def prepare():
    raw=load_data(); validate_data(raw)
    text=clean_text(raw); reduced=drop_useless(text); imputed=handle_missing(reduced)
    encoded=encode_categoricals(imputed)
    Xtr,Xte,ytr,yte=split_data(encoded)
    Xtr_s,Xte_s,scaler=scale_features(Xtr,Xte)
    return dict(raw=raw,text=text,reduced=reduced,imputed=imputed,encoded=encoded,
        Xtr=Xtr,Xte=Xte,ytr=ytr,yte=yte,Xtr_s=Xtr_s,Xte_s=Xte_s,scaler=scaler)

try: d=prepare()
except Exception as e:
    st.error(f"Pipeline could not run: {e}")
    st.info("Check your internet connection for the first dataset load, then run: python -m pip install -r requirements.txt")
    st.stop()

raw=d['raw']; encoded=d['encoded']
def metric(label,value,note):
    st.markdown(f'<div class="metric"><div class="mlabel">{label}</div><div class="mvalue">{value}</div><div class="mnote">{note}</div></div>',unsafe_allow_html=True)
def csvdata(frame): return frame.to_csv(index=False).encode('utf-8')
def layout(fig,height=320):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#e4e7ff',family='DM Sans'),margin=dict(l=10,r=10,t=45,b=10),title_font=dict(family='Space Grotesk',size=16,color='#f4f5ff'))
    fig.update_xaxes(gridcolor='rgba(165,149,255,.10)');fig.update_yaxes(gridcolor='rgba(165,149,255,.10)')
    return fig

with st.sidebar:
    st.markdown("## ◈ DATA STUDIO")
    st.caption("TITANIC · PREPROCESSING PIPELINE")
    st.markdown('---')
    page=st.radio('NAVIGATION',['Overview','Data Explorer','Data Quality','Preprocessing Lab','Exports','About the Project'])
    st.markdown('---')
    st.markdown('**PIPELINE STATUS**')
    for label in ['Dataset loaded','Schema validated','Missing values handled','Categories encoded','Stratified split created','Features scaled']:
        st.markdown(f'🟢 {label}')
    st.markdown('---')
    st.caption('Python · Pandas · Scikit-learn · Plotly · Streamlit')

if page=='Overview':
    st.markdown('<div class="eyebrow">DATA ENGINEERING · PROJECT 01</div>',unsafe_allow_html=True)
    st.markdown('<div class="hero">Titanic <span class="grad">Data Preparation</span><br>Pipeline</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub">A reproducible preprocessing workspace that transforms raw passenger data into clean, encoded, scaled datasets ready for downstream machine learning — without training a prediction model.</div>',unsafe_allow_html=True)
    st.write(''); a,b,c,e=st.columns(4)
    with a: metric('RAW PASSENGERS',f'{len(raw):,}','Original dataset')
    with b: metric('SOURCE COLUMNS',str(raw.shape[1]),'Before feature reduction')
    with c: metric('ENCODED FEATURES',str(encoded.drop(columns=[TARGET]).shape[1]),'After categorical encoding')
    with e: metric('FINAL NULL CELLS',str(int(encoded.isna().sum().sum())),'After current preprocessing')
    st.write(''); left,right=st.columns([1.2,1])
    with left:
        st.markdown('<div class="kicker">DATASET SNAPSHOT</div>',unsafe_allow_html=True);st.subheader('Passenger records')
        st.dataframe(raw.head(8),use_container_width=True,hide_index=True)
        st.caption('Original records before cleaning and encoding.')
    with right:
        st.markdown('<div class="kicker">TARGET DISTRIBUTION</div>',unsafe_allow_html=True);st.subheader('Survival classes')
        counts=raw['survived'].map({0:'Did not survive',1:'Survived'}).value_counts().rename_axis('Outcome').reset_index(name='Passengers')
        fig=px.pie(counts,names='Outcome',values='Passengers',hole=.65,color='Outcome',color_discrete_map={'Did not survive':'#8b7cf6','Survived':'#67e8f9'})
        fig.update_traces(textposition='outside',textinfo='percent+label',marker=dict(line=dict(color='#10152a',width=3)))
        st.plotly_chart(layout(fig),use_container_width=True)
        st.caption('Descriptive statistics only; no prediction is performed.')
    st.write('');st.markdown('<div class="kicker">PIPELINE AT A GLANCE</div>',unsafe_allow_html=True);st.subheader('From raw data to model-ready features')
    stages=[('01','Load & validate','Check schema and target integrity.'),('02','Clean & reduce','Normalize text and remove redundant fields.'),('03','Handle missing data','Median for numeric; mode for categories.'),('04','Encode categories','Binary, ordinal, and one-hot encoding.'),('05','Split & scale','Stratified 80/20 split; fit scaler on train.')]
    cols=st.columns(5)
    for col,(num,title,desc) in zip(cols,stages):
        with col: st.markdown(f'<div class="panel"><div class="kicker">{num}</div><b>{title}</b><p style="color:#9ba6c7;font-size:.83rem">{desc}</p></div>',unsafe_allow_html=True)

elif page=='Data Explorer':
    st.markdown('<div class="eyebrow">EXPLORE THE DATA</div>',unsafe_allow_html=True);st.title('Data Explorer')
    choice=st.selectbox('Choose a dataset stage',['Original dataset','After text cleaning','After column reduction','After missing-value handling','Encoded dataset','Scaled training features'])
    frames={'Original dataset':raw,'After text cleaning':d['text'],'After column reduction':d['reduced'],'After missing-value handling':d['imputed'],'Encoded dataset':encoded,'Scaled training features':d['Xtr_s']}
    frame=frames[choice];search=st.text_input('Filter rows across displayed values',placeholder='Try: female, Southampton, 3…')
    view=frame
    if search.strip(): view=frame.loc[frame.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1)]
    st.caption(f'Showing {len(view):,} of {len(frame):,} rows · {frame.shape[1]} columns')
    st.dataframe(view,use_container_width=True,hide_index=True,height=430)
    st.download_button('Download this view as CSV',csvdata(view),file_name=f'{choice.lower().replace(" ","_")}.csv',mime='text/csv')
    with st.expander('Column types and unique values'):
        info=pd.DataFrame({'Column':frame.columns,'Data type':[str(frame[c].dtype) for c in frame.columns],'Non-null values':[int(frame[c].notna().sum()) for c in frame.columns],'Unique values':[int(frame[c].nunique(dropna=True)) for c in frame.columns]})
        st.dataframe(info,use_container_width=True,hide_index=True)

elif page=='Data Quality':
    st.markdown('<div class="eyebrow">TRUST THE INPUTS</div>',unsafe_allow_html=True);st.title('Data Quality Report')
    before=raw.isna().sum();after=encoded.isna().sum();a,b,c,e=st.columns(4)
    with a: metric('SOURCE ROWS',f'{len(raw):,}','Rows loaded')
    with b: metric('SOURCE COLUMNS',str(raw.shape[1]),'Original fields')
    with c: metric('DUPLICATE ROWS',str(int(raw.duplicated().sum())),'Exact row duplicates')
    with e: metric('FINAL NULL CELLS',str(int(after.sum())),'After current pipeline')
    left,right=st.columns(2)
    with left:
        st.subheader('Missing values by source column');miss=before[before>0].sort_values(ascending=False).rename('Missing values').reset_index().rename(columns={'index':'Column'})
        if len(miss): st.plotly_chart(layout(px.bar(miss,x='Missing values',y='Column',orientation='h',color='Missing values',color_continuous_scale=['#5d55bb','#b6aaff']),max(300,35*len(miss))),use_container_width=True)
        else: st.success('No missing values in the source dataset.')
    with right:
        st.subheader('Missing cells: before vs after');comp=pd.DataFrame({'Stage':['Before','After'],'Missing cells':[int(before.sum()),int(after.sum())]})
        fig=px.bar(comp,x='Stage',y='Missing cells',color='Stage',text='Missing cells',color_discrete_map={'Before':'#8b7cf6','After':'#67e8f9'});fig.update_traces(textposition='outside')
        st.plotly_chart(layout(fig),use_container_width=True)
    st.subheader('Column-level audit')
    audit=pd.DataFrame({'Column':raw.columns,'Data type':[str(raw[c].dtype) for c in raw.columns],'Missing before':[int(raw[c].isna().sum()) for c in raw.columns],'Missing %':[round(float(raw[c].isna().mean()*100),1) for c in raw.columns],'Unique values':[int(raw[c].nunique(dropna=True)) for c in raw.columns],'In encoded output':[c in encoded.columns for c in raw.columns]})
    st.dataframe(audit,use_container_width=True,hide_index=True)
    st.warning('Methodology note: this assignment-style version imputes and encodes before splitting. For stronger production practice, fit imputers and encoders on training data only, then transform test data. The scaler is already fitted on training features only.')

elif page=='Preprocessing Lab':
    st.markdown('<div class="eyebrow">UNDER THE HOOD</div>',unsafe_allow_html=True);st.title('Preprocessing Lab')
    steps=[('01 · Validation','Required columns exist, the dataset is non-empty, and the target has no missing values.'),('02 · Text normalization','String values are trimmed and lowercased.'),('03 · Column reduction','Redundant, leakage-prone, and mostly-empty columns are removed.'),('04 · Missing-value handling','Numeric values use median imputation; categories use the mode.'),('05 · Categorical encoding','Sex is binary encoded, pclass remains ordinal, and remaining categories are one-hot encoded.'),('06 · Stratified split','An 80/20 split preserves the target class distribution.'),('07 · Feature scaling','The scaler learns from training features and transforms both subsets.')]
    for title,desc in steps:
        with st.expander(title,expanded=title.startswith('07')): st.write(desc)
    st.subheader('Train / test comparison')
    split=pd.DataFrame({'Subset':['Training','Testing'],'Rows':[len(d['Xtr']),len(d['Xte'])],'Survived (%)':[round(float(d['ytr'].mean()*100),1),round(float(d['yte'].mean()*100),1)]})
    left,right=st.columns(2)
    with left:
        fig=px.bar(split,x='Subset',y='Rows',color='Subset',text='Rows',color_discrete_map={'Training':'#8b7cf6','Testing':'#67e8f9'});fig.update_traces(textposition='outside');st.plotly_chart(layout(fig),use_container_width=True)
    with right:
        fig=px.bar(split,x='Subset',y='Survived (%)',color='Subset',text='Survived (%)',color_discrete_map={'Training':'#8b7cf6','Testing':'#67e8f9'});fig.update_traces(texttemplate='%{text}%',textposition='outside');st.plotly_chart(layout(fig),use_container_width=True)
    st.dataframe(split,use_container_width=True,hide_index=True)
    with st.expander('Inspect scaling statistics'):
        stats=pd.DataFrame({'Feature':d['Xtr_s'].columns,'Training mean after scaling':d['Xtr_s'].mean().values.round(4),'Training std after scaling':d['Xtr_s'].std(ddof=0).values.round(4)})
        st.dataframe(stats,use_container_width=True,hide_index=True);st.caption('Training means should be near 0 and population standard deviations near 1.')

elif page=='Exports':
    st.markdown('<div class="eyebrow">TAKE YOUR DATA WITH YOU</div>',unsafe_allow_html=True);st.title('Export Center');st.caption('Download prepared datasets for inspection or later model development.')
    items=[('Encoded dataset',encoded,'titanic_encoded_dataset.csv','All encoded rows, including the target.'),('Training features — scaled',d['Xtr_s'],'X_train_scaled.csv','Scaled training features; target is separate.'),('Testing features — scaled',d['Xte_s'],'X_test_scaled.csv','Scaled test features.'),('Training target',d['ytr'].to_frame(TARGET),'y_train.csv','Training survival labels.'),('Testing target',d['yte'].to_frame(TARGET),'y_test.csv','Testing survival labels.'),('Training features — unscaled',d['Xtr'],'X_train_unscaled.csv','Encoded training features before scaling.'),('Testing features — unscaled',d['Xte'],'X_test_unscaled.csv','Encoded test features before scaling.')]
    for i in range(0,len(items),2):
        cols=st.columns(2)
        for col,(title,frame,filename,desc) in zip(cols,items[i:i+2]):
            with col:
                st.markdown('<div class="panel">',unsafe_allow_html=True);st.subheader(title);st.caption(desc);st.write(f'**Shape:** {frame.shape[0]:,} rows × {frame.shape[1]} columns')
                st.download_button('Download CSV',csvdata(frame),file_name=filename,mime='text/csv',key=filename);st.markdown('</div>',unsafe_allow_html=True)

else:
    st.markdown('<div class="eyebrow">PROJECT DOCUMENTATION</div>',unsafe_allow_html=True);st.title('About this project')
    st.markdown('''### Objective
Prepare the Titanic passenger dataset for downstream machine learning. This project ends at data preparation and **does not train a prediction model**.

### Technology stack
- **Python** — application logic
- **Pandas** — tabular data processing
- **Scikit-learn** — stratified split and scaling
- **Streamlit** — interactive web interface
- **Plotly** — interactive charts

### What this project demonstrates
Input validation, text normalization, feature reduction, missing-value handling, categorical encoding, stratified splitting, train-only scaling, quality reporting, and CSV exports.

### Important limitation
This assignment-style implementation imputes and encodes before the split. For a stricter production workflow, fit imputers and encoders on training data only and transform the test set with those fitted objects. Do not describe this project as a trained AI model because no estimator is fitted here.''')
    st.info('Portfolio tip: capture a screenshot of the Overview page and add it to your GitHub README.')
