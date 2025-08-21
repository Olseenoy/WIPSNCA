import streamlit as st
import pandas as pd
from src.ml.similarity import SimilarityEngine
from src.ml.rca import suggest_rca

st.set_page_config(page_title="Smart NC Analyzer", layout="wide")
st.title("Smart Non-Conformance Analyzer — Streamlit MVP")

@st.cache_data
def load_data(path="data/sample_nc.csv"):
    return pd.read_csv(path, parse_dates=["detected_at"])

df = load_data()

st.sidebar.header("Create / Query NC")
with st.sidebar.form("ncr_form"):
    site = st.selectbox("Site", options=sorted(df['site'].unique()))
    line = st.text_input("Line", value="Line 1")
    title = st.text_input("Title", value="Seam leak detected")
    description = st.text_area("Description", value="Describe the non-conformance in detail...")
    submit = st.form_submit_button("Run similarity & RCA")

engine = SimilarityEngine()
if submit:
    query_text = f"{title}\n{description}"
    topk = engine.topk(query_text, k=5)
    st.subheader("Top similar historical NCs")
    for i, item in enumerate(topk, 1):
        st.markdown(f"**{i}. {item['title']}** — score: {item['score']:.3f}")
        st.write(item['description'])
        st.markdown(f"_Meta_: site={item.get('site')}, line={item.get('line')}, defect_type={item.get('defect_type')}")
    st.subheader("RCA Suggestion (heuristic)")
    rca = suggest_rca({'title': title, 'description': description}, topk)
    st.json(rca)

st.header("Dataset & Dashboard (sample)")
st.dataframe(df[['detected_at','site','line','title','defect_type','severity']])

st.subheader("Summary charts")
c1, c2 = st.columns(2)
with c1:
    st.bar_chart(df['defect_type'].value_counts().rename_axis('defect').reset_index(name='count').set_index('defect'))
with c2:
    agg = df.groupby('line').size().reset_index(name='count').set_index('line')
    st.bar_chart(agg)