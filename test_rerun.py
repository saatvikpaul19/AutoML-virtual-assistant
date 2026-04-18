import streamlit as st
import time

if "count" not in st.session_state:
    st.session_state.count = 0

st.write("Count:", st.session_state.count)

if st.session_state.count == 0:
    st.session_state.count += 1
    st.write("First run. Rerunning now...")
    st.rerun()
else:
    st.write("Second run.")
