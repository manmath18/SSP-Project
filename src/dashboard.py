import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display_dashboard(class_counts: dict, slot_numbers: dict):
    st.title("Parking Dashboard")

    # Display the counts in a table
    st.subheader("Parking Space Counts")
    df = pd.DataFrame(list(class_counts.items()), columns=["Status", "Count"])
    st.table(df)

    # Display bar chart
    st.subheader("Bar Chart")
    st.bar_chart(df.set_index("Status"))

    # Display pie chart
    st.subheader("Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(df["Count"], labels=df["Status"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # Display parking slot status
    st.subheader("Parking Slot Status")
    max_slots = max(slot_numbers.keys()) if slot_numbers else 0
    slot_status = ["Empty" if slot_numbers.get(i) == 0 else "Full" for i in range(1, max_slots + 1)]
    slot_df = pd.DataFrame({"Slot Number": range(1, max_slots + 1), "Status": slot_status})
    st.table(slot_df)