import requests
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# -----------------------------
# State Definition
# -----------------------------
class AqiState(TypedDict):
    city: str
    latitude: float
    longitude: float
    aqi: float
    category: str
    description: str
    color: str


# -----------------------------
# Node 1: Fetch AQI
# -----------------------------
def measure_aqi(state: AqiState) -> AqiState:
    city = state["city"]

    geo_res = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    ).json()

    state["latitude"] = geo_res["results"][0]["latitude"]
    state["longitude"] = geo_res["results"][0]["longitude"]

    aqi_res = requests.get(
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={state['latitude']}&longitude={state['longitude']}&current=us_aqi"
    ).json()

    state["aqi"] = aqi_res["current"]["us_aqi"]
    return state


# -----------------------------
# Node 2: Label AQI
# -----------------------------
def label_aqi(state: AqiState) -> AqiState:
    aqi = state["aqi"]

    if 0 <= aqi <= 50:
        state["category"] = "Good"
        state["description"] = "Air quality is satisfactory."
        state["color"] = "green"

    elif 51 <= aqi <= 100:
        state["category"] = "Moderate"
        state["description"] = "Acceptable air quality."
        state["color"] = "yellow"

    elif 101 <= aqi <= 150:
        state["category"] = "Unhealthy for Sensitive Groups"
        state["description"] = "Sensitive people may experience issues."
        state["color"] = "orange"

    elif 151 <= aqi <= 200:
        state["category"] = "Unhealthy"
        state["description"] = "Health effects possible for everyone."
        state["color"] = "red"

    elif 201 <= aqi <= 300:
        state["category"] = "Very Unhealthy"
        state["description"] = "Health alert!"
        state["color"] = "purple"

    else:
        state["category"] = "Hazardous"
        state["description"] = "Emergency conditions."
        state["color"] = "maroon"

    return state


# -----------------------------
# Build LangGraph
# -----------------------------
graph = StateGraph(AqiState)
graph.add_node("measure_aqi", measure_aqi)
graph.add_node("label_aqi", label_aqi)

graph.add_edge(START, "measure_aqi")
graph.add_edge("measure_aqi", "label_aqi")
graph.add_edge("label_aqi", END)

workflow = graph.compile()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AQI Checker", page_icon="🌫️", layout="centered")

st.title("🌍 Air Quality Index Checker")

city = st.text_input("Enter city name", "Udupi")

if st.button("Check AQI"):
    with st.spinner("Fetching AQI data..."):
        result = workflow.invoke({"city": city})

    # Emoji based on AQI category
    color_map = {
        "Good": "🟢",
        "Moderate": "🟡",
        "Unhealthy for Sensitive Groups": "🟠",
        "Unhealthy": "🔴",
        "Very Unhealthy": "🟣",
        "Hazardous": "🟤"
    }

    indicator = color_map.get(result["category"], "⚪")

    # -------- CARD UI --------

    st.html(
        f"""
        <div style="
            display: inline-block;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            min-width: 320px;
            max-width: 500px;
            font-family: Arial, sans-serif;
        ">

            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">📍 {city}</h3>
                <span style="font-size: 22px;">{indicator}</span>
            </div>

            <!-- AQI Value -->
            <h1 style="margin: 8px 0;">{result['aqi']}</h1>

            <!-- AQI Category -->
            <p style="margin: 4px 0; font-size: 16px;">
                <b>AQI Level:</b> {result['category']}
            </p>

            <!-- Location Info -->
            <p style="margin: 4px 0; color: #555;">
                <b>Latitude:</b> {result['latitude']}<br>
                <b>Longitude:</b> {result['longitude']}
            </p>

            <!-- Description -->
            <p style="margin-top: 8px; color: #555;">
                <b>Description:</b> {result['description']}
            </p>

        </div>
        """
    )


