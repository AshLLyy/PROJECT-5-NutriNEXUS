#%%
import streamlit as st
import requests
import os
import json
import logging

from ultralytics import YOLO
from PIL import Image

# %%
# API configuration
BASE_API_URL = "https://6e49-2001-d08-1381-def6-79cd-80fa-893a-2541.ngrok-free.app"
FLOW_ID = "48955cd2-1abb-4841-81f9-48fb2a1a8fbd"
ENDPOINT = "dietry"

# Initialize logging
logging.basicConfig(level=logging.INFO)

def run_flow(message: str, endpoint: str, output_type: str = "chat", input_type: str = "chat") -> dict:
    """
    Run a flow with a given message and optional tweaks.
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    response = requests.post(api_url, json=payload)
    return response.json()

def extract_message(response: dict) -> str:
    """Extract the assistant's response message from the API."""
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

model = YOLO(r'best_yolo11n.pt')

def classify_with_yolo(image):
    """Classify the input image using the YOLO model."""
    try:
        results = model(image, task="classify")  # Ensure task="classify"
        if results and len(results) > 0:
            probs = results[0].probs
            predicted_class = results[0].names[probs.top1]  # Access the top-1 class index
            confidence = probs.top1conf.item()  # Access the top-1 confidence score
            return predicted_class, confidence
        else:
            return None, None
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return None, None
    
def main():
    # Default image path
    IMAGE_PATH = os.path.join(os.getcwd(), 'static', '09944-feature1-nutrition.jpg')

    # Load default display image
    image = Image.open(IMAGE_PATH)
    st.image(image, use_column_width=True)

    st.title("👩🏻‍🎓 NutriNEXUS: Your Personalized Nutrition Companion.")
    st.write("⬅️ Please capture an image first before you start asking a question.")
    st.write("NOTE: IF USING PHONE PLEASE CLICK THE ARROW BUTTON ABOVE.")
    st.write("⬇️ You may ask any question about a healthy diet below.")

# Sidebar setup
    with st.sidebar:
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)

        # Initialize variables to avoid UnboundLocalError
        predicted_class, confidence = None, None

        if picture is not None:
            # Convert the captured image
            image = Image.open(picture).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform YOLO inference
            with st.spinner("Classifying..."):
                predicted_class, confidence = classify_with_yolo(image)

    # Display results
    if predicted_class:
        st.success(f"**Predicted Class**: {predicted_class}")
        st.write(f"**Confidence**: {confidence:.2f}")
    else:
        st.warning("No prediction available or no image provided.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Chat input for the user
    if query := st.chat_input("Ask me anything about diet and nutrition:"):
        # Log user query
        st.session_state.messages.append(
            {"role": "user", "content": query, "avatar": "👩🏻"}
        )
        with st.chat_message("user", avatar="👩🏻"):
            st.write(query)

        # Get assistant response
        with st.chat_message("assistant", avatar="👩🏻‍🎓"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                assistant_response = extract_message(run_flow(query, endpoint=ENDPOINT))
                message_placeholder.write(assistant_response)

        # Log assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response, "avatar": "👩🏻‍🎓"}
        )

if __name__ == "__main__":
    main()
# %%
