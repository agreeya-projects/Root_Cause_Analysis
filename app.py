import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# Set your OpenAI API key
#openai.api_key = "sk-proj-adrlB9ADFCMHWnTanDvOT3BlbkFJfSDT7c1RSObrPneTUAjv"

import os
load_dotenv()
os.environ["OPENAI_API_KEY"] == os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Function to generate response using GPT-3.5-turbo model
def generate_response(prompt):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ],
    temperature= 0.7
    )
    #return response.choices[0].text.strip()
    return response.choices[0].message.content

# Streamlit UI
st.title("PRC AI Assistant")

# Form to create IT ticket
st.subheader("Create IT Ticket")
name = st.text_input("Your Name:")
email = st.text_input("Your Email:")
issue_description = st.text_area("Issue Description:")
priority = st.selectbox("Priority:", ["Low", "Medium", "High"])
submit_button = st.button("Submit Ticket")

if submit_button:
    # Save ticket details to database (you can use pandas DataFrame for simplicity)
    ticket_data = pd.DataFrame({"Name": [name], "Email": [email], "Issue Description": [issue_description], "Priority": [priority]})
    st.write("Ticket submitted successfully!")

# Admin interface
if st.checkbox("Admin Interface"):
    st.subheader("View IT Tickets")
    # Display list of IT tickets from the database
    # This is a placeholder, you should replace it with actual data retrieval from your database
    ticket_df = pd.DataFrame(columns=["Ticket ID", "Name", "Email", "Issue Description", "Priority"])
    st.dataframe(ticket_df)

    # Admin response section
    st.subheader("Respond to IT Ticket")
    ticket_id = st.number_input("Enter Ticket ID:")
    admin_response = st.text_area("Admin Response:")
    generate_response_button = st.button("Rephrase PRC")

    if generate_response_button:
        # Prompt template for providing a detailed answer in steps
        prompt = f"Ticket ID: {ticket_id}\nAdmin Response: {admin_response}\n\nPlease provide a detailed response in steps to resolve the issue:\n\n1. First step...\n2. Second step...\n3. Third step"
        response = generate_response(prompt)
        st.write("Generated Response:")
        st.write(response)
        #st.text_area(response)
        
    generate_code_button = st.button("Generate Automation Recipe")
    
    
    if generate_code_button:
        # Prompt template for providing a detailed answer in steps
        prompt = f"Ticket ID: {ticket_id}\nAdmin Response: {admin_response}\n\nPlease provide code for the input query"
        response = generate_response(prompt)
        st.write("Automation Recipe Generation:")
        st.write(response)
        #st.text_area(response)    
    

