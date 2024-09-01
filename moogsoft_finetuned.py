import streamlit as st
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize the LLM model
llm = CTransformers(
    model="model/Llama-2-7b-chat-finetune-rootcause.Q4_K.gguf",
    model_type="llama",
    config={'max_new_tokens': 400, 'temperature': 0.01, 'context_length': 4000}
)

# Instruction template for the LLM
instruction = '''
You are an advanced AI model trained to assist with IT-related inquiries.
Please provide detailed and comprehensive responses to the following IT question.
If the initial response is incomplete, enhance it by adding detailed, step-by-step points or bullet points
based on your fine-tuned knowledge and general IT principles.
Additionally, if the response requires code, provide the necessary code snippets with explanations.
'''

# Define the prompt template
template = f'[INST] {instruction} \n {{resolution}} [/INST]'
prompt = PromptTemplate(template=template, input_variables=["resolution"])

# Initialize the LLMChain
LLM_Chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app layout
st.title("IT Resolution Rephraser")

# Input field for resolution
resolution = st.text_area("Enter resolution:")

# Function to get the rephrased resolution
def get_rephrased_resolution(resolution):
    response = LLM_Chain.run({"resolution": resolution})
    return response

# Rephrased PRC button
if st.button("Rephrased PRC"):
    if resolution:
        rephrased = get_rephrased_resolution(resolution)
        st.text_area("Rephrased resolution preview:", value=rephrased, height=300)
    else:
        st.error("Please enter a resolution to rephrase.")

# Retry button below the rephrased resolution preview
if 'rephrased' in locals() or 'rephrased' in globals():
    if st.button("Retry"):
        if resolution:
            rephrased = get_rephrased_resolution(resolution)
            st.text_area("Rephrased resolution preview:", value=rephrased, height=300)
        else:
            st.error("Please enter a resolution to rephrase.")
