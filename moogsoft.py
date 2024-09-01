import streamlit as st
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
#from langchain import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationChain


embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',                                  
                                 model_kwargs={'device':'cpu'})


#persist_directory = "db"
#persist_directory = "db2"
persist_directory = "db3"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)



# Initialize the LLM model
llm = CTransformers(
    model="model/Llama-2-7b-chat-finetune-rootcause.Q4_K.gguf",
    model_type="llama",
    config={'max_new_tokens': 400, 'temperature': 0.01, 'context_length': 4000}
)


memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
# retriever=vectordb.as_retriever(search_kwargs={"k":3})
# qa = ConversationalRetrievalChain.from_llm(llm,                                             
#                                            memory=memory)

chain = ConversationChain(llm=llm)


# Instruction template for the LLM
instruction = '''You are an advanced AI model trained to elaborate IT resolution in detailed points. Your task is to expand 
upon the provided IT resolution by offering a detailed and step by step response based upon the context and you knowledge. 

Context: {docs}
IT Resolution:
{resolution}




Ensure your response includes step-by-step bullet points based on your knowledge and the given context. Follow these guidelines:

1. Step-by-step Elaboration: Break down the IT resolution into detailed steps, providing at least 5-6 points.
2. Completion: If the initial IT resolution is incomplete, finish it with detailed, step-by-step points based on your knowledge.
3. Code Snippets: If the IT resolution has code column, provide the necessary code snippets with clear and detailed explanations.
'''


retriever = vectordb.as_retriever(search_kwargs={"k":2})


# B_INST, E_INST= "[INST]", "[/INST]"
# B_SYS, E_SYS = "<>\n", "\n<>\n\n"

# DEFAULT_SYSTEM_PROMPT="""\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """

# SYSTEM_PROMPT=B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS






# Streamlit app layout
st.title("PRC AI Assistant")

# Input field for resolution
resolution = st.text_area("Enter resolution:")
 
# Function to get the rephrased resolution
def get_rephrased_resolution(resolution, instruction):
    docs = retriever.get_relevant_documents(resolution)
    temp_doc=[]
    for i in docs:
        temp_doc.append(i.page_content)
    print(temp_doc)
    instruction = instruction.format(docs=temp_doc, resolution=resolution)
    print(instruction)
    template = '[INST] {instruction}  [/INST]'.format(instruction=instruction)
    #template = B_INST + SYSTEM_PROMPT + instruction + resolution + E_INST
    print(template)
    result = chain.invoke(template)
    print(f"Result - {result}")
    return result['response']


# Rephrased PRC button
if st.button("Rephrased PRC"):
    if resolution:
        rephrased = get_rephrased_resolution(resolution, instruction)
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
