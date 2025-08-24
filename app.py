import streamlit as st
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import AgentType,initialize_agent
#Used to provide step by step action taken by tthe agents,provides logs also
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_groq import ChatGroq

#Arxiv and Wikipedia tools
api_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv=ArxivQueryRun(api_wrapper=api_arxiv)

api_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki=WikipediaQueryRun(api_wrapper=api_wiki)
#Used to search on internet
search=DuckDuckGoSearchRun(name='Search')


#Sreamlit app title
st.title('Langchain-Chat with Search')

#Sidebar settings
st.sidebar.title('Settings')
api_key=st.sidebar.text_input("Enter your Groq Api Key: ",type='password')


if "messages" not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant',
         'content':'Hi ,I am a chatbot who can search the web. How can I assist you today!'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if api_key:
    if prompt:=st.chat_input(placeholder='What is machine learning?'):
        st.session_state.messages.append({'role':'user','content':prompt})
        st.chat_message('user').write(prompt)

        model=ChatGroq(model='llama3-8b-8192',groq_api_key=api_key,streaming=True)

        tools=[arxiv,search,wiki]

        search_agent=initialize_agent(tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handle_parsing_errors=True)
        
        with st.chat_message('assistant'):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write(response)
