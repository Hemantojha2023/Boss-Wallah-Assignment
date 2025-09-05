First i created a folder named Chatbot then created a virtual environment (venv) using python -m venv venv in terminal then activate it by using venv\Scripts\Activate. then created requirements.txt file for 
neccessary libraries and framework and run the command pip install -r requirements.txt in terminal. Also, i created .env file for API keys. Since I have already openai model credits so keep OPENAI_API_KEY="sk-....". 
We can also use hugging face api and impelement model using langchain_huggingface component.

Now, i loaded the dataset "data.csv" using pandas then converted Released Languages column as Hindi for 6, Kannada for 7, Malayalam for 11, Tamil for 20, Telugu for 21, English for 24. Then converted 
this pandas dataframent into document format, output same as CSVLoader i.e docs[0] as row 1, docs[1] as row 2 and so on. which are individually shown in language_conversion.ipynb file only for understanding purpose.

I imported necessary LangChain components in 1.RAG_based_chatbot.py file as ChatOpenAI for gpt-4 model, OpenAIEmbeddings to convert text into numeric vectors, text_splitter(RecurciveCharacter), prompts as 
ChatPromptTemplate, output parser to get output in string, runnables to make chain, memory for multi-turn chat bot, Chroma database to store vector database, and Vector store retriever to retrieve query from 
vector store database. To covert this chatbot as Agentic chatbot i used LangChain agents component to make ReAct based agent.

I created custom tool for RAG system to answer the question related to dataset and send the final output in tool that i got from chain. Then I created a custom web tool to answer the question which are not 
related to dataset. This web tool search from duckduckgo.com, since my limit for duckduckgosearch over so i created custom tool for web search using duckduckgo.com. You can use directly Builtin tool for 
websearch as DuckDuckGoSearchRun if you have credits.

Provided instructions to system so that Agent don't confuse that question belongs to dataset or outside. Initialize Agent and provided both custom tools rag_tool and web_tool and provided memory inside agent as 
ConversationBufferMemory to store previous chats into memory. Also i provided system instructions inside agent.

Finally, created chatbot function to get the agent output inside gradio interface.

This Agentic Chatbot is able to answer both type of question: dataset and websearch type question both.
