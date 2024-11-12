import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 미리 업로드된 텍스트 파일 경로 지정
text_file_path = "datas/lemonade.txt"

# 텍스트 파일 로드
loader = TextLoader(text_file_path)
data = loader.load()

# 텍스트를 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_data = text_splitter.split_documents(data)

# 텍스트 청크를 벡터로 변환
embeddings = OpenAIEmbeddings()
vectors = FAISS.from_documents(split_data, embeddings)

# ConversationalRetrievalChain 으로 대화형 챗봇 구성
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name="gpt-4"),
    retriever=vectors.as_retriever(),
)


# 사용자의 질문에 대해 대화형으로 답변하는 함수
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state["history"]})
    st.session_state["history"].append((query, result["answer"]))
    return result["answer"]


# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "안녕하세요! 텍스트 파일에 대해 무엇이든 물어보세요!"
    ]

if "past" not in st.session_state:
    st.session_state["past"] = ["안녕하세요!"]

if "loading" not in st.session_state:
    st.session_state["loading"] = False

# 챗봇 이력에 대한 컨테이너
response_container = st.container()
input_container = st.container()

# 입력 폼 생성
with input_container:
    with st.form(key="Conv_Question", clear_on_submit=True):
        user_input = st.text_input(
            "Query:",
            placeholder="텍스트 파일에 대해 이야기 해볼까요? (:",
            key="input",
            disabled=st.session_state["loading"],
        )
        submit_button = st.form_submit_button(label="Send")

    # 사용자가 질문을 입력하거나, [Send] 버튼을 눌렀을 때 처리
    if submit_button and user_input:
        st.session_state["loading"] = True
        st.experimental_rerun()

# 로딩 중일 때 스피너 표시
if "loading" in st.session_state and st.session_state["loading"]:
    with st.spinner("답변을 생성 중입니다..."):
        output = conversational_chat(st.session_state["input"])
        # 사용자의 질문이나 LLM에 대한 결과를 계속 추가(append)
        st.session_state["past"].append(st.session_state["input"])
        st.session_state["generated"].append(output)
        st.session_state["loading"] = False

    # 입력 필드와 버튼을 다시 활성화
    st.experimental_rerun()

# LLM이 답변을 해야 하는 경우에 대한 처리
if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="fun-emoji",
                seed="Nala",
            )
            message(
                st.session_state["generated"][i],
                key=str(i),
                avatar_style="bottts",
                seed="Fluffy",
            )
