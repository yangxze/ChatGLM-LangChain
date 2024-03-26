# 一、Reference

> 1. [GitHub - ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B?tab=readme-ov-file)
> 2. [ChatGLM+LangChain实践](https://www.bilibili.com/video/BV1t8411y7fp?p=1)
> 3. [ChatGLM+LangChain实践  GitHub项目代码](https://github.com/IronSpiderMan/MachineLearningPractice)
> 4. [ChatGLM+LangChain官方框架讲解](https://www.bilibili.com/video/BV13M4y1e7cN/?spm_id_from=333.788.recommend_more_video.5&vd_source=a2b906f1078e767936dd0bbcf1275e2e)



推荐两个国产LLM智能助手，大家如果其中代码有不懂的直接问LLM。

> 1. [kimi](https://kimi.moonshot.cn/)
> 2. [通义千问 ](https://tongyi.aliyun.com/qianwen/?spm=5176.28326591.0.0.40f713f4hDPj4r)





# 二、本地环境准备

​	安装 [Anaconda](https://www.anaconda.com/download/)，具体操作参考文档 Python 深度学习：安装 Anaconda 与 PyTorch（GPU 版）库.pdf。文件来源于b站视频[Python深度学习：安装Anaconda、PyTorch（GPU版）库与PyCharm_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1cD4y1H7Tk/?spm_id_from=333.337.search-card.all.click&vd_source=a2b906f1078e767936dd0bbcf1275e2e)





### 2.1虚拟环境

进入刚刚安装的anaconda Powershell Prompt

```cmd
# 虚拟环境列表
conda env list

# 创建虚拟环境ChatBot
conda create -n ChatBot python=3.10

# 删除虚拟环境ChatBot
# conda remove -n ChatBot --all

# 进入/离开虚拟环境（如果需要安装库，进入对应虚拟环境安装）
conda activate ChatBot  / conda deactivate

# 列出 Jupyter 的虚拟环境连接列表，
jupyter kernelspec list

# 安装 ipykernel
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ipykernel

# 将虚拟环境导入 Jupyter 的 kernel 中
python -m ipykernel install --user --name=ChatBot

# 删除对应连接
jupyter kernelspec remove ChatBot 

#显卡信息 
nvidia -smi
```



### 2.2项目环境安装

```cmd
# torch-GPU
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 安装requirements环境
pip install -r requirements.txt
```





### 2.3文件下载

```cmd
# 进入E盘program文件夹中
cd E:\Jupyter\Program\

# 建立项目文件夹ChatGLM-LangChain
mkdir ChatGLM-LangChain

# 进入项目文件夹
cd ChatGLM-LangChain

# 下载模型参数chatglm-6b-int4，可以进入对应url查看模型
git clone https://huggingface.co/THUDM/chatglm-6b-int4

# 显存大可以下载模型chatglm-6b，
git clone https://huggingface.co/THUDM/chatglm-6b
```



​	安装cuda toolkit[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)。如果不安装的话chatglm-6b-int4模型会报错，以及chatglm-6b.quantize(4)也会报错。



```cmd
#win+r然后输入cmd进入命令行
# 输入以下命令进入jupyter notebook
jupyter lab
```



# 三、项目代码Step

​	新建jupyter notebook文件，命名为ChaBot。环境为开始创建的虚拟环境ChaBot。

### 3.1测试模型

```python
import warnings
warnings.filterwarnings("ignore")

# 加载模型
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()

# 测试输入
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```





### API

​	单独在同一目录建立py文件api.py

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("./chatglm-6b-int4", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
```

​	然后

```python
# 此时有两种方法运行api.py文件

#1.在jupyter lab中打开命令行
#2.通过Anaconda powershell prompt进入命令行（不是win+r cmd的）。conda activate ChatBot进入ChatBot虚拟环境，cd E:\Jupyter\Program\ChatGLM-LangChain进入项目文件夹

# 运行前面创建的api.py
python api.py
```



### API测试

​	进入ChatBot.ipynp输入以下代码测试api

```python
import requests
def chat(prompt, history) :
    resp = requests .post(
        url = 'http://127.0.0.1:8000',
        json = {"prompt": prompt,"history": history},
        headers = {"Content-Type": "application/json;charset=utf-8"}
    )
    return resp.json()['response'], resp.json()['history']
             
history = []
while True:
    response, history = chat(input("Question:"), history)
    print('Answer:', response)
```



### 文档划分

```python
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
def load_documents(directory="./Books/"):
    """
    加载books下的文件，进行拆分
    :param directory:
    :return:
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=35, chunk_overlap=10)
    split_docs = text_spliter.split_documents(documents)
    return split_docs
split_docs = load_documents()
split_docs
```



### 加载Embedding模型

```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
}


def load_embedding_model(model_name="text2vec3"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
embeddings = load_embedding_model()
embeddings
```



### 向量库chrome

```python
from langchain.vectorstores import Chroma
import os
def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    讲文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db

# 加载数据库
if not os.path.exists('VectorStore'):
    split_docs = load_documents()
    db = store_chroma(split_docs, embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
```



### Prompt_QA

```python
from langchain.llms import ChatGLM
# 创建llm
llm = ChatGLM(
    endpoint_url='http://127.0.0.1:8000',
    max_token=80000,
    top_p=0.9
)

from langchain.prompts import PromptTemplate
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
如果你不知道答案，就回答不知道，不要试图编造答案。答案最多3句话，保持答案简介。
总是在答案结束时说”谢谢你的提问！“{context}
问题：{question}""")
print(QA_CHAIN_PROMPT)

from langchain.chains import RetrievalQA

# search_kwargs={"k": 1}设计检索的文档数量
retriever = db.as_retriever(search_kwargs={"k": 1}) 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


```



### Test

```python
qa.run('可以笑一个吗')
```





### Gradio_UI



```python
import gradio as gr
import time
def chat(question, history):
    response = qa.run(question)
    return response
demo = gr.ChatInterface(chat)
demo.launch(inbrowser=True)
```



换成以下代码增加上传文档功能

```python
# 增加上传文档功能
# https://www.gradio.app/docs/chatbot

import gradio as gr
import time
def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    """
    上传文件后的回调函数，将上传的文件向量化存入数据库
    :param history:
    :param file:
    :return:
    """
    global qa
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    db = store_chroma(documents, embeddings)
    retriever = db.as_retriever()
    qa.retriever = retriever
    history = history + [((file.name,), None)]
    return history


def bot(history):
    """
    聊天调用的函数
    :param history:
    :return:
    """
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功！！"
    else:
        response = qa({"query": message})['result']
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname('__file__'), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=['txt'])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
```





