from typing import List
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from models.dataset import Dataset
from langchain.schema import Document
from langchain import OpenAI
import logging


refine_prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "这是原始问题: {question}\n"
    "已有的回答: {existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    "\n\n"
    "{context_str}\n"
    "\\nn"
    "请根据新的文段，进一步完善回答，给出完整的答案。如果这些文段跟原始问题无关，保持已有回答不变。"
    "除特殊字符外用中文回答\n\n"
    "### Response: "
)

initial_qa_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 用中文回答这个问题：{question}。\n\n"
    "### Response: "
)


class DocumentQAChain():
    docs = List[Document]

    def __init__(self, llm: OpenAI, docs: list[Document]) -> BaseCombineDocumentsChain:
        self.docs = docs
        self.llm = OpenAI
        refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=refine_prompt_template,
        )
        initial_qa_prompt = PromptTemplate(
            input_variables=["context_str", "question"], template=initial_qa_template
        )
        # output intermediate_steps, output_text
        self.chain = load_qa_chain(llm, chain_type="refine", return_refine_steps=True,
                                   question_prompt=initial_qa_prompt, refine_prompt=refine_prompt)

    def run(self, question: str) -> str:
        if self.chain == None:
            raise (ValueError("DocumentQAChain no llm chain "))
        print(f"qa chain start question:\n{question}")
        response = self.chain(
            {"input_documents": self.docs, "question": question}, return_only_outputs=True)
        print(f"doc qa chain resp:\n{response}")
        return response.get('output_text')
