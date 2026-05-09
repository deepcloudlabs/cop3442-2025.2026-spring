import json
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Literal


class FHIRAnswer(BaseModel):
    valueString: str = Field(
        description="Clinically meaningful answer text"
    )


class FHIRItem(BaseModel):
    linkId: str = Field(description="Stable Item Identifier")
    text: str = Field(description="Clinically anamnesis question or section")
    answer: List[FHIRAnswer]


class AnamnesisFHIR(BaseModel):
    resourceType: Literal["QuestionnaireResponse"] = "QuestionnaireResponse"
    status: Literal["completed"] = "completed"
    authored: str
    subject: dict
    item: List[FHIRItem]


loader = PyPDFLoader("clinical_guideline.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./clinical_guideline_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10
    }
)

llm = ChatOllama(
    base_url="192.168.1.111:12026",
    model="medgemma",
    temperature=0.7
)

structured_llm = llm.with_structured_output(AnamnesisFHIR)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a clinical documentation assistant.
        Generate a disease specific anamnesis document as HL7 FHIR JSON.
        Use the FHIR QuestionnaireResponse resource
        Important constraints:
        - Do not provide diagnosis or treatment
        - Do not invent patient identity
        - Include chief complaint, history of present illness, onset, duration, severity, history of present illness, medication history, allergy history, red flags, past medical history, social history, risk factors, clinical summary
        - Return ONLY structured data matching the schema
        """
    ),
    (
        "human",
        """
        Disease: {disease},
        Patient reference: {patient_reference},
        PDF context:
        {context}
        
        Generate the anamnesis document
        """
    )
])


def generate_anamnesis_fhir(disease: str, patient_reference: str = "Patient/example") -> dict:
    chain = prompt | structured_llm
    docs = retriever.invoke(disease)
    context = "\n\n".join(doc.page_content for doc in docs)
    result: AnamnesisFHIR = chain.invoke({
        "disease": disease,
        "patient_reference": patient_reference,
        "context": context
    })

    fhir_json = result.model_dump()

    fhir_json["authored"] = datetime.now(timezone.utc).isoformat()
    fhir_json["subject"] = {
        "reference": patient_reference
    }

    return fhir_json


if __name__ == "__main__":
    result = generate_anamnesis_fhir("asthma")

    print(json.dumps(result, indent=2, ensure_ascii=False))
