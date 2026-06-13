from langchain_core.runnables import RunnableLambda

normalize = RunnableLambda(lambda x: x.strip().lower())
add_prefix = RunnableLambda(lambda x: f"normalized_query: {x}")

pipeline = normalize | add_prefix

print(pipeline.invoke("  LangChain Pipelines  "))
print(pipeline.batch([" RAG ", " Structured Output ", " Streaming "]))
