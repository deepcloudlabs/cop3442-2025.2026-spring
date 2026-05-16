from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(base_url="192.168.1.114:22026",model="gemma4", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data science teaching assistant. Maintain continuity with the conversation."
    ),
    #MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model | StrOutputParser()

history = [
    HumanMessage(content="We are studying data cleaning."),
    AIMessage(content="Good. Data cleaning includes missing values, duplicate values, outliers, and transformations."),
    HumanMessage(content="We also discussed median imputation"),
    AIMessage(content="Median imputation is robust when distributions are skewed or contain outliers."),
]

result = chain.invoke({
   # "history": history,
    "question": "Now explain how outlier handling relates to median imputation?"
})

print(result)

"""
That's an excellent question, as it connects two critical concepts in data cleaning.

The relationship is one of **robustness** and **best practice**. Essentially, understanding how outliers affect your data dictates *why* median imputation is a superior choice over other methods.

Here is a detailed breakdown:

### 1. The Problem: Outliers and the Mean

When outliers are present, they disproportionately influence measures of central tendency.

*   **The Mean ($\bar{x}$):** The mean is calculated using every single data point. If you have a few extreme outliers (very high or very low values), the mean will be "pulled" toward those outliers. This means the calculated mean no longer accurately represents the typical center of the data.
*   **Example:** Imagine a dataset of salaries: [30k, 35k, 40k, 45k, **1,000k**]. The mean will be significantly inflated by the $1,000k outlier, making it seem like the average salary is much higher than what most people actually earn.

### 2. The Solution: The Median

The median is the middle value when the data is sorted. It is **resistant** (or **robust**) to outliers.

*   **How it works:** Because the median only cares about the *position* of the data points, not their actual magnitude, extreme outliers do not affect its value. In the salary example above, the median would remain close to the center of the bulk of the data, ignoring the extreme outlier.

### 3. The Connection: Why Median Imputation is Preferred

When you have missing values (NaNs) and you suspect the data contains outliers, using the mean to impute those missing values is risky because the imputed value will be artificially skewed by the outliers.

**Therefore, the relationship is:**

> **Outlier handling dictates the choice of imputation method.** If you identify outliers, you should use the median for imputation because it provides a more stable and representative estimate of the missing value than the mean.

***

### 💡 Summary Rule of Thumb

| Scenario | Best Measure of Central Tendency | Imputation Method |
| :--- | :--- | :--- |
| **Data is normally distributed** (symmetrical, no extreme outliers) | Mean | Mean Imputation |
| **Data is skewed or contains outliers** (non-normal) | Median | **Median Imputation** |
"""