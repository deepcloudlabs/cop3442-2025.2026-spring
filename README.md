# COP3442: Prompt Engineering (2025-2026 Spring)

Welcome to the official repository for **COP3442: Prompt Engineering**. 

This course explores the principles, techniques, and practical applications of Large Language Models (LLMs). Students will learn how to design, test, and deploy robust LLM-powered applications, transitioning from basic prompt construction to orchestrating complex, context-aware multi-agent systems. 

This repository contains all course materials, lecture code, assignment templates, and project resources.

---

## Course Modules

The curriculum is structured into eight progressive modules, taking you from the fundamentals of generative AI to advanced agentic architectures.

*   **Module 1: Introduction to LLMs and Prompt Engineering**
    *   Fundamentals of Large Language Models, tokenization, model parameters, and foundational prompting strategies (Zero-shot, Few-shot).
*   **Module 2: Designing LLM Applications**
    *   Architectural considerations, API integrations, use-case analysis, and transitioning from web interfaces to programmatic execution.
*   **Module 3: Prompt Content and Assembling the Prompt**
    *   Managing context windows, dynamic prompt templating, instruction formatting, and system-level prompt engineering.
*   **Module 4: Conversational Agency and LLM Workflows**
    *   Handling multi-turn conversations, managing chat history, state preservation, and structuring linear LLM workflows.
*   **Module 5: Testing LLM Applications**
    *   Evaluation metrics, unit testing for prompts, handling edge cases, and mitigating model hallucinations and prompt injections.
*   **Module 6: Advanced Techniques for Text Generation with LangChain**
    *   Introduction to LangChain, working with chains, document loaders, vector stores, and Retrieval-Augmented Generation (RAG) architectures.
*   **Module 7: Autonomous Agents with Memory and Tools**
    *   Implementing function calling, enabling LLMs to use external tools (APIs, search, calculators), and integrating short-term and long-term memory.
*   **Module 8: Building the Context-Aware Multi-Agent System**
    *   Orchestrating multiple specialized AI agents, defining agent roles and communication protocols, and solving complex, multi-step tasks collaboratively.

---

## 🛠️ Technologies & Tools

This course heavily relies on modern AI development frameworks and tools. Ensure your development environment is set up to support the following:

*   **Language:** Python 3.10+
*   **Frameworks:** LangChain, LangGraph (for multi-agent routing)
*   **APIs & Models:** OpenAI API, Anthropic API, or local execution via Ollama
*   **Environment:** Jupyter Notebooks, VS Code

---

## 🚀 Getting Started

To access the course materials and run the local assignments, follow these steps:

**1. Clone the repository:**
`bash
git clone https://github.com/your-username/COP3442-Prompt-Engineering.git
cd COP3442-Prompt-Engineering
`

**2. Set up a virtual environment (Recommended):**
`bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
`

**3. Install dependencies:**
`bash
pip install -r requirements.txt
`

**4. Environment Variables:**
Rename the `.env.example` file to `.env` and add your specific API keys required for the assignments.
`text
OPENAI_API_KEY=your_api_key_here
`

---

## Academic Integrity & Contributions

*   **Students:** Please do not submit Pull Requests containing solutions to graded assignments or projects. You are encouraged to fork this repository for your own version control during the semester.
*   **Bug Reports:** If you find a typo in the lecture notes or a bug in the starter code, please open an Issue using the repository's issue tracker.

---

## 📄 License

This material is provided for educational purposes under the [MIT License](LICENSE).
