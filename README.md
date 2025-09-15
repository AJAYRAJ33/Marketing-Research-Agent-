
Steps to Run the Project Locally
To run this FastAPI project on your local machine, follow these steps.

1. Prerequisites
First, make sure you have the following installed:

Python 3.8+: The project is built with Python.

Git: To clone the repository from GitHub.

A text editor (like VS Code or Sublime Text).

2. Get the Code
Open your terminal or command prompt and clone the repository you created on GitHub. Replace the URL with your project's URL.

Bash

git clone https://github.com/your-username/your-repo-name.git
Next, navigate into the project directory:

Bash

cd your-repo-name
3. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies.

Create a virtual environment:

Bash

python -m venv .venv
Activate the virtual environment:

On Windows:

Bash

.venv\Scripts\activate
On macOS/Linux:

Bash

source .venv/bin/activate
Install dependencies: After activating the environment, install the required Python packages. You'll need to create a requirements.txt file if you haven't already.

Bash

pip install fastapi uvicorn python-dotenv langchain_core langchain_groq langgraph langchain_huggingface langchain_community langchain_text_splitters pydantic
Once these are installed, you can generate your requirements.txt file for future use:

Bash

pip freeze > requirements.txt
Now, to install dependencies in the future, you would use:

Bash

pip install -r requirements.txt
4. Configure API Key
The application requires a GROQ API key.

Create a file named .env in your project's root directory.

Add your API key to this file in the following format:

GROQ_API_KEY="your_groq_api_key_here"

5. Run the Application
Finally, start the FastAPI server using uvicorn. The reload flag will automatically restart the server when you make code changes.

Bash

uvicorn app:app --reload
The server will now be running on your local machine. You can access the application by opening your web browser and navigating to:

UI: http://127.0.0.1:8000

API Docs: http://127.0.0.1:8000/docs (for testing the API endpoints like /chat and /upload_pdf)
