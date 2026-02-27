# adobe-deep-research

Do NOT commit your API key.
Use a .env file.
Ensure .env is in .gitignore.
If you accidentally commit a key, revoke it immediately.

Please add a .env file in the root of the repo. In your .env file, add this:
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx


Install:
pip install python-dotenv

"from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment")"