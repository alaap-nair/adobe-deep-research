# adobe-deep-research

Do NOT commit your API key.
Use a .env file.
Ensure .env is in .gitignore.
If you accidentally commit a key, revoke it immediately.

Please add a .env file in the root of the repo. In your .env file, add this:
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxx

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Ingest

Build the triples, graph, and vector stores before asking questions:

```bash
python3 src/run_all.py data/passage.txt
```

## Ask

Query the system with the CLI deliverable:

```bash
python3 ask.py "What does glycolisis produce"
```

The command should print JSON with `question`, `answer`, `citations`, and `reasoning`, and also write matching files under `outputs/answers/`.
