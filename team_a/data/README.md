# Data directory for pipeline input

Place your input files here.

## Testing on 7.2 Glycolysis

To run the full pipeline (PDF → text → entities → triples) on the **7.2 glycolysis** section:

1. Add the textbook PDF or extracted text:
   - **Option A:** Save the 7.2 glycolysis section as `7.2_glycolysis.pdf` in this directory.
   - **Option B:** Save as `7.2_glycolysis.txt` if you already have plain text.

2. From the project root (with venv activated):

   ```bash
   python src/run_all.py data/7.2_glycolysis.pdf
   ```
   or
   ```bash
   python src/run_all.py data/7.2_glycolysis.txt
   ```

   Or use the helper script:
   ```bash
   ./scripts/run_glycolysis.sh
   ```

3. Output: `triples.csv` at project root, plus console summary (entities, valid triples, schema validation).

## Default input

If you run `python src/run_all.py` with no arguments, the pipeline uses `data/passage.txt` or `data/passage.pdf` if present.

## OCR configuration

PDF parsing uses **Mistral OCR 3** via `src/parse_pdf.py`.

- Set `MISTRAL_API_KEY` in your environment (or `.env`).
- Optional: set `MISTRAL_OCR_MODEL` (defaults to `mistral-ocr-latest`).
