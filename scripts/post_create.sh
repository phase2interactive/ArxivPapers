python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_lg

# Run this Python code
python - <<EOF
import nltk
nltk.download('punkt')
EOF
