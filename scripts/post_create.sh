sudo apt-get update
sudo apt-get install -y ffmpeg
sudo apt-get install -y texlive-latex-base
sudo apt-get install -y texlive-latex-extra
sudo apt-get install -y texlive-fonts-extra
sudo apt-get install -y texlive-science
sudo apt-get install -y latex2html

pip install -r requirements.txt

python -m spacy download en_core_web_lg

# Run this Python code
python - <<EOF
import nltk
nltk.download('punkt')
EOF
