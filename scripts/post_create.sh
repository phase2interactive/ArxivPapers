sudo apt-get update
sudo apt-get install -y ffmpeg
sudo apt-get install -y texlive-latex-base
sudo apt-get install -y texlive-latex-extra
sudo apt-get install -y texlive-fonts-extra
sudo apt-get install -y texlive-science
sudo apt-get install -y latex2html
sudo apt-get install -y texlive-publishers
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli

pip install -r requirements.txt

python -m spacy download en_core_web_lg

# Run this Python code
python - <<EOF
import nltk
nltk.download('punkt')
EOF
