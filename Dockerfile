FROM smizy/scikit-learn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader all

WORKDIR /app