FROM rayproject/ray:2.35.0
RUN pip install torch sentence-transformers transformers ray[serve]
COPY embedding_service.py /app/embedding_service.py
WORKDIR /app
