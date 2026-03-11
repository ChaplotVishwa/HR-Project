pip install paddlepaddle==3.2.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleocr spacy
python -m spacy download en_core_web_sm
pip install fastapi uvicorn python-multipart nest-asyncio torch PyMuPDF pydantic huggingface_hub spacy scikit-learn nltk python-dateutil nest_asyncio

python -m app.main