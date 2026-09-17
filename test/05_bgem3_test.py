from app.lm.embedding_utils import generate_embeddings

texts = ["helloworld", "你好，我是张三"]
print(generate_embeddings(texts))