from gensim.models import FastText

# FastText 모델 로드
model_path = 'models/fasttext'


def load_fasttext_model():
    model = FastText.load(model_path)
    return model
