import re
from soynlp.hangle import compose, decompose, character_is_korean
from gensim.models import FastText

doublespace_pattern = re.compile('\s+')


def jamo_sentence(sent):
    """
    자모 분리 함수
    """
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
        return cjj_

    sent_ = []
    for char in sent:
        if character_is_korean(char):
            sent_.append(transform(char))
        else:
            sent_.append(char)
    sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
    return sent_


def jamo_to_word(jamo):
    """
    자모 결합 함수
    """
    jamo_list, idx = [], 0
    while idx < len(jamo):
        if not character_is_korean(jamo[idx]):
            jamo_list.append(jamo[idx])
            idx += 1
        else:
            jamo_list.append(jamo[idx:idx + 3])
            idx += 3
    word = ""
    for jamo_char in jamo_list:
        if len(jamo_char) == 1:
            word += jamo_char
        elif jamo_char[2] == "-":
            word += compose(jamo_char[0], jamo_char[1], " ")
        else:
            word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
    return word


def correct_ingredient(model: FastText, ingredient: str, threshold: float = 0.65) -> str:
    """
    자모 분리 후 FastText 모델을 사용해 성분명을 교정하고, 다시 결합하여 반환
    """
    # 성분명을 자모로 분리
    jamo_ingredient = jamo_sentence(ingredient)

    # 정확하게 일치하는 경우 그대로 반환
    if jamo_ingredient in model.wv.key_to_index:  # 정확하게 일치하는 경우
        return ingredient  # 원래 성분명을 그대로 반환

    # FastText 모델을 이용해 유사도가 가장 높은 단어 찾기
    similar_ingredients = model.wv.most_similar(jamo_ingredient, topn=1)

    if similar_ingredients:
        candidate, similarity = similar_ingredients[0]
        if similarity >= threshold:
            # 교정된 성분명을 다시 자모에서 한글로 변환
            return jamo_to_word(candidate)

    return ingredient
