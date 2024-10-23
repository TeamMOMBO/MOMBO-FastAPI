import re
from soynlp.hangle import compose, decompose, character_is_korean
from gensim.models import FastText
from Levenshtein import distance as levenshtein

# 공백을 여러 개 입력했을 때 이를 하나로 치환하기 위한 정규식 패턴
doublespace_pattern = re.compile('\s+')


def jamo_sentence(sent):
    """
    자모 분리 함수
    입력된 문자열에서 한글 문자는 자모(초성,중성,종성) 단위로 분리하여 변환
    예시: "한글" -> "ㅎㅏㄴㄱㅡㄹ"
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
    자모 단위로 분리된 문자열을 다시 한글 문자로 결합.
    예시: "ㅎㅏㄴㄱㅡㄹ" -> "한글"
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


def predict_with_similarity(model: FastText, input_name: str, threshold: float = 0.95):
    """
    FastText 모델을 이용해 입력된 단어와 가장 유사한 단어를 예측하는 함수
    :param model: FastText 모델 객체
    :param input_name: 입력된 단어
    :param threshold: 유사도를 판단하는 기준값
    :return: 유사도가 높은 단어가 있으면 해당 단어 반환, 없으면 None 반환
    """
    if input_name in model.wv.key_to_index:
        return input_name

    similar_ingredients = model.wv.most_similar(input_name, topn=1)

    similar_word, similarity = similar_ingredients[0]

    if similarity >= threshold:
        return similar_word

    return None


def levenshtein_distance(word1, word2):
    """
    편집 거리 계산 함수
    두 문자열 간의 편집 거리를 계산하여 반환
    """
    return levenshtein(word1, word2)


def correct_ingredient(model: FastText, ingredient: str, threshold: float = 0.95) -> str:
    """
    자모 분리 후 유사도를 기반으로 재정렬한 단어를 반환하는 함수
    :param model: FastText 모델 객체
    :param ingredient: 입력된 성분명
    :param threshold: 유사도를 판단하는 기준값
    :return: 수정된 성분명
    """
    jamo_ingredient = jamo_sentence(ingredient)
    predicted_output = predict_with_similarity(model, jamo_ingredient, threshold)

    if predicted_output:
        return jamo_to_word(predicted_output)

    # FastText 모델의 모든 단어에서 가장 가까운 단어를 편집 거리를 이용해 찾음
    all_words = model.wv.index_to_key
    closest_word = min(all_words, key=lambda word: levenshtein_distance(jamo_ingredient, word))
    edit_distance = levenshtein_distance(jamo_ingredient, closest_word)

    jamo_ingredient_length = len(jamo_ingredient)

    # 입력된 성분명의 길이에 따라 편집 거리 임계값 설정
    if jamo_ingredient_length <= 10:
        edit_distance_threshold = 1
    else:
        edit_distance_threshold = 5

    # 편집 거리가 임계값을 넘으면 원래 단어를 그대로 반환, 아니면 가까운 단어로 수정하여 반환
    if edit_distance > edit_distance_threshold:
        return jamo_to_word(jamo_ingredient)  # 원래 단어 유지
    else:
        return jamo_to_word(closest_word)  # 편집 거리가 가까운 단어 반환
