import nltk

nltk.download('punkt')
text = "안전상의 이유로 익명을 요구한 북한 내 소식통은 “지금보다도 더 먹을 것을 줄여가며 2020년대 중반까지 버티라는 일방적 지시에 주민들의 불신과 원성은 그 어느 때보다 팽배해 있다”고 RFA에 전했다"
token = nltk.word_tokenize(text)
print(token)
