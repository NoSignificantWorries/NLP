# %%
# !pip install datasets

# %%
# !python -m spacy download ru_core_news_sm

# %%
from typing import List, Tuple

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import spacy
from datasets import load_dataset

# %%
# Загрузка датасета SIB-200 (русский, кириллица) и подготовка меток
# Возвращает кортежи (тексты, метки) для train/val/test и список названий классов в фиксированном порядке

def load_sib200_ru() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]], List[str]]:
    trainset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='train')
    X_train = trainset['text']
    y_train = trainset['category']
    valset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='validation')
    X_val = valset['text']
    y_val = valset['category']
    testset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='test')
    X_test = testset['text']
    y_test = testset['category']

    # Проверяем, что во валидации/тесте нет новых классов, отсутствующих в трейне
    categories = set(y_train)
    unknown_categories = set(y_val) - categories
    if len(unknown_categories) > 0:
        err_msg = f'The categories {unknown_categories} are represented in the validation set, but they are not represented in the training set.'
        raise RuntimeError(err_msg)
    unknown_categories = set(y_test) - categories
    if len(unknown_categories) > 0:
        err_msg = f'The categories {unknown_categories} are represented in the test set, but they are not represented in the training set.'
        raise RuntimeError(err_msg)

    # Фиксируем порядок классов и переводим строковые метки в индексы
    categories = sorted(list(categories))
    y_train = [categories.index(it) for it in y_train]
    y_val = [categories.index(it) for it in y_val]
    y_test = [categories.index(it) for it in y_test]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), categories

# %%
# Нормализация текста: токенизация spaCy, лемматизация, замена чисел на <NUM>, удаление пунктуации

def normalize_text(s: str, nlp_pipeline: spacy.Language) -> str:
    doc = nlp_pipeline(s)
    lemmas = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if token.like_num:
            lemmas.append('<NUM>')
        elif token.is_stop:  # Удаляем стоп-слова
            continue
        elif token.lemma_.strip():  # Проверяем, что лемма не пустая
            lemmas.append(token.lemma_.lower())
    
    return ' '.join(lemmas) if lemmas else ''

# %%
# Режим ускорения: если True — сильно сокращаем сетку и число фолдов, чтобы обучаться быстрее
FAST_MODE = True

train_data, val_data, test_data, classes_list = load_sib200_ru()

# %%
# print(f'Categories: {classes_list}')

# %%
# print(len(train_data[0]))
# print(len(train_data[1]))

# %%
# print(len(val_data[0]))
# print(len(val_data[1]))

# %%
# print(len(test_data[0]))
# print(len(test_data[1]))

# %%
# Загружаем модель spaCy для русского. Для скорости можно отключить лишние компоненты,
# но в ru_core_news_sm лемматизация зависит от теггера; оставим по умолчанию.
nlp = spacy.load('ru_core_news_sm')

# Предварительная нормализация корпусов для train/val/test
train_norm = [normalize_text(t, nlp) for t in train_data[0]]
val_norm = [normalize_text(t, nlp) for t in val_data[0]]
test_norm = [normalize_text(t, nlp) for t in test_data[0]]

# %%
# print(train_data[0][0])

# %%
# print(normalize_text(train_data[0][0], nlp))

# %%
# print(val_data[0][0])

# %%
# print(normalize_text(val_data[0][0], nlp))

# %%
# print(test_data[0][0])

# %%
# print(normalize_text(test_data[0][0], nlp))

# %%
# max_df вычислялся как 1 - 0.2 * p(class), что практически равно ~0.999 при 200 классах
# и почти не фильтрует частые токены. Оставим расчёт, но дальше дадим сетке перебрать разумные значения.
class_probability = 1.0 / len(classes_list)
max_df = 1.0 - 0.2 * class_probability
# print(f'Maximal document frequency of term is {max_df}.')

# %%
# БАЗОВЫЙ ВАРИАНТ (оставляем закомментированным для сравнения):
# classifier = Pipeline(steps=[
#     ('vectorizer', TfidfVectorizer(token_pattern='\\w+', max_df=max_df, min_df=1)),
#     ('cls', LogisticRegression(solver='saga', max_iter=100, random_state=42))
# ])

# УЛУЧШЕННЫЙ ВАРИАНТ:
# 1) Переносим нормализацию внутрь пайплайна че��ез FunctionTransformer, чтобы избежать утечек при CV.
# 2) Добавляем признаковое объединение: word TF-IDF + char n-gram TF-IDF (char_wb), что часто повышает F1 на русском.
# 3) Включаем sublinear_tf и настраиваем min_df/max_df.
# 4) Увеличиваем max_iter и добавляем опции class_weight/multi_class для LR через сетку.

# preprocess больше не используется внутри Pipeline из-за проблем совместимости и производительности.
# preprocess = FunctionTransformer(_normalize_batch, validate=False)

# Две ветки признаков: слова и символы
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'(?u)\b\w+\b',
    sublinear_tf=True,
    max_df=max_df,
    min_df=1,
    ngram_range=(1, 3)
)
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    sublinear_tf=True,
    ngram_range=(2, 6),
    min_df=1
)
feature_union = FeatureUnion([
    ('word', word_vectorizer),
    ('char', char_vectorizer)
])

# Убираем кэширование шагов пайплайна (Memory) из-за проблем �� сериализацией FunctionTransformer в некоторых окружениях
# Модель по-прежнему сохраняется на диск после обучения.
# Ранее нормализация была шагом пайплайна:
# classifier = Pipeline(steps=[
#     ('preprocess', preprocess),
#     ('features', feature_union),
#     ('cls', LogisticRegression(solver='saga', max_iter=2000, random_state=42))
# ])
# Теперь используем пайплайн без шага preprocess. Нормализуем тексты заранее (см. fit/predict ниже).
classifier = Pipeline(steps=[
    ('features', feature_union),
    ('cls', MultinomialNB())
])

# %%

best_params = {'cls__alpha': [0.025],
               'cls__fit_prior': [False],
               'features__char__ngram_range': [(3, 5)],
               'features__word__max_df': [0.9],
               'features__word__min_df': [1],
               'features__word__ngram_range': [(1, 1)]}

# БАЗОВАЯ СЕТКА (оставлена для справки):
# cv = GridSearchCV(
#     estimator=classifier,
#     param_grid={
#         'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
#         'cls__C': [1e-1, 1, 10, 100, 1000],
#         'cls__penalty': ['l1', 'l2']
#     },
#     scoring='f1_macro',
#     cv=5,
#     refit=True,
#     n_jobs=-1,
#     verbose=True
# )

# НОВАЯ СЕТКА ДЛЯ УЛУЧШЕННОГО ПАЙПЛАЙНА:
# - Тюним n-gram для слов, min_df/max_df, а также силу регуляризации LR и class_weight/multi_class.
# Сформируем сетку и параметры CV в зависимости от FAST_MODE
if FAST_MODE:
    # Ускоренный режим: меньше комбинаций, меньше фолдов
    param_grid = {
        'features__word__ngram_range': [(1, 2), (1, 3)],
        'features__word__min_df': [1, 2],
        'features__word__max_df': [0.95, 0.98, 1.0],
        'features__char__ngram_range': [(3, 6), (2, 6)],
        'cls__alpha': [0.05, 0.1, 0.25, 0.5, 1.0],
        'cls__fit_prior': [True, False],
    }
    cv_folds = 3
    verbose_level = 3  # подробный прогресс
else:
    param_grid = {
        'features__word__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'features__word__min_df': [1, 2, 3, 5],
        'features__word__max_df': [0.9, 0.95, 0.98, 1.0],
        'features__char__ngram_range': [(3, 5), (3, 6), (2, 6)],
        'cls__alpha': [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        'cls__fit_prior': [True, False],
    }
    cv_folds = 5
    verbose_level = 2

cv = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid if best_params is None else best_params,
    scoring='f1_macro',
    cv=cv_folds,
    refit=True,
    n_jobs=-1,
    verbose=verbose_level
)

# %%
# Ранее нормализация считалась вне пайплайна (это могло приводить к несогласованности при CV):
# cv.fit([normalize_text(it, nlp) for it in train_data[0]], train_data[1])
# Теперь нормализация выполняется вне пайплайна один раз для всех текстов
cv.fit(train_norm, train_data[1])

# %%
print('Best parameters:', cv.best_params_)

# %%
print('Best F1-macro:', cv.best_score_)

# Сохраним лучшую модель и её параметры на диск, чтобы не переобучать в следующий раз
# Убрано сохранение на диск по просьбе: не сохраняем модель и параметры

# %%
# Размер словаря теперь является суммой по двум векторизаторам. Для демонстрации выведем размеры по веткам.
best_est = cv.best_estimator_
word_vocab_size = len(best_est.named_steps['features'].transformer_list[0][1].vocabulary_)
char_vocab_size = len(best_est.named_steps['features'].transformer_list[1][1].vocabulary_)
print(f'Word vocab size: {word_vocab_size}; Char vocab size: {char_vocab_size}; Total approx: {word_vocab_size + char_vocab_size}')

# %%
# Предсказание на валидации
y_pred = cv.predict(val_norm)
print(classification_report(y_true=val_data[1], y_pred=y_pred, target_names=classes_list))

# %%
# Предсказание на тесте
y_pred = cv.predict(test_norm)
print(classification_report(y_true=test_data[1], y_pred=y_pred, target_names=classes_list))
cv.fit(train_norm, train_data[1])

# %%
print('Best parameters:', cv.best_params_)

# %%
print('Best F1-macro:', cv.best_score_)

# Сохраним лучшую модель и её параметры на диск, чтобы не переобучать в следующий раз
# Убрано сохранение на диск по просьбе: не сохраняем модель и параметры

# %%
# Размер словаря теперь является суммой по двум векторизаторам. Для демонстрации выведем размеры по веткам.
best_est = cv.best_estimator_
word_vocab_size = len(best_est.named_steps['features'].transformer_list[0][1].vocabulary_)
char_vocab_size = len(best_est.named_steps['features'].transformer_list[1][1].vocabulary_)
print(f'Word vocab size: {word_vocab_size}; Char vocab size: {char_vocab_size}; Total approx: {word_vocab_size + char_vocab_size}')

# %%
# Предсказание на валидации
y_pred = cv.predict(val_norm)
print(classification_report(y_true=val_data[1], y_pred=y_pred, target_names=classes_list))

# %%
# Предсказание на тесте
y_pred = cv.predict(test_norm)
print(classification_report(y_true=test_data[1], y_pred=y_pred, target_names=classes_list))


