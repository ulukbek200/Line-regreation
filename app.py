import streamlit as st          # фреймворк для создания веб-приложения из Python-кода
import pandas as pd             # работа с табличными данными (CSV/Excel), фильтрация, агрегаты
import numpy as np              # численные операции и генерация случайных данных
from sklearn.model_selection import train_test_split   # разбиение данных на train/test
from sklearn.preprocessing import StandardScaler       # нормализация признаков (приведение к общему масштабу)
from sklearn.compose import ColumnTransformer          # применяет трансформации к выбранным колонкам
from sklearn.pipeline import Pipeline                  # «конвейер»: препроцессинг + модель в одной цепочке
from sklearn.linear_model import LinearRegression      # линейная регрессия (основная модель)
from sklearn.metrics import r2_score, mean_absolute_error  # метрики качества (R² и MAE)
import matplotlib.pyplot as plt                        # построение графиков
import os                                              # работа с файлами/путями (проверяем наличие CSV)

st.set_page_config(
    page_title="Car Price Linear Regression (Kaggle)",  # заголовок вкладки браузера
    layout="wide"                                       # широкая вёрстка на весь экран
)

st.sidebar.title("Навигация")  # заголовок панели слева
# Радио-переключатель между двумя страницами приложения
page = st.sidebar.radio("Страница", ["Главная", "Датасет"])

# -------------------------- ЗАГРУЗКА ДАННЫХ --------------------------
@st.cache_data  # кэшируем результат, чтобы не перечитывать файл при каждом ререндере
def load_data():
    """
    Пытаемся загрузить реальный Kaggle-файл из папки проекта:
      - CarP.csv  (именно такое имя ты сохранил)
    Если его нет — генерируем синтетический датасет со схожими признаками,
    чтобы приложение не падало и можно было показать работу модели.
    """
    if os.path.exists("CarP.csv"):
        df = pd.read_csv("CarP.csv")  # читаем CSV в DataFrame
        source = "kaggle"              # пометка: источник — реальный CSV
    else:
        # --- синтетический запасной вариант ---
        rng = np.random.default_rng(42)    # генератор случайных чисел с фиксированным сидом
        N = 1000                           # размер выборки
        # создаём несколько правдоподобных признаков автомобиля
        df = pd.DataFrame({
            "enginesize": rng.integers(70, 350, size=N),
            "horsepower": rng.integers(50, 300, size=N),
            "carwidth": rng.uniform(60, 75, size=N),
            "carlength": rng.uniform(150, 205, size=N),
            "curbweight": rng.integers(1500, 4000, size=N),
            "wheelbase": rng.uniform(85, 120, size=N),
            "citympg": rng.integers(10, 45, size=N),
            "highwaympg": rng.integers(15, 55, size=N),
        })
        # строим «цену» как линейную комбинацию признаков + немного шума
        price = (
            100 * df["enginesize"]
            + 120 * df["horsepower"]
            + 800 * (df["carwidth"] - 65)
            + 20 * (df["carlength"] - 170)
            + 2.0 * df["curbweight"]
            + 50 * (df["wheelbase"] - 95)
            - 150 * (df["citympg"] - 25)
            - 120 * (df["highwaympg"] - 30)
            + rng.normal(0, 1500, size=N)  # гауссов шум
            + 8000
        )
        df["price"] = price.round(2)
        source = "synthetic"               # пометка: источник — синтетический
    return df, source

# вызываем загрузку данных (df — таблица; source — флаг источника)
df, source = load_data()

# -------------------------- ВЫБОР ПРИЗНАКОВ ПОД KAGGLE --------------------------
# Целевая колонка в kaggle-наборе — 'price'.
# Берём сильные числовые признаки (линейная регрессия их «любит»).
NUM_FEATURES = [
    # геометрия/масса
    "wheelbase", "carlength", "carwidth", "carheight", "curbweight",
    # двигатель/характеристики
    "enginesize", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm",
    # расход
    "citympg", "highwaympg",
    # иногда встречается индекс безопасности/класса
    "symboling"
]

# Оставляем только те фичи, которые реально присутствуют в загруженном df
available_features = [c for c in NUM_FEATURES if c in df.columns]

# Ищем целевую колонку: в kaggle — 'price'; в запасных вариантах может быть 'price_usd'
target_col = "price" if "price" in df.columns else "price_usd"

# Если вдруг целевой колонки нет (кривой CSV) — создадим её эвристически,
# чтобы интерфейс продолжал работать и можно было показать пайплайн
if target_col not in df.columns:
    base = 10000.0
    if "enginesize" in df.columns:
        base += 120 * df["enginesize"]
    if "horsepower" in df.columns:
        base += 100 * df["horsepower"]
    df[target_col] = base

# -------------------------- ФУНКЦИЯ: СОЗДАТЬ И ОБУЧИТЬ МОДЕЛЬ --------------------------
def make_model(df):
    """
    1) Оставляем только выбранные колонки + целевую.
    2) Дропаем пропуски (NaN).
    3) Строим пайплайн: StandardScaler (нормализация) -> LinearRegression (модель).
    4) Делим на train/test, обучаем и считаем метрики R² и MAE.
    """
    work = df.copy()

    # итоговый список колонок для X (только те, что есть)
    cols = [c for c in available_features if c in work.columns]
    cols = list(dict.fromkeys(cols))  # на всякий случай удаляем дубликаты
    work = work[cols + [target_col]].dropna()  # выбрасываем строки с пропусками

    # Матрица признаков X и целевая переменная y
    X = work[cols]
    y = work[target_col]

    # Препроцессинг: стандартизируем только числовые колоноки из cols
    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), cols)],
        remainder="drop",   # все остальные колонки выбрасываем
    )

    # Пайплайн: препроцессинг -> линейная регрессия
    model = Pipeline([
        ("preproc", preproc),
        ("lr", LinearRegression())
    ])

    # Разбиение на обучающую и тестовую выборки (20% на тест)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение модели на тренировочных данных
    model.fit(X_train, y_train)

    # Предсказания на тесте
    preds = model.predict(X_test)

    # Метрики качества:
    r2 = r2_score(y_test, preds)             # доля объяснённой вариации (0..1)
    mae = mean_absolute_error(y_test, preds) # средняя абсолютная ошибка в единицах цены

    return model, (X_test, y_test, preds, r2, mae), cols

# обучаем модель и получаем всё нужное для визуализации
model, eval_pack, feature_names = make_model(df)
X_test, y_test, preds, r2, mae = eval_pack

# -------------------------- СТРАНИЦА: ГЛАВНАЯ --------------------------
if page == "Главная":
    st.title("Линейная регрессия: прогноз цены авто (Kaggle)")

    # Сообщение о источнике данных (реальный Kaggle или синтетический)
    if source == "kaggle":
        st.success("✅ Загружен реальный датасет: CarP.csv")
    else:
        st.warning("⚠️ Kaggle-файл не найден — используется синтетический датасет.")

    # Верхние ключевые метрики + статус порога 0.90 для R²
    c1, c2, c3 = st.columns(3)
    c1.metric("R² на тесте", f"{r2:.3f}")      # качество модели (чем ближе к 1, тем лучше)
    c2.metric("MAE", f"{mae:,.0f}")            # средняя ошибка прогноза
    status = "✅ OK (≥0.90)" if r2 >= 0.90 else "⚠️ Ниже 0.90"
    c3.metric("Порог качества", status)        # индикатор выполнения твоего ТЗ

    # --------- График: фактическая цена vs прогноз ---------
    st.subheader("График: Факт vs Прогноз")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.6)       # точки (y_true, y_pred)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx])                # диагональ = идеальный прогноз
    ax.set_xlabel("Фактическая цена")
    ax.set_ylabel("Прогноз")
    st.pyplot(fig)

    # --------- Таблица коэффициентов (в исходном масштабе признаков) ---------
    st.subheader("Пояснения к факторам (коэффициенты модели)")
    lr = model.named_steps["lr"]                                    # доступ к LinearRegression внутри пайплайна
    scaler = model.named_steps["preproc"].named_transformers_["num"]# доступ к StandardScaler
    # Коэффициенты lr обучались на стандартизированных фичах,
    # делим на scale_, чтобы приблизить эффект к «1 единице» исходного признака
    coefs = lr.coef_ / scaler.scale_
    expl_df = pd.DataFrame(
        {"feature": feature_names, "coef_effect_per_unit": coefs}
    ).sort_values("coef_effect_per_unit", key=np.abs, ascending=False)
    st.dataframe(expl_df, use_container_width=True)

    # Справка для пользователя/преподавателя
    with st.expander("Как читать график и таблицу?"):
        st.markdown(
            f"- Точки ближе к диагонали → точнее прогноз.  \n"
            f"- **R²={r2:.3f}** — доля объяснённой вариации цены.  \n"
            f"- Положительный коэффициент повышает цену, отрицательный — понижает."
        )

    # --------- Калькулятор прогноза (ручной ввод признаков) ---------
    st.divider()
    st.subheader("Калькулятор прогноза цены (ввод признаков вручную)")
    inputs = {}
    cols = st.columns(4)  # раскладываем поля ввода в 4 колонки
    for i, feat in enumerate(feature_names):
        # Значение по умолчанию — медиана по колонке (более устойчиво к выбросам)
        default = float(df[feat].median()) if pd.api.types.is_numeric_dtype(df[feat]) else 0.0
        inputs[feat] = cols[i % 4].number_input(feat, value=float(default))

    if st.button("Предсказать цену"):
        # создаём DataFrame из одного наблюдения и подаём в пайплайн
        row = pd.DataFrame([{k: v for k, v in inputs.items()}])
        pred = model.predict(row)[0]
        st.success(f"Оценочная цена: **{pred:,.0f}**")

    # --------- Блок с подсказками/промптом для анализа ---------
    st.divider()
    st.subheader("Промпт по теме продажи машин")
    st.markdown("""
- *Как влияют enginesize и horsepower на цену?*  
- *Что будет с ценой, если увеличить carwidth и curbweight?*  
- *Насколько экономичность (citympg/highwaympg) снижает цену?*
""")

# -------------------------- СТРАНИЦА: ДАТАСЕТ --------------------------
elif page == "Датасет":
    st.title("Датасет")
    st.caption("Просмотр и скачивание текущих данных.")
    st.dataframe(df, use_container_width=True)  # показываем весь текущий DataFrame



# desition tree