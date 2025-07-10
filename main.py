import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import warnings


def load_data():
    """Загрузка данных: ручной ввод или из файла."""
    print("Выберите способ ввода данных:")
    print("1 - Ввести вручную")
    print("2 - Загрузить из файла (CSV или Excel)")
    choice = input("Ваш выбор (1/2): ")

    if choice == '1':
        print("Введите первую выборку через пробел или запятую:")
        data1_input = input().replace(',', ' ').split()
        print("Введите вторую выборку через пробел или запятую:")
        data2_input = input().replace(',', ' ').split()

        # Проверка на числовые данные
        try:
            data1 = list(map(float, data1_input))
            data2 = list(map(float, data2_input))
        except ValueError:
            raise ValueError("данные должны быть числовыми!")

        # Проверка на пустые выборки
        if len(data1) == 0 or len(data2) == 0:
            raise ValueError("выборки не могут быть пустыми!")

        return np.array(data1), np.array(data2)

    elif choice == '2':
        file_path = input("Введите путь к файлу: ")

        # Чтение файла
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Формат файла не поддерживается. Используйте CSV или Excel.")
        except Exception as e:
            raise ValueError(f"Ошибка при чтении файла: {e}")

        print("Доступные столбцы:", list(df.columns))
        col1 = input("Выберите столбец для первой выборки: ")
        col2 = input("Выберите столбец для второй выборки: ")

        # Проверка наличия столбцов
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError("Один или оба указанных столбца отсутствуют в файле!")

        # Извлечение данных и проверка на числовые значения
        try:
            sample1 = pd.to_numeric(df[col1].dropna()).values
            sample2 = pd.to_numeric(df[col2].dropna()).values
        except ValueError:
            raise ValueError("данные в столбцах должны быть числовыми!")

        # Проверка на пустые выборки
        if len(sample1) == 0 or len(sample2) == 0:
            raise ValueError("после удаления пропусков одна или обе выборки пусты!")

        return sample1, sample2

    else:
        raise ValueError("Неверный выбор. Введите 1 или 2.")


def analyze_samples(sample1, sample2):
    """Анализ выборок и рекомендация теста."""
    # Размеры выборок
    n1, n2 = len(sample1), len(sample2)
    print(f"\nРазмеры выборок: {n1} и {n2}")

    # Проверка нормальности
    _, p1 = shapiro(sample1)
    _, p2 = shapiro(sample2)
    normal = p1 > 0.05 and p2 > 0.05
    print(f"Тест Шапиро-Уилка на нормальность: p-значения = {p1:.3f}, {p2:.3f}")
    print("Выборки нормально распределены." if normal else "Выборки НЕ нормально распределены.")

    # Если нормальные, проверяем дисперсии
    if normal:
        _, p_var = levene(sample1, sample2)
        equal_var = p_var > 0.05
        print(f"Тест Левена на равенство дисперсий: p-значение = {p_var:.3f}")
        print("Дисперсии равны." if equal_var else "Дисперсии НЕ равны.")

    # Рекомендация теста
    if normal:
        if equal_var:
            test_name = "t-тест Стьюдента (двухвыборочный с равными дисперсиями)"
            test_func = ttest_ind
        else:
            test_name = "t-тест Уэлча (двухвыборочный с неравными дисперсиями)"
            test_func = lambda x, y: ttest_ind(x, y, equal_var=False)
    else:
        test_name = "U-тест Манна-Уитни"
        test_func = mannwhitneyu

    print(f"\nРекомендованный тест: {test_name}")

    # Применение теста
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value = test_func(sample1, sample2)

    print(f"\nРезультаты теста:")
    print(f"Статистика = {stat:.3f}, p-значение = {p_value:.3f}")

    # Интерпретация p-значения
    alpha = 0.05
    if p_value < alpha:
        print("Есть статистически значимые различия между выборками (p < 0.05).")
    else:
        print("Нет статистически значимых различий между выборками (p >= 0.05).")


def main():
    print("Анализ двух независимых выборок")
    print("------------------------------")

    try:
        sample1, sample2 = load_data()
        analyze_samples(sample1, sample2)
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()