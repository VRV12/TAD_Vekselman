import pickle

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def main():
    model = load_model("model.pkl")
    test_data = load_test_data("preprocessed_data.csv")
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )



    if page == "Описание задачи и данных": # (V)
        st.title("Описание задачи и данных")
        st.write("Для выбора другой страницы выберите другую страницу слева")

        st.header("Описание задачи")
        st.markdown("""
        Данный датасет содержит в себе характеристики, связанные с возникновением пожаров и отображающие реальные показатели детектеров дыма.
        Данные датасет используется для разработки устройств обнаружения дыма и определения, возник пожар или нет, на основе искусственного интеллекта.
        Объем набора данных составляет почти 60 000 показаний. 
        Чтобы отслеживать данные, к каждому показанию датчика добавляется временная метка UTC.""")

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:
* UTC – Временная метка в секундах UTC,
* Temperature – Температура воздуха, измеряемая в градусах Цельсия,
* Humidity – Влажность воздуха,
* TVOC – Общее количество летучих органических соединений; измеряется в ppb (частях на миллиард),
* eCo2 – эквивалентная концентрация CO2; рассчитывается на основе различных значений, таких как TVCO,
* Raw H2 – необработанный молекулярный водород; не компенсирован (смещение, температура и т.д.),
* Raw Ethanol – сырой этанольный газ,
* Pressure – Давление воздуха,
* PM1.0 – Твердые частицы диаметром менее 1,0 микрометра,
* PM2.5 – Твердые частицы диаметром менее 2,5 микрометра,
* NC0.5 – Концентрация твердых частиц диаметром менее 0,5 микрометра.
* NC1.0 - Концентрация твердых частиц диаметром менее 1,0 микрометра
* NC2.5 - Концентрация твердых частиц диаметром менее 2,5 микрометров
* CNT - Простой подсчет
* Fire Alarm - (реальное значение) 1 при пожаре и 0 при пожаре""")




    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Для выбора другой страницы выберите другую страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["матрица путаницы", "5 первых предсказанных значений", "Исходная таблица", "Пользовательский пример"]
        )

        if request == "матрица путаницы": # (V)
            st.header("матрица путаницы")
            X_pain = test_data.drop(columns = ['Fire Alarm'])
            y_pain = test_data['Fire Alarm']
            X_pain = StandardScaler().fit_transform(X_pain)
            matrix = confusion_matrix(y_pain, model.predict(X_pain))
            st.write(matrix)

        elif request == "5 первых предсказанных значений": # (V)
            st.header("5 предсказанных значений")
            X_pain = test_data.drop(columns = ['Fire Alarm'])
            y_pain = test_data['Fire Alarm']
            X_pain = StandardScaler().fit_transform(X_pain)
            first_5_test = X_pain[:5][:]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                st.write(f"{item:.2f}")

        elif request == "Пользовательский пример": #(V)
            st.header("Пользовательский пример")

            utc = st.number_input("Введите характеристику UTC (рекомендуется ввести значение в промежутке [-0.73, 3.07])")

            temperature = st.number_input("Введите характеристику Temperature (рекомендуется ввести значение в промежутке [-2.65, 3.06])")

            humidity = st.number_input("Введите характеристику Humidity (рекомендуется ввести значение в промежутке [-4.27, 3.01])")
            
            TVOC = st.number_input("Введите характеристику TVOC (рекомендуется ввести значение в промежутке [-0.25, 7.46])")
            
            eCO2 = st.number_input("Введите характеристику eCO2 (рекомендуется ввести значение в промежутке [-0.14, 31.15])")
            
            Raw_H2 = st.number_input("Введите характеристику Raw H2 (рекомендуется ввести значение в промежутке [-8.35, 3.16])")
            
            Raw_Ethanol = st.number_input("Введите характеристику Raw Ethanol (рекомендуется ввести значение в промежутке [-7.28, 2.72])")
            
            Pressure = st.number_input("Введите характеристику Pressure (рекомендуется ввести значение в промежутке [-5.84, 0.93])")
            
            PM1_0 = st.number_input("Введите характеристику PM1.0 (рекомендуется ввести значение в промежутке [-0.11, 15.43])")
            
            PM2_5 = st.number_input("Введите характеристику PM2.5 (рекомендуется ввести значение в промежутке [-0.09, 22.95])")
            
            NC0_5 = st.number_input("Введите характеристику NC0.5 (рекомендуется ввести значение в промежутке [-0.12, 14.30])")
            
            NC1_0 = st.number_input("Введите характеристику NC1.0 (рекомендуется ввести значение в промежутке [-0.09, 23.40])")
            
            NC2_5 = st.number_input("Введите характеристику NC2.5 (рекомендуется ввести значение в промежутке [-0.07, 27.64])")
            
            CNT = st.number_input("Введите характеристику CNT (рекомендуется ввести значение в промежутке [-1.38, 1.91])")

            if st.button('Предсказать'):
                data = [utc, temperature, humidity, TVOC, eCO2, Raw_H2, Raw_Ethanol, Pressure, PM1_0, PM2_5, NC0_5, NC1_0, NC2_5, CNT]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"Предсказанное значение: {pred[0]:.2f}")
            else:
                pass

        elif request == "Исходная таблица": # (V)
            st.header("Вывод данных")
            st.write("Вывод первых 10 строк исходного датасета")
            st.write(test_data.head(10))


@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";")
    return df


if __name__ == "__main__":
    main()
