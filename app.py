import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from loggers import CustomProgbarLogger

from utils import concat_data_pred, plot_feature_importance, plot_scatter
from utils import create_lstm, plot_nn_results, concat_nn_data_pred, convert_df
from utils import download_true, download_false, download_nn_true, download_nn_false
 
cb_feature_cols = ['shop_id', 'item_id', 'category', 'month', 'items_lag_1', 'items_lag_2', 'items_lag_3']
cb_target = 'items'           

app_mode = st.sidebar.selectbox('Выберите модель',['Градиентный бустинг', 'Нейронная сеть'])

if app_mode == 'Градиентный бустинг':
    cb_mode = st.sidebar.selectbox('Выберите режим',['Обучение', 'Предсказания'])

    # Обучение catboost
    if cb_mode == 'Обучение':
        st.header('Обучение модели градиентного бустинга')
        progress_tab, results_tab = st.tabs(["Процесс обучения", "Оценка качества обученной модели"])

         # Загрузка данных
        uploaded_train_data = st.sidebar.file_uploader("Загрузите обучающий датасет", key='cb_train_data')
        if uploaded_train_data is not None:
            cb_train_shape = pd.read_csv(st.session_state['cb_train_data']).shape[1]
            if cb_train_shape != 9:
                st.sidebar.warning("Размерность признаков не совпадает с ожидаемой. Проверьте правильность датасета, которые вы загрузили.")

        uploaded_val_data = st.sidebar.file_uploader("Загрузите валидационный датасет", key='cb_val_data')
        if uploaded_val_data is not None:
            cb_val_shape = pd.read_csv(st.session_state['cb_val_data']).shape[1]
            if cb_val_shape != 9:
                st.sidebar.warning("Размерность признаков не совпадает с ожидаемой. Проверьте правильность датасета, которые вы загрузили.")        


        with st.sidebar.form(key='cb_fit_form'):      
            # Установка гиперпараметров        
            cb_iterations = st.slider('Выберите количество деревьев:', 100, 5000, 1000, 100)
            cb_max_depth = st.slider('Выберите максимальную глубину деревьев:', 1, 15, 4)
            
            fit_button = st.form_submit_button("Обучить модель", disabled=(uploaded_train_data is None) or (uploaded_val_data is None))

        if fit_button:
            if (uploaded_train_data is not None) and (uploaded_val_data is not None):
                with progress_tab:
                    with st.spinner('Обучение градиентного бустинга...'):
                        container = st.empty()
                        cb_train_data = pd.read_csv(uploaded_train_data)
                        X_train = cb_train_data[cb_feature_cols]
                        y_train = cb_train_data[cb_target]

                        regressor = CatBoostRegressor(random_state=0, iterations=cb_iterations,
                                                    max_depth=cb_max_depth, verbose=1,)
                        
                        regressor.fit(X_train, y_train, log_cout=container)
                        st.session_state['catboost'] = regressor             
                    st.success('Модель градиентого бустинга обучена')


                with results_tab:
                    cb_val_data = pd.read_csv(uploaded_val_data)
                    X_val = cb_val_data[cb_feature_cols]
                    y_val = cb_val_data[cb_target]
                    regressor = st.session_state['catboost']
                    train_pred = regressor.predict(X_train)
                    val_pred = regressor.predict(X_val)
                    st.write("RMSE на обучении:", np.sqrt(mean_squared_error(y_train, train_pred)))
                    st.write("RMSE на валидации:", np.sqrt(mean_squared_error(y_val, val_pred)))

                    fig, ax = plot_feature_importance(regressor, X_val)
                    st.pyplot(fig)
                    fig, ax = plot_scatter(y_val, val_pred)
                    st.pyplot(fig)
            else:
                st.sidebar.warning("Загрузите тестовый и валидационный датасет")

    # Предсказания catboost
    if cb_mode == 'Предсказания':
        st.header('Предсказания')

        # Загрузка данных
        uploaded_test_data = st.sidebar.file_uploader("Загрузите тестовый датасет", 
                                                      key='cb_test_features', on_change=download_false, args=(st.session_state,))
        if uploaded_test_data is not None:
            cb_test_shape = pd.read_csv(st.session_state['cb_test_features']).shape[1]
            if cb_test_shape != 7:
                st.sidebar.warning("Размерность признаков не совпадает с ожидаемой. Проверьте правильность датасета, которые вы загрузили.") 
       
        if uploaded_test_data is not None:
            pred_button = st.sidebar.button("Получить предсказания")

            if pred_button:
                if 'catboost' not in st.session_state:
                    st.write('Модель не обучена. Вначале обучите модель')
                else:
                    cb_test_data = pd.read_csv(uploaded_test_data)
                    X_test = cb_test_data[cb_feature_cols]
                    regressor = st.session_state['catboost']
                    test_pred = regressor.predict(X_test)
                    test_data_with_preds = concat_data_pred(cb_test_data, test_pred)
                    st.session_state['cb_test_pred'] = test_data_with_preds
                    st.write(test_data_with_preds)

                    # Скачать датафрейм
                    download_button = st.download_button(
                            label="Скачать в формате CSV",
                            data=convert_df(test_data_with_preds),
                            file_name='catboost_test_pred.csv',
                            mime='text/csv', key='download_cb_test', on_click=download_true, args=(st.session_state,)
                            )
                    
        # Выводим прошлые предсказания, после нажатия на кнопку "Скачать"
        try:
            if not pred_button and st.session_state['download_cb_test_button'] == True:
                st.text("Предыдущие предсказания")
                st.write(st.session_state['cb_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['cb_test_pred']),
                    file_name='catboost_test_pred.csv',
                    mime='text/csv', key='download_cb_test_again', on_click=download_true, args=(st.session_state,)
                    )
            elif not pred_button and "cb_test_pred" in st.session_state.keys():
                st.text("Предыдущие предсказания")
                st.write(st.session_state['cb_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['cb_test_pred']),
                    file_name='catboost_test_pred.csv',
                    mime='text/csv', key='download_cb_test_again', on_click=download_true, args=(st.session_state,)
                    )
        except:
            if "cb_test_pred" in st.session_state.keys():
                st.text("Предыдущие предсказания")
                st.write(st.session_state['cb_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['cb_test_pred']),
                    file_name='catboost_test_pred.csv',
                    mime='text/csv', key='download_cb_test_again', on_click=download_true, args=(st.session_state,)
                    )


# Нейронная сеть
elif app_mode == 'Нейронная сеть':
    nn_mode = st.sidebar.selectbox('Выберите режим',['Обучение', 'Предсказания'])

    # Обучение нейронной сети
    if nn_mode == 'Обучение':
        st.header('Обучение нейронной сети')
        progress_tab, results_tab = st.tabs(["Процесс обучения", "Оценка качества обученной модели"])

        # Загрузка данных
        uploaded_train_data = st.sidebar.file_uploader("Загрузите обучающий датасет", key='nn_train_data')
        if uploaded_train_data is not None:
            nn_train_shape = pd.read_csv(st.session_state['nn_train_data']).shape[1]
            if nn_train_shape != 34:
                st.sidebar.warning("Размерность признаков не совпадает с ожидаемой. Проверьте правильность датасета, которые вы загрузили.") 

        with st.sidebar.form(key='nn_form'):      
            # Установка гиперпараметров        
            output_neurons = st.slider('Выберите количество нейронов скрытого слоя:', 1, 100, 64)
            batch_size =  st.slider('Выберите размер батча:', 32, 4096, 4096, 32)
            epochs = st.slider('Выберите количество эпох обучения:', 1, 20, 15, 1)
            optimizer = st.radio("Выберите метод оптимизации", ["adam", "rmsprop"])
            
            fit_button = st.form_submit_button("Обучить модель", disabled=(uploaded_train_data is None))

        if fit_button:
            if uploaded_train_data is not None:
                nn_train_data = pd.read_csv(uploaded_train_data)  
                X_train = np.expand_dims(nn_train_data.values[:,:-1], axis = 2)
                y_train = nn_train_data.values[:,-1:]                

                with progress_tab:
                    container_epoch = st.empty()
                    container = st.empty()
                    lstm = create_lstm(output_neurons, optimizer)
                    with st.spinner('Обучение нейронной сети...'):
                        history = lstm.fit(
                                X_train,
                                y_train,
                                epochs=epochs, 
                                batch_size=batch_size, # 4096 1024 2048
                                verbose=1, 
                                shuffle=True,
                                validation_split=0.3,
                                callbacks=[CustomProgbarLogger(container=container, container_epoch=container_epoch)])
                        st.session_state['lstm'] = lstm
                    st.success('Нейрононная сеть обучена')                    

                with results_tab:
                    st.write(f"RMSE на обучении: {np.sqrt(history.history['mean_squared_error'][-1])}")
                    st.write(f"RMSE на валидации: {np.sqrt(history.history['val_mean_squared_error'][-1])}")

                    fig, ax = plot_nn_results(history)
                    st.pyplot(fig)

    # Предсказания нейронной сети
    if nn_mode == 'Предсказания':
        st.header('Предсказания модели')

        # Загрузка данных
        uploaded_test_data = st.sidebar.file_uploader("Загрузите тестовый датасет", 
                                                      key='nn_test_data', on_change=download_nn_false, args=(st.session_state,))
        if uploaded_test_data is not None:
            nn_test_shape = pd.read_csv(st.session_state['nn_test_data']).shape[1]
            if nn_test_shape != 35:
                st.sidebar.warning("Размерность признаков не совпадает с ожидаемой. Проверьте правильность датасета, которые вы загрузили.") 

        if uploaded_test_data is not None:
            predict_button = st.sidebar.button("Получить предсказания")   
            if predict_button:
                nn_test_data = pd.read_csv(uploaded_test_data)
                X_test = nn_test_data.drop(nn_test_data.columns[[0, 1]], axis=1).fillna(0)
                X_test = np.expand_dims(X_test, axis = 2)

                if 'lstm' not in st.session_state:
                    st.write('Модель не обучена. Вначале обучите модель')
                else:
                    lstm = st.session_state['lstm']
                    with st.spinner('Формирование предсказаний...'):
                        test_pred = lstm.predict(X_test)
                    nn_test_data_with_preds = concat_nn_data_pred(nn_test_data, test_pred)
                    st.session_state['nn_test_pred'] = nn_test_data_with_preds
                    st.write(nn_test_data_with_preds)

                    # Скачать датафрейм
                    download_button = st.download_button(
                            label="Скачать в формате CSV",
                            data=convert_df(nn_test_data_with_preds),
                            file_name='nn_test_pred.csv',
                            mime='text/csv', key='download_nn_test', on_click=download_nn_true, args=(st.session_state,)
                            )
        # Выводим прошлые предсказания, после нажатия на кнопку "Скачать"
        try:
            if not predict_button and st.session_state['download_nn_test_button'] == True:
                st.text("Предыдущие предсказания")
                st.write(st.session_state['nn_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['nn_test_pred']),
                    file_name='nn_test_pred.csv',
                    mime='text/csv', key='download_nn_test_again', on_click=download_nn_true, args=(st.session_state,)
                    )
            elif not predict_button and "nn_test_pred" in st.session_state.keys():
                st.text("Предыдущие предсказания")
                st.write(st.session_state['nn_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['nn_test_pred']),
                    file_name='nn_test_pred.csv',
                    mime='text/csv', key='download_nn_test_again', on_click=download_nn_true, args=(st.session_state,)
                    )
        except:
            if "nn_test_pred" in st.session_state.keys():
                st.text("Предыдущие предсказания")
                st.write(st.session_state['nn_test_pred'])
                download_button = st.download_button(
                    label="Скачать в формате CSV",
                    data=convert_df(st.session_state['nn_test_pred']),
                    file_name='nn_test_pred.csv',
                    mime='text/csv', key='download_nn_test_again', on_click=download_nn_true, args=(st.session_state,)
                    )


                    
                
                

                

                
        