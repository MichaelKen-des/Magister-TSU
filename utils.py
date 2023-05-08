import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import streamlit as st
import numpy as np

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False)


def download_true(session_state):
    session_state['download_cb_test_button'] = True


def download_false(session_state):
    session_state['download_cb_test_button'] = False


def download_nn_true(session_state):
    session_state['download_nn_test_button'] = True


def download_nn_false(session_state):
    session_state['download_nn_test_button'] = False


def concat_data_pred(cb_test_data, predictions):
    test_features_with_preds = cb_test_data.copy()
    test_features_with_preds['predictions'] = predictions
    # Удаляем ненужные столбцы
    test_features_with_preds_clean = test_features_with_preds\
        .drop(columns=['category', 'month', 'items_lag_1', 'items_lag_2', 'items_lag_3'])
    return test_features_with_preds_clean


def concat_nn_data_pred(nn_test_data, test_pred):
    submission = pd.DataFrame(
        {'shop_id': nn_test_data['shop_id'], 
            'item_id': nn_test_data['item_id'], 
            'item_cnt_month': test_pred.ravel()})
    return submission


def plot_feature_importance(regressor, X):
    fig, ax = plt.subplots()
    pd.DataFrame({'Feature Importances': regressor.feature_importances_}, 
                 X.columns).sort_values(by='Feature Importances').plot.barh(grid=True, ax=ax)
    ax.set_title('CatBoost Feature Importances')
    return fig, ax


def plot_scatter(y, val_pred):
    fig, ax = plt.subplots()
    ax.scatter(val_pred, y)
    ax.plot(val_pred, val_pred, color='r')
    ax.set_title('Истинное и предсказанное количество продаж')
    ax.set_xlabel('Предсказанные значения')
    ax.set_ylabel('истинные значения')
    ax.grid()
    return fig, ax


def create_lstm(output_neurons, optimizer):
    model = tf.keras.models.Sequential()    
    model.add(tf.keras.layers.LSTM(output_neurons, 
                                   input_shape=(33, 1), 
                                   return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        loss = 'mse',
        optimizer = optimizer, 
        metrics = ['mean_squared_error']       
        )
    return model


def plot_nn_results(history):
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(history.history['mean_squared_error']), label='Обучение')
    ax.plot(np.sqrt(history.history['val_mean_squared_error']), label='Валидация')
    ax.grid()
    ax.set_xlabel("Количество эпох")
    ax.set_ylabel("MSE")
    ax.set_title("Результаты обучения нейронной сети")
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    return fig, ax