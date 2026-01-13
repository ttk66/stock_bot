import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class StockPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models_metrics = {}
        
    def load_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """Загрузка исторических данных."""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"Не удалось загрузить данные для тикера {ticker}")
            
        # Используем только цену закрытия
        df = df[['Close']].copy()
        df.columns = ['price']
        df = df.asfreq('B')  # Бизнес-дни
        df = df.fillna(method='ffill')
        
        return df
    
    def create_features(self, df: pd.DataFrame, n_lags: int = 30) -> pd.DataFrame:
        """Создание признаков для ML моделей."""
        df_features = df.copy()
        
        # Лаговые признаки
        for lag in range(1, n_lags + 1):
            df_features[f'lag_{lag}'] = df_features['price'].shift(lag)
        
        # Технические индикаторы
        df_features['ma_7'] = df_features['price'].rolling(window=7).mean()
        df_features['ma_30'] = df_features['price'].rolling(window=30).mean()
        df_features['std_30'] = df_features['price'].rolling(window=30).std()
        
        # Временные признаки
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        
        # Целевая переменная (цена через 1 день)
        df_features['target'] = df_features['price'].shift(-1)
        
        return df_features.dropna()
    
    def train_ml_model(self, df: pd.DataFrame, model_type: str = 'random_forest'):
        """Обучение классической ML модели."""
        # Создание признаков
        df_features = self.create_features(df)
        
        # Разделение на train/test
        split_idx = int(len(df_features) * 0.8)
        train = df_features.iloc[:split_idx]
        test = df_features.iloc[split_idx:]
        
        # Подготовка данных - запоминаем имена колонок
        feature_columns = [col for col in train.columns if col not in ['price', 'target']]
        
        X_train = train[feature_columns]
        y_train = train['target']
        X_test = test[feature_columns]
        y_test = test['target']
        
        # Создаем новый scaler для этой модели
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение модели
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Прогноз
        y_pred = model.predict(X_test_scaled)
        
        # Метрики
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Возвращаем модель, scaler и feature_columns
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'type': model_type
        }, {'rmse': rmse, 'mape': mape}
    
    def train_arima_model(self, df: pd.DataFrame):
        """Обучение ARIMA модели."""
        # Разделение на train/test
        split_idx = int(len(df) * 0.8)
        train = df['price'].iloc[:split_idx]
        test = df['price'].iloc[split_idx:]
        
        # Обучение ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Прогноз
        forecast_steps = len(test)
        y_pred = model_fit.forecast(steps=forecast_steps)
        
        # Метрики
        rmse = np.sqrt(mean_squared_error(test, y_pred))
        mape = mean_absolute_percentage_error(test, y_pred) * 100
        
        return {
            'model': model_fit,
            'scaler': None,
            'feature_columns': None,
            'type': 'ARIMA'
        }, {'rmse': rmse, 'mape': mape}
    
    def train_prophet_model(self, df: pd.DataFrame):
        """Обучение Prophet модели с обработкой часовых поясов."""
        try:
            # Подготовка данных для Prophet
            prophet_df = df.reset_index()[['Date', 'price']].copy()
            prophet_df.columns = ['ds', 'y']
            
            if hasattr(prophet_df['ds'].dtype, 'tz'):
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            elif prophet_df['ds'].dt.tz is not None:
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            
            print(f"Тип данных ds: {prophet_df['ds'].dtype}")
            
            split_idx = int(len(prophet_df) * 0.8)
            train = prophet_df.iloc[:split_idx]
            test = prophet_df.iloc[split_idx:]
            
            # Обучение Prophet с отключением ненужных сезонов
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            # Добавляем дополнительные регрессоры, если нужно
            # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            model.fit(train)
            
            # Прогноз
            future = model.make_future_dataframe(periods=len(test), freq='B')
            forecast = model.predict(future)
            
            y_pred = forecast['yhat'].iloc[split_idx:].values
            y_test = test['y'].values
            
            # Метрики
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            print(f"Prophet обучен: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            
            return {
                'model': model,
                'scaler': None,
                'feature_columns': None,
                'type': 'Prophet'
            }, {'rmse': rmse, 'mape': mape}

            
        except Exception as e:
            print(f"Ошибка в Prophet: {e}")
            # Возвращаем заглушку
            return None, {'rmse': float('inf'), 'mape': float('inf')}
    
    def train_lstm_model(self, df: pd.DataFrame):
        """Обучение LSTM модели."""
        # Подготовка данных
        data = df['price'].values.reshape(-1, 1)
        
        scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Создание последовательностей
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 30
        X, y = create_sequences(data_scaled, seq_length)
        
        # Разделение на train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Построение LSTM модели
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Обучение
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Прогноз
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Метрики
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': None,
            'type': 'LSTM'
            }, {'rmse': rmse, 'mape': mape}
    
    def train_and_select_model(self, df: pd.DataFrame):
        """Обучение всех моделей и выбор лучшей."""
        models_to_train = [
            ('Random Forest', self.train_ml_model, {'model_type': 'random_forest'}),
            ('ARIMA', self.train_arima_model, {}),
            ('Prophet', self.train_prophet_model, {}),
            ('LSTM', self.train_lstm_model, {})
        ]
        
        best_model_data = None
        best_model_name = None
        best_metrics = {'rmse': float('inf'), 'mape': float('inf')}
        
        for model_name, model_func, kwargs in models_to_train:
            try:
                print(f"Обучение модели: {model_name}")
                result = model_func(df, **kwargs)
                
                if result is None:
                    print(f"  Модель {model_name} не вернула результат")
                    continue
                    
                # Разбираем результат в зависимости от модели
                if len(result) == 2:
                    model_or_data, metrics = result
                    
                    # Сохраняем метрики
                    self.models_metrics[model_name] = metrics
                    
                    # Выбор модели с наименьшим RMSE
                    if metrics['rmse'] < best_metrics['rmse']:
                        best_metrics = metrics
                        best_model_data = model_or_data
                        best_model_name = model_name
                        print(f"  Новая лучшая модель: {model_name} (RMSE: {metrics['rmse']:.2f})")
                else:
                    print(f"  Модель {model_name} вернула некорректный результат")
                    
            except Exception as e:
                print(f"  Ошибка при обучении {model_name}: {str(e)[:100]}...")
                import traceback
                traceback.print_exc()
                continue
        
        # Проверяем, что хоть одна модель обучена
        if best_model_data is None:
            raise ValueError("Не удалось обучить ни одну модель. Проверьте данные.")
        
        print(f"\nЛучшая модель выбрана: {best_model_name} с RMSE: {best_metrics['rmse']:.2f}")
        
        return best_model_data, best_model_name, best_metrics
    
    def make_forecast(self, model_data, model_name: str, df: pd.DataFrame, days: int = 30):
        """Построение прогноза на 30 дней."""
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        last_price = df['price'].iloc[-1]
        
        if model_name == 'Random Forest':
            # Извлекаем модель, scaler и feature_columns
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Создаем финальный датафрейм с признаками
            df_features = self.create_features(df)
            last_row = df_features.iloc[-1:]
            
            # Убедимся, что все колонки присутствуют
            X_last = last_row[feature_columns]
            X_last_scaled = scaler.transform(X_last)
            
            # Рекурсивный прогноз
            forecasts = []
            current_features = X_last.copy()
            
            for i in range(days):
                # Масштабируем текущие признаки
                current_scaled = scaler.transform(current_features)
                pred = model.predict(current_scaled)[0]
                forecasts.append(pred)
                
                # Обновляем признаки для следующего шага
                # Сдвигаем лаги на 1 вперед
                new_features = {}
                for col in feature_columns:
                    if col.startswith('lag_'):
                        lag_num = int(col.split('_')[1])
                        if lag_num == 1:
                            new_features[col] = pred
                        else:
                            prev_lag = f'lag_{lag_num-1}'
                            new_features[col] = current_features[prev_lag].values[0]
                    elif col == 'ma_7':
                        # Для упрощения используем предыдущее значение
                        new_features[col] = current_features[col].values[0]
                    elif col == 'ma_30':
                        new_features[col] = current_features[col].values[0]
                    elif col == 'std_30':
                        new_features[col] = current_features[col].values[0]
                    elif col in ['day_of_week', 'month', 'quarter']:
                        # Для временных признаков - прогнозируем вперед
                        if col == 'day_of_week':
                            future_date = df.index[-1] + pd.Timedelta(days=i+1)
                            new_features[col] = future_date.dayofweek
                        elif col == 'month':
                            future_date = df.index[-1] + pd.Timedelta(days=i+1)
                            new_features[col] = future_date.month
                        elif col == 'quarter':
                            future_date = df.index[-1] + pd.Timedelta(days=i+1)
                            new_features[col] = (future_date.month - 1) // 3 + 1
                    else:
                        # Для других признаков используем предыдущее значение
                        new_features[col] = current_features[col].values[0]
                
                current_features = pd.DataFrame([new_features])
            
            forecast_series = pd.Series(forecasts, index=pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=days,
                freq='B'
            ))
            
        elif model_name == 'ARIMA':
            forecast = model.forecast(steps=days)
            forecast_series = pd.Series(
                forecast,
                index=pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=days,
                    freq='B'
                )
            )
            
        elif model_name == 'Prophet':
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            forecast_series = forecast['yhat'].iloc[-days:].reset_index(drop=True)
            forecast_series.index = pd.date_range(
                start=df.index[-1] + timedelta(days=1),
                periods=days,
                freq='B'
            )
            
        elif model_name == 'LSTM':
            # Подготовка последней последовательности
            data_scaled = self.scaler.transform(df['price'].values.reshape(-1, 1))
            last_sequence = data_scaled[-30:].reshape(1, 30, 1)
            
            # Рекурсивный прогноз
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                forecasts.append(pred)
                
                # Обновление последовательности
                new_sequence = np.roll(current_sequence[0], -1, axis=0)
                new_sequence[-1, 0] = pred_scaled
                current_sequence = new_sequence.reshape(1, 30, 1)
            
            forecast_series = pd.Series(forecasts, index=pd.date_range(
                start=df.index[-1] + timedelta(days=1),
                periods=days,
                freq='B'
            ))
        
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
        
        return forecast_series, last_price
    
    def generate_recommendations(self, forecast: pd.Series, last_price: float, amount: float):
        """Генерация торговых рекомендаций и расчет прибыли."""
        # Поиск локальных минимумов и максимумов
        min_indices = []
        max_indices = []
        
        for i in range(1, len(forecast) - 1):
            if forecast.iloc[i] < forecast.iloc[i-1] and forecast.iloc[i] < forecast.iloc[i+1]:
                min_indices.append(i)
            elif forecast.iloc[i] > forecast.iloc[i-1] and forecast.iloc[i] > forecast.iloc[i+1]:
                max_indices.append(i)
        
        # Формирование рекомендаций
        recommendations = "**ТОРГОВЫЕ РЕКОМЕНДАЦИИ**\n\n"
        
        if min_indices:
            recommendations += "**Дни для покупки (локальные минимумы):**\n"
            for idx in min_indices[:3]:  # Показываем первые 3 минимума
                date = forecast.index[idx].strftime('%d.%m.%Y')
                price = forecast.iloc[idx]
                recommendations += f"• {date} - ${price:.2f}\n"
            recommendations += "\n"
        
        if max_indices:
            recommendations += "**Дни для продажи (локальные максимумы):**\n"
            for idx in max_indices[:3]:  # Показываем первые 3 максимума
                date = forecast.index[idx].strftime('%d.%m.%Y')
                price = forecast.iloc[idx]
                recommendations += f"• {date} - ${price:.2f}\n"
        
        # Расчет потенциальной прибыли
        if min_indices and max_indices:
            # Берем первый минимум и первый максимум после него
            first_min_idx = min_indices[0]
            first_max_after_min = next((idx for idx in max_indices if idx > first_min_idx), None)
            
            if first_max_after_min:
                buy_price = forecast.iloc[first_min_idx]
                sell_price = forecast.iloc[first_max_after_min]
                
                shares = amount / buy_price
                revenue = shares * sell_price
                profit = revenue - amount
            else:
                profit = 0
        else:
            profit = 0
        
        return recommendations, profit
    
    def create_plot(self, historical: pd.DataFrame, forecast: pd.Series, ticker: str, model_name: str):
        """Создание графика с историческими данными и прогнозом."""
        plt.figure(figsize=(12, 6))
        
        # Исторические данные
        plt.plot(historical.index, historical['price'], 
                label='Исторические данные', color='blue', linewidth=2)
        
        # Прогноз
        plt.plot(forecast.index, forecast.values,
                label=f'Прогноз ({model_name})', color='red', linewidth=2, linestyle='--')
        
        # Заполнение области прогноза
        plt.fill_between(forecast.index,
                        forecast.values * 0.95,
                        forecast.values * 1.05,
                        color='red', alpha=0.1)
        
        plt.title(f'Прогноз цен акций {ticker} на 30 дней', fontsize=16, fontweight='bold')
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Цена ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохранение в буфер
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        return buf