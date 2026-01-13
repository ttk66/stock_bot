import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, ConversationHandler

from models import StockPredictor
from utils import setup_logging, log_request

import os
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния для ConversationHandler
TICKER, AMOUNT = range(2)

class StockTradingBot:
    def __init__(self):
        self.predictor = StockPredictor()
        
    async def start(self, update: Update, context: CallbackContext) -> int:
        """Начало диалога с пользователем."""
        user = update.effective_user
        await update.message.reply_text(
            f"Привет, {user.first_name}!\n"
            "Я - бот для прогнозирования цен акций.\n\n"
            "Введите тикер компании (например, AAPL, MSFT, TSLA):"
        )
        return TICKER
    
    async def get_ticker(self, update: Update, context: CallbackContext) -> int:
        """Получение тикера от пользователя."""
        ticker = update.message.text.upper()
        context.user_data['ticker'] = ticker
        
        # Проверка доступности тикера
        try:
            data = yf.download(ticker, period='1d')
            if data.empty:
                await update.message.reply_text(
                    f"Тикер {ticker} не найден. Пожалуйста, введите корректный тикер:"
                )
                return TICKER
        except Exception:
            await update.message.reply_text(
                f"Ошибка при проверке тикера {ticker}. Пожалуйста, введите корректный тикер:"
            )
            return TICKER
        
        await update.message.reply_text(
            f"Тикер {ticker} принят.\n"
            "Теперь введите сумму для условной инвестиции (в USD):"
        )
        return AMOUNT
    
    async def get_amount(self, update: Update, context: CallbackContext) -> int:
        """Получение суммы инвестиции."""
        try:
            amount = float(update.message.text)
            if amount <= 0:
                await update.message.reply_text(
                    "Сумма должна быть положительной. Введите сумму еще раз:"
                )
                return AMOUNT
            context.user_data['amount'] = amount
        except ValueError:
            await update.message.reply_text(
                "Пожалуйста, введите числовое значение суммы:"
            )
            return AMOUNT
        
        # Начинаем обработку
        await update.message.reply_text(
            "Отлично! Начинаю анализ... Это займет несколько минут."
        )
        
        # Запускаем прогнозирование
        await self.process_prediction(update, context)
        
        return ConversationHandler.END
    
    async def process_prediction(self, update: Update, context: CallbackContext):
        """Основной процесс прогнозирования."""
        ticker = context.user_data['ticker']
        amount = context.user_data['amount']
        user_id = update.effective_user.id
        
        try:
            # Загрузка данных
            await update.message.reply_text("Загружаю исторические данные...")
            data = self.predictor.load_data(ticker)
            
            # Обучение моделей и выбор лучшей
            await update.message.reply_text("Обучаю модели...")
            best_model, best_model_name, metrics = self.predictor.train_and_select_model(data)
            
            # Прогнозирование
            await update.message.reply_text("Строю прогноз...")
            forecast, last_price = self.predictor.make_forecast(best_model, best_model_name, data)
            
            # Генерация рекомендаций
            await update.message.reply_text("Анализирую и формирую рекомендации...")
            recommendations, potential_profit = self.predictor.generate_recommendations(
                forecast, last_price, amount
            )
            
            # Создание графиков
            await update.message.reply_text("Создаю визуализации...")
            plot_buffer = self.predictor.create_plot(data, forecast, ticker, best_model_name)
            
            # Отправка результатов пользователю
            await update.message.reply_photo(
                photo=plot_buffer,
                caption=f"Прогноз для {ticker} (модель: {best_model_name})"
            )
            
            # Формирование сводки
            price_change_pct = ((forecast.iloc[-1] - last_price) / last_price * 100)
            summary = (
                f" **СВОДКА ПО {ticker}**\n\n"
                f" **Период прогноза:** 30 дней\n"
                f" **Лучшая модель:** {best_model_name}\n"
                f" **Точность (RMSE):** {metrics['rmse']:.4f}\n"
                f" **Точность (MAPE):** {metrics['mape']:.2f}%\n\n"
                f" **Текущая цена:** ${last_price:.2f}\n"
                f" **Прогноз через 30 дней:** ${forecast.iloc[-1]:.2f}\n"
                f" **Изменение:** {price_change_pct:+.2f}%\n\n"
                f" **Инвестируемая сумма:** ${amount:.2f}\n"
                f" **Потенциальная прибыль:** ${potential_profit:.2f}\n"
                f" **Доходность:** {(potential_profit/amount*100):.2f}%\n\n"
                f" **Время анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await update.message.reply_text(summary, parse_mode='Markdown')
            
            # Отправка рекомендаций
            await update.message.reply_text(
                recommendations,
                parse_mode='Markdown'
            )
            
            # Логирование
            log_request(
                user_id=user_id,
                ticker=ticker,
                amount=amount,
                best_model=best_model_name,
                metrics=metrics,
                potential_profit=potential_profit
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            await update.message.reply_text(
                f"Произошла ошибка при обработке запроса: {str(e)}\n"
                "Попробуйте еще раз с другим тикером."
            )
    
    async def help_command(self, update: Update, context: CallbackContext) -> None:
        """Отправка справки пользователю."""
        help_text = (
            " **Помощь по использованию бота**\n\n"
            "Этот бот помогает прогнозировать цены акций и дает торговые рекомендации.\n\n"
            "**Доступные команды:**\n"
            "/start - начать новый прогноз\n"
            "/help - показать это сообщение\n"
            "/cancel - отменить текущую операцию\n\n"
            "**Примеры тикеров:**\n"
            "• AAPL - Apple\n"
            "• MSFT - Microsoft\n"
            "• TSLA - Tesla\n"
            "• GOOGL - Alphabet (Google)\n"
            "• AMZN - Amazon\n\n"
            "**Как это работает:**\n"
            "1. Бот загружает исторические данные за 2 года\n"
            "2. Обучает 3 разные модели машинного обучения\n"
            "3. Выбирает лучшую модель по точности\n"
            "4. Строит прогноз на 30 дней\n"
            "5. Дает рекомендации по покупке/продаже"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def cancel(self, update: Update, context: CallbackContext) -> int:
        """Отмена диалога."""
        await update.message.reply_text(
            "Операция отменена. Используйте /start для начала нового анализа."
        )
        return ConversationHandler.END

load_dotenv()

def main():
    """Запуск бота."""
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Создание приложения
    application = Application.builder().token(TOKEN).build()
    
    # Создание бота
    bot = StockTradingBot()
    
    # Настройка ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', bot.start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, bot.get_ticker)],
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, bot.get_amount)],
        },
        fallbacks=[CommandHandler('cancel', bot.cancel)],
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', bot.help_command))
    application.add_handler(CommandHandler('cancel', bot.cancel))
    
    # Запуск бота
    print("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()