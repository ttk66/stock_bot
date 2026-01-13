import logging
from datetime import datetime
import json
import os

def setup_logging():
    """Настройка системы логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bot.log'),
            logging.StreamHandler()
        ]
    )

def log_request(user_id: int, ticker: str, amount: float, best_model: str, 
                metrics: dict, potential_profit: float):
    """Логирование запроса пользователя."""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': user_id,
        'ticker': ticker,
        'amount': amount,
        'best_model': best_model,
        'metrics': metrics,
        'potential_profit': potential_profit
    }
    
    # Запись в текстовый файл
    with open('logs.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    # Также логируем в консоль
    logging.info(f"Request logged: {log_entry}")

def validate_ticker(ticker: str) -> bool:
    """Проверка корректности тикера."""
    # Базовая проверка: тикер должен состоять из букв и быть длиной 1-5 символов
    if not ticker.isalpha() or not 1 <= len(ticker) <= 5:
        return False
    return True

def format_currency(amount: float) -> str:
    """Форматирование денежной суммы."""
    return f"${amount:,.2f}"