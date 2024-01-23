import sys
import time
import requests
import pandas as pd
import ccxt
import talib as ta
import numpy as np
from talib import CDLDOJI, CDLHAMMER, CDLINVERTEDHAMMER, CDLLONGLEGGEDDOJI
from binance.client import Client
import os
import configparser
from concurrent.futures import ThreadPoolExecutor
import logging
import json

# Paramètres configurables
CONFIG_FILE = 'config.ini'
CONDITION_ACHAT_PERCENTAGE = 1.5
WAIT_TIME_SECONDS = 5
PROFIT_TRANSFER_PERCENTAGE = 0.5
ECHLLE_TEMPS = '3m'
MAX_WORKERS = 8

# Initialisation du client Binance
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

api_key = config.get('API', 'API_KEY')
api_secret = config.get('API', 'API_SECRET')

# Configuration du module de journalisation
logging.basicConfig(filename='trading_log.log', level=logging.DEBUG)

# Vérifier si les clés sont définies
if not (api_key and api_secret):
    logging.error("Les clés API ne sont pas définies dans les variables d'environnement.")
    raise ValueError("Les clés API ne sont pas définies dans les variables d'environnement.")

# Initialisation de l'exchange Binance avec clé API et secret
exchange = ccxt.binance({
    'rateLimit': 100,
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret
})

# Initialize the Binance client
client = Client(api_key, api_secret)

# Liste des symboles pour lesquels des ordres d'achat ont été exécutés
symboles_achetes = set()

# Récupération de l'adresse du wallet externe depuis le fichier de configuration
with open('config.txt', 'r') as config_file:
    external_wallet_address = config_file.read().strip()

# Fonction pour attendre l'exécution complète de l'ordre de vente
def wait_for_sell_order(symbol, sell_order_id):
    while True:
        try:
            # Récupérer l'état de l'ordre de vente
            sell_order_status = exchange.fetch_order(symbol=symbol, id=sell_order_id)['status']

            if sell_order_status == 'closed':
                print(f"Ordre de vente pour {symbol} complètement exécuté.")
                break

            # Attendre un certain temps avant de vérifier à nouveau
            time.sleep(WAIT_TIME_SECONDS)

        except ccxt.NetworkError as ne:
            handle_error(f"Erreur réseau lors de la récupération de l'état de l'ordre pour {symbol}: {ne}")
        except ccxt.ExchangeError as ee:
            handle_error(f"Erreur d'échange lors de la récupération de l'état de l'ordre pour {symbol}: {ee}")
        except Exception as e:
            handle_error(f"Erreur inattendue lors de la récupération de l'état de l'ordre pour {symbol}: {e}")

# Fonction pour transférer une partie de la plus-value vers le wallet externe
def transfer_profit_to_external_wallet(symbol, buy_price, current_price, buy_quantity):
    try:
        # Calcul de la plus-value
        profit = (current_price - buy_price) * buy_quantity

        if profit > 0:
            # Transférer une partie de la plus-value vers le wallet externe
            withdrawal_amount = profit * PROFIT_TRANSFER_PERCENTAGE

            # Effectuer le retrait sur le réseau BSC
            withdrawal_result = client.sapi_post('/v1/asset/withdraw', {
                'asset': symbol.replace('USDT', ''),
                'address': external_wallet_address,
                'amount': withdrawal_amount,
                'network': 'BSC'
            })

            logging.info(f"Transfert réussi de {withdrawal_amount} {symbol.replace('USDT', '')} vers le wallet externe sur BSC.")
            logging.debug("Résultat du retrait: %s", withdrawal_result)

    except ccxt.NetworkError as ne:
        handle_error(f"Erreur réseau lors du transfert de fonds pour {symbol}: {ne}")
    except ccxt.ExchangeError as ee:
        handle_error(f"Erreur d'échange lors du transfert de fonds pour {symbol}: {ee}")
    except Exception as e:
        handle_error(f"Erreur inattendue lors du transfert de fonds pour {symbol}: {e}")

# Fonction d'analyse technique
def analyze_technicals(ohlcv_df):
    close_prices = ohlcv_df['close'].values
    high_prices = ohlcv_df['high'].values
    low_prices = ohlcv_df['low'].values

    n = 10  # ou une autre valeur appropriée

    # Sélectionner uniquement les tokens en Momentum haussier
    if not any(close_prices[-1] > close_prices[-n] for n in [5, 10, 20]):
        return False

    # Éviter les tokens en retournement sur William's Percent Range
    williams_range = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
    if williams_range[-1] < -80:
        return False

    # Identifie et sélectionne uniquement les tokens avec un MACD haussier
    macd, signal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    if macd[-1] <= signal[-1]:
        return False

    return True

# Fonction pour traiter chaque chunk de symboles
def process_symbol_chunk(chunk, investment_amount, df):
    for symbol in chunk:
        process_symbol(symbol, investment_amount, df)


# Fonction de traitement pour chaque symbole
def process_symbol(symbol, investment_amount, df):
    try:
        print(f"\n------ Traitement du token {symbol} ------")

        # Récupération des données OHLCV
        ohlcv = exchange.fetch_ohlcv(symbol, ECHLLE_TEMPS, limit=100)

        # Création d'un DataFrame avec les données historiques
        ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Analyse de la variation sur 5 secondes
        ohlcv_df['price_change_5s'] = (ohlcv_df['close'].diff() / ohlcv_df['close'].shift(1)) * 100

        # Vérification des conditions d'achat
        if ohlcv_df['price_change_5s'].iloc[-2] > CONDITION_ACHAT_PERCENTAGE:  # Utiliser l'indice -2 pour obtenir la valeur précédente

            # Conditions d'analyse technique supplémentaires
            if not analyze_technicals(ohlcv_df):
                print(f"{symbol} ne satisfait pas aux conditions d'analyse technique. Éviter cet achat.")
                return

            print(f"Condition d'achat remplie pour {symbol}! Procéder à l'achat.")


# Fonction pour interroger ChatGPT
def chatgpt_query(prompt):
    url = 'https://api.openai.com/v1/engines/davinci-codex/completions'
    headers = {"Authorization": f"Bearer {chatgpt_api_key}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": 100}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    else:
        logging.error(f"ChatGPT query failed with status code {response.status_code}")
        return ""

# Lecture des tokens depuis le fichier usdt.txt
with open('usdt.txt', 'r') as file:
    tokens = [line.strip() for line in file if line.strip()]

# Analyse des opportunités pour chaque token
opportunites = []
for token in tokens:
    opportunite = analyze_opportunities(token)  # Votre fonction d'analyse des opportunités
    opportunites.append(opportunite)

# Préparation des descriptions pour ChatGPT
description_opportunites = ""
for opp in opportunites:
    description = f"Token: {opp['symbol']}, Action: {opp['action']}, Prix: {opp['price']}, Volume: {opp['volume']}"
    description_opportunites += description + "\\n"

# Envoi des opportunités à ChatGPT pour décision
chatgpt_prompt = f"Quelle est la meilleure opportunité de trading parmi ces options ?\\n{description_opportunites}"
chatgpt_decision = chatgpt_query(chatgpt_prompt)

# Logique pour agir sur la décision de ChatGPT
for opp in opportunites:
    if opp['symbol'] in chatgpt_decision:
        print(f"Décision d'investir dans {opp['symbol']} basée sur la décision de ChatGPT.")

        # Exécution des ordres d'achat
        try:
            montant_investissement = 1000  # exemple de montant d'investissement
            prix_actuel = opp['price']
            quantite = montant_investissement / prix_actuel

            # Exécutez l'ordre d'achat
            order = client.create_order(
                symbol=opp['symbol'],
                type='market',
                side='buy',
                quantity=quantite
            )
            print(f"Ordre d'achat exécuté : {order}")

            # Logique pour la vente
            prix_vente = prix_actuel * 1.05
            order_vente = client.create_order(
                symbol=opp['symbol'],
                type='limit',
                side='sell',
                price=prix_vente,
                quantity=quantite
            )
            print(f"Ordre de vente placé à : {prix_vente}")

        except Exception as e:
            print(f"Erreur lors de la transaction : {e}")
        break
else:
    print("Aucune action prise basée sur la décision de ChatGPT.")

# Fonction pour sauvegarder l'état
def save_state():
    state = {
        'symboles_achetes': list(symboles_achetes),
        # Ajoutez d'autres éléments d'état au besoin
    }

    with open('state.json', 'w') as state_file:
        json.dump(state, state_file)

# Fonction pour restaurer l'état
def restore_state():
    global symboles_achetes

    try:
        with open('state.json', 'r') as state_file:
            state = json.load(state_file)
            symboles_achetes = set(state.get('symboles_achetes', []))
            # Restaurez d'autres éléments d'état au besoin

    except FileNotFoundError:
        # Aucun fichier d'état précédent trouvé
        pass

# Fonction pour gérer les erreurs de manière centralisée
def handle_error(error_message):
    logging.error(error_message)
    # Ajoutez ici des actions supplémentaires en cas d'erreur

# Chargement de l'état précédent
restore_state()

# Boucle principale
try:
    while True:
        print("----------------- NOUVELLE ITÉRATION -----------------")

        # Récupération du solde et affichage
        balance = exchange.fetch_balance()
        usdt_balance = balance["USDT"]["free"]
        print("********************************   SOLDE EN USDT   :******************************** ", usdt_balance, "$")

        # Calcul de la capacité à investir
        investment_amount = int(usdt_balance)
        print("********************************   CAPITAL   :********************************", investment_amount, "$")

        # Analyse des actifs négociables
        exchange.load_markets()

        # Récupération des paires de trading depuis le fichier de configuration
        with open('usdt.txt', 'r') as config_file:
            trading_pairs = config_file.read().strip().split('\n')

        # Récupération des informations ticker pour les paires de trading
        ticker_info = [exchange.fetch_ticker(pair) for pair in trading_pairs]

        df = pd.DataFrame(ticker_info)
        df['percentage'] = ((df['last'] - df['open']) / df['open']) * 100
        df = df.sort_values(by='percentage', ascending=False)

        print("********************************    TOP 5      ********************************:")
        print(df.head(5))

        df['symbol'] = df['symbol'].apply(lambda x: x.replace("/USDT", "USDT").replace(":USDT", ""))

        # Utilisation de ThreadPoolExecutor pour paralléliser le traitement des symboles
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            symbol_chunks = np.array_split(df['symbol'].tolist(), MAX_WORKERS)
            executor.map(process_symbol_chunk, symbol_chunks, [investment_amount] * len(symbol_chunks), [df] * len(symbol_chunks))


        # Sauvegarde de l'état après chaque itération
        save_state()

        # Attendre un certain temps avant de relancer la boucle
        print("\n******************************** ATTENTE AVANT PROCHAINE ITÉRATION ********************************\n")
        time.sleep(3)  # 3 secondes
except KeyboardInterrupt:
    print("Arrêt de la boucle principale.")
