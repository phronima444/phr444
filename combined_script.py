import sys
import time
import pandas as pd
import ccxt
import talib as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import tensorflow as tf
import pickle
import configparser
import shutil  # Nouvelle importation

print("TensorFlow version:", tf.__version__)

# Forcer l'utilisation du CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Paramètres configurables
sequence_length = 10
num_filters = 64
kernel_size = 3
lstm_units = 50
# Modifiez dense_units en fonction du nombre de colonnes dans vos données d'entraînement
dense_units = 5
learning_rate = 0.001
epochs = 10
batch_size = 32
validation_split = 0.2

# Lecture des clés API depuis le fichier config.ini
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('API', 'api_key')
api_secret = config.get('API', 'api_secret')

# Fonction pour créer le modèle CNN-LSTM
def create_cnn_lstm_model(input_shape, num_filters, kernel_size, lstm_units, dense_units, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(LSTM(lstm_units, activation='relu', return_sequences=True))
    model.add(LSTM(lstm_units, activation='relu'))
    model.add(Dense(dense_units, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Fonction pour prétraiter les données
def preprocess_data(df, scaler=None, sequence_length=sequence_length):
    # Normalisation des données
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
    else:
        scaled_data = scaler.transform(df.values)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length, :-1])
        y.append(scaled_data[i+sequence_length, -1])

    X, y = np.array(X), np.array(y)

    return X, y, scaler

# Fonction pour extraire les données historiques via l'API Binance
def fetch_ohlcv_data(symbol, timeframe='1d', limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
    ohlcv_df.set_index('timestamp', inplace=True)
    return ohlcv_df

# Fonction pour effectuer des prédictions avec le modèle entraîné
def make_predictions(model, X, scaler):
    predicted_values = model.predict(X)
    predicted_values = scaler.inverse_transform(predicted_values)
    return predicted_values

# Création de l'instance d'échange Binance
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

# Récupération des symboles échangeables en USDT
markets = exchange.load_markets()
usdt_symbols = [symbol for symbol in markets.keys() if 'USDT' in symbol]

# Création d'un répertoire pour sauvegarder les poids des modèles
weights_directory = 'modele_poids'
if not os.path.exists(weights_directory):
    os.makedirs(weights_directory)

# Liste pour stocker les prédictions
predictions_list = []

# Boucle sur tous les symboles
for symbol_to_analyze in usdt_symbols:
    print(f"\nAnalyse du symbole : {symbol_to_analyze}")

    # Récupération des données historiques
    historical_data = fetch_ohlcv_data(symbol_to_analyze)

    # Charger les données d'entraînement de l'itération précédente si elles existent
    previous_data_file = f'modele_poids/{symbol_to_analyze.replace("/", "_")}_donnees_precedentes.pkl'
    restore_successful = False

    if os.path.exists(previous_data_file):
        try:
            with open(previous_data_file, 'rb') as file:
                previous_data = pickle.load(file)
            previous_X, previous_y, previous_scaler = previous_data['X'], previous_data['y'], previous_data['scaler']

            # Vérifier que les dimensions correspondent
            if previous_X.shape[1] == historical_data.shape[1]:
                # Concaténer les données historiques avec les données de l'itération précédente
                historical_data = pd.concat([pd.DataFrame(previous_X[-sequence_length:], columns=historical_data.columns), historical_data])
                restore_successful = True
        except Exception as e:
            print(f"Échec de la restauration des données pour le symbole {symbol_to_analyze}. Erreur : {e}")

    # Si la restauration échoue, réinitialiser les données
    if not restore_successful:
        print(f"Restauration échouée pour le symbole {symbol_to_analyze}. Réinitialisation des données.")
        previous_X, previous_y, previous_scaler = None, None, None

        # Supprimer le fichier de données précédentes
        if os.path.exists(previous_data_file):
            os.remove(previous_data_file)

    # Prétraitement des données
    X, y, scaler = preprocess_data(historical_data, scaler=previous_scaler, sequence_length=sequence_length)

    # Vérification de la forme de X
    if X.shape[1] == 0 or X.shape[2] == 0:
        print(f"Données insuffisantes pour le symbole {symbol_to_analyze}. Passez à la prochaine itération.")
        continue

    # Création du modèle
    input_shape = (X.shape[1], X.shape[2])

    # Vérification des dimensions de X
    if input_shape[0] == 0 or input_shape[1] == 0:
        print(f"Données insuffisantes pour le symbole {symbol_to_analyze}. Passez à la prochaine itération.")
        continue

    model = create_cnn_lstm_model(input_shape, num_filters, kernel_size, lstm_units, dense_units, learning_rate)

    # Callback pour sauvegarder les poids du modèle
    symbol_weights_file = os.path.join(weights_directory, f"modele_poids_{symbol_to_analyze.replace('/', '_')}.h5")
    checkpoint_callback = ModelCheckpoint(filepath=symbol_weights_file, save_best_only=True, save_weights_only=True)

    # Callback pour arrêter l'entraînement prématurément si la performance sur l'ensemble de validation ne s'améliore pas
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    # Entraînement du modèle
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[checkpoint_callback, early_stopping_callback])

    # Sauvegarde des données d'entraînement pour l'itération suivante
    current_data = {'X': X, 'y': y, 'scaler': scaler}
    with open(previous_data_file, 'wb') as file:
        pickle.dump(current_data, file)

    # Faire des prédictions avec le modèle entraîné
    predictions = make_predictions(model, X, scaler)
    print("Prédictions:")
    print(predictions)

    # Stocker les prédictions dans la liste
    predictions_list.append({'symbol': symbol_to_analyze, 'predictions': predictions})

# Enregistrement des prédictions
predictions_file = 'predictions.pkl'
with open(predictions_file, 'wb') as file:
    pickle.dump(predictions_list, file)
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


            # Ajoutez ici votre code pour la procédure d'achat
            # Par exemple :
            try:
                order = client.order_market_buy(
                    symbol=symbol,
                    quantity=int(investment_amount / ohlcv_df['close'].iloc[-1])
                )
                print("++++++++++++++++Ordre d'achat effectué:+++++++++++++++++++++", order)

                # Ajouter le symbole à la liste des symboles achetés
                symboles_achetes.add(symbol)

                # Récupérer le prix d'achat et la quantité de l'ordre
                buy_price = float(order['fills'][0]['price'])
                buy_quantity = float(order['fills'][0]['qty'])
                print(f"Prix d'achat pour {symbol}: {buy_price}")
                print(f"Quantité achetée pour {symbol}: {buy_quantity}")

                # Fetch the last executed order
                orders = exchange.fetch_orders(symbol=symbol, limit=1)

                # Vérification des symboles à vendre
                symboles_a_vendre = symboles_achetes.intersection(set(df['symbol']))

                # Get the initial buy price
                initial_buy_price = float(orders[0]['price'])
                print('-------------------------------------------')
                print('Prix achat:', symbol, initial_buy_price)

                # Place a sell limit order of 3% below the buy price
                market_price = exchange.fetch_ticker(symbol)['last']
                print('-------------------------------------------')
                print('prix du marché:', market_price)

                if market_price > initial_buy_price * 1.03:
                    sell_limit_price = market_price
                else:
                    sell_limit_price = initial_buy_price * 1.03

                volume_de_vente = buy_quantity / 1.001

                print('-------------------------------------------')
                print('++++++++++++++Quantité achetée:+++++++++++++++', buy_quantity)
                print('-------------------------------------------')
                print('++++++++++++++Quantité à vendre:++++++++++++++', volume_de_vente)
                print('-------------------------------------------')

                sell_order = exchange.create_order(
                    symbol=symbol,
                    price=sell_limit_price,
                    type='limit',
                    side='SELL',
                    amount=volume_de_vente
                )

                print('-------------------------------------------')
                print('+++++++++++++++++++++++++Ordre de vente:++++++++++++++++++++++++++++', sell_order)
            except Exception as e:
                print(f"Erreur lors de l'achat ou de la vente : {e}")

    except Exception as e:
        print(f"ECHEC AVEC CE TOKEN: {symbol} {e} ESSAI AUTRE TOKEN")


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

# Fonction d'analyse technique
def analyze_technicals(ohlcv_df):
    close_prices = ohlcv_df['close'].values
    high_prices = ohlcv_df['high'].values
    low_prices = ohlcv_df['low'].values

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

# Modification de la fonction process_symbol pour inclure l'analyse technique
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
            # Le reste de la logique d'achat reste inchangé
            # ...
    except Exception as e:
        print(f"Erreur lors de l'analyse du token {symbol} : {e}")

# Nouvelle fonction pour sélectionner le token à acheter
def select_token_to_buy(symbols, exchange, time_frame, investment_amount, df):
    selected_token = None
    best_change_percentage = -float('inf')

    for symbol in symbols:
        try:
            # Récupération des données OHLCV
            ohlcv = exchange.fetch_ohlcv(symbol, time_frame, limit=100)

            # Création d'un DataFrame avec les données historiques
            ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Analyse de la variation sur 5 secondes
            ohlcv_df['price_change_5s'] = (ohlcv_df['close'].diff() / ohlcv_df['close'].shift(1)) * 100

            # Vérification des conditions d'analyse technique
            if analyze_technicals(ohlcv_df) and ohlcv_df['price_change_5s'].iloc[-2] > CONDITION_ACHAT_PERCENTAGE:
                # Choix du token avec la plus grande variation positive sur 5 secondes
                if ohlcv_df['price_change_5s'].iloc[-2] > best_change_percentage:
                    best_change_percentage = ohlcv_df['price_change_5s'].iloc[-2]
                    selected_token = symbol

        except Exception as e:
            print(f"Erreur lors de l'analyse du token {symbol} : {e}")

    return selected_token
