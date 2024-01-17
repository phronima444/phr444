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
