import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Oldal konfiguráció
st.set_page_config(
    page_title="LSTM Idősor-előrejelző",
    page_icon="📈",
    layout="wide"
)

st.title("📈 LSTM Idősor-előrejelző Alkalmazás")
st.markdown("**Heti és havi előrejelzések készítése Excel adatokból**")

st.divider()

# Szekvencia generátor függvény
def make_sequences_multi(series_1d, window, horizon):
    X, y = [], []
    n = len(series_1d)
    for i in range(n - window - horizon + 1):
        X.append(series_1d[i:i + window])
        y.append(series_1d[i + window:i + window + horizon])
    return np.array(X), np.array(y)

# LSTM tanító és előrejelző függvény
def run_lstm_forecast(series, lookback, horizon, item_name, frequency):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = make_sequences_multi(scaled_series, lookback, horizon)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    n_samples = len(X)
    if n_samples < 10:
        st.warning(f"⚠️ Alacsony mintaszám ({n_samples}). Az eredmények instabilak lehetnek.")

    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    inputs = Input(shape=(lookback, 1))
    x = LSTM(32, activation="tanh")(inputs)
    outputs = Dense(horizon)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=800, batch_size=1, shuffle=False,
        verbose=0, validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    end_time = time.time()
    train_time = end_time - start_time

    y_pred_scaled = model.predict(X_test, verbose=0)

    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_true, y_pred)
    r2 = r2_score(y_test_true, y_pred)

    n_test = y_test.shape[0]
    y_test_mat = y_test_true.reshape(n_test, horizon)
    y_pred_mat = y_pred.reshape(n_test, horizon)
    error_mat = y_pred_mat - y_test_mat

    last_window = scaled_series[-lookback:].reshape(1, lookback, 1)
    future_scaled = model.predict(last_window, verbose=0)
    future = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

    return {
        'history': history,
        'y_pred': y_pred,
        'y_test_true': y_test_true,
        'future': future,
        'mse': mse,
        'r2': r2,
        'train_time': round(train_time, 2),
        'error_mat': error_mat,
        'series': series
    }

# Fájl feltöltő
uploaded_file = st.file_uploader(
    "📂 Tölts fel egy Excel fájlt (BaseDatasLSTM)",
    type=['xlsx', 'xls'],
    help="Az Excel fájl első oszlopában az 'Item' neveknek kell lenniük, majd az oszlopok a heti adatokat tartalmazzák."
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        st.success("✔ Excel fájl sikeresen betöltve!")
        
        with st.expander("📊 Nyers adatok előnézete"):
            st.dataframe(df_raw.head(10))
        
        df_weekly = df_raw.set_index("Item")
        df_weekly = df_weekly.apply(pd.to_numeric, errors='coerce').fillna(0)
        df_weekly = df_weekly.T
        n_weeks = df_weekly.shape[0]
        
        weeks_per_month = 4
        n_months = n_weeks // weeks_per_month
        
        df_monthly = pd.DataFrame({
            col: [df_weekly[col].iloc[i*weeks_per_month:(i+1)*weeks_per_month].sum()
                  for i in range(n_months)]
            for col in df_weekly.columns
        })
        
        st.info(f"📅 Adatok: **{n_weeks} hét** és **{n_months} hónap** ({len(df_weekly.columns)} termék)")
        
        lookback_weeks = 12
        forecast_horizon_weeks = 12
        lookback_months = 6
        forecast_horizon_months = 6
        
        st.divider()
        st.subheader("🧠 LSTM Modell Tanítása és Előrejelzések")
        
        if st.button("🚀 Elemzés Indítása", type="primary"):
            training_stats = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_items = len(df_weekly.columns)
            
            for idx, item in enumerate(df_weekly.columns):
                status_text.text(f"⏳ Feldolgozás: {item} ({idx + 1}/{total_items})")
                
                weekly_series = df_weekly[item].values
                monthly_series = df_monthly[item].values
                
                weekly_results = run_lstm_forecast(
                    weekly_series, lookback_weeks, forecast_horizon_weeks, item, "Heti"
                )
                
                monthly_results = run_lstm_forecast(
                    monthly_series, lookback_months, forecast_horizon_months, item, "Havi"
                )
                
                training_stats.append({
                    "Termék": item,
                    "Heti MSE": round(weekly_results['mse'], 2),
                    "Havi MSE": round(monthly_results['mse'], 2),
                    "Heti R²": round(weekly_results['r2'], 2),
                    "Havi R²": round(monthly_results['r2'], 2),
                    "Heti tanítási idő (s)": weekly_results['train_time'],
                    "Havi tanítási idő (s)": monthly_results['train_time']
                })
                
                st.markdown(f"### 📌 {item}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📅 Heti Elemzés")
                    
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.plot(weekly_results['history'].history['loss'], label='Tanítási veszteség')
                    ax1.plot(weekly_results['history'].history['val_loss'], label='Validációs veszteség')
                    ax1.set_title("Veszteség Görbe (MSE)")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("MSE")
                    ax1.legend()
                    ax1.grid(True)
                    st.pyplot(fig1)
                    plt.close()
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    im = ax2.imshow(weekly_results['error_mat'], aspect='auto', cmap='coolwarm')
                    plt.colorbar(im, ax=ax2, label='Hiba (előrejelzés - valós)')
                    ax2.set_xlabel("Előrejelzési lépés")
                    ax2.set_ylabel("Teszt szekvencia index")
                    ax2.set_title("Előrejelzési Hiba Hőtérkép")
                    st.pyplot(fig2)
                    plt.close()
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    ax3.plot(weekly_results['series'], label="Heti adatok", color="blue", marker="x")
                    ax3.plot(
                        range(len(weekly_results['series']) - 1, len(weekly_results['series']) + forecast_horizon_weeks),
                        np.concatenate([[weekly_results['series'][-1]], weekly_results['future']]),
                        label="Heti előrejelzés",
                        color="red",
                        marker="o"
                    )
                    ax3.set_title(f"{item} – Heti Adatok és Előrejelzés")
                    ax3.grid(True)
                    ax3.legend()
                    st.pyplot(fig3)
                    plt.close()
                
                with col2:
                    st.markdown("#### 📆 Havi Elemzés")
                    
                    fig4, ax4 = plt.subplots(figsize=(10, 5))
                    ax4.plot(monthly_results['history'].history['loss'], label='Tanítási veszteség')
                    ax4.plot(monthly_results['history'].history['val_loss'], label='Validációs veszteség')
                    ax4.set_title("Veszteség Görbe (MSE)")
                    ax4.set_xlabel("Epoch")
                    ax4.set_ylabel("MSE")
                    ax4.legend()
                    ax4.grid(True)
                    st.pyplot(fig4)
                    plt.close()
                    
                    fig5, ax5 = plt.subplots(figsize=(10, 5))
                    im = ax5.imshow(monthly_results['error_mat'], aspect='auto', cmap='coolwarm')
                    plt.colorbar(im, ax=ax5, label='Hiba (előrejelzés - valós)')
                    ax5.set_xlabel("Előrejelzési lépés")
                    ax5.set_ylabel("Teszt szekvencia index")
                    ax5.set_title("Előrejelzési Hiba Hőtérkép")
                    st.pyplot(fig5)
                    plt.close()
                    
                    fig6, ax6 = plt.subplots(figsize=(10, 5))
                    ax6.plot(monthly_results['series'], label="Havi adatok", color="black", marker="x")
                    ax6.plot(
                        range(len(monthly_results['series']) - 1, len(monthly_results['series']) + forecast_horizon_months),
                        np.concatenate([[monthly_results['series'][-1]], monthly_results['future']]),
                        label="Havi előrejelzés",
                        color="orange",
                        marker="o"
                    )
                    ax6.set_title(f"{item} – Havi Adatok és Előrejelzés")
                    ax6.grid(True)
                    ax6.legend()
                    st.pyplot(fig6)
                    plt.close()
                
                st.divider()
                progress_bar.progress((idx + 1) / total_items)
            
            status_text.text("✅ Elemzés kész!")
            st.success("🎉 Minden termék feldolgozva!")
            
            st.subheader("📊 Összefoglaló Táblázat")
            stats_df = pd.DataFrame(training_stats)
            st.dataframe(stats_df, use_container_width=True)
            
            csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Eredmények letöltése CSV-ként",
                data=csv,
                file_name='lstm_eredmenyek.csv',
                mime='text/csv',
            )
    
    except Exception as e:
        st.error(f"❌ Hiba az Excel fájl feldolgozása során: {e}")
else:
    st.info("👆 Kérlek, tölts fel egy Excel fájlt az elemzés megkezdéséhez!")