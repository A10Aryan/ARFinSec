import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import pytz
import upstox_client
from upstox_client.rest import ApiException
import logging
import asyncio
import uuid

# ========== CONFIGURATION ========== #
st.set_page_config(page_title="NIFTY & BANKNIFTY Futures Dashboard", layout="wide")
st.title("📊 NIFTY & BANKNIFTY Futures Dashboard")
st.markdown("Real-time analysis of NIFTY and BANKNIFTY futures contracts with advanced analytics.")

# Apply dark theme via CSS for compatibility with older Streamlit versions
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stApp {
            background-color: #1E1E1E;
        }
        .stTextInput > div > div > input {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        .stSelectbox > div > div > select {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        .stSlider > div > div > div > div {
            background-color: #4CAF50;
        }
        .stMetric {
            background-color: #2E2E2E;
            padding: 10px;
            border-radius: 5px;
        }
        .stDataFrame {
            background-color: #2E2E2E;
        }
    </style>
""", unsafe_allow_html=True)

# Setup logging
logging.basicConfig(filename="dashboard.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ========== UPSTOX CLIENT SETUP ========== #
def setup_upstox_client():
    try:
        # Replace with your Upstox access token
        access_token = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIySkNaM1ciLCJqdGkiOiI2OGE1ZDU1N2M1MDFlYjUyMzRhNDI2YTkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc1NTY5ODUxOSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzU1NzI3MjAwfQ.t1utIsq_TLB9O-pAdqYqSLafHbC7lZ3AqT78mMH1iAA"
        if not access_token or access_token == "your_upstox_access_token_here":
            raise ValueError("Upstox access token is missing or invalid. Please provide a valid token.")
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        api_client = upstox_client.ApiClient(configuration)
        logging.info("Upstox client initialized successfully.")
        return api_client
    except Exception as e:
        st.error(f"Failed to initialize Upstox client: {str(e)}. Please provide a valid access token.")
        logging.error(f"Upstox client setup failed: {str(e)}")
        return None

# ========== DATA LOADING ========== #
@st.cache_data(ttl=3600)
def load_instruments(_api_client):
    try:
        api_instance = upstox_client.MarketQuoteApi(_api_client)
        # Placeholder for actual API call
        # instruments = api_instance.get_instruments(exchange='NSE')
        
        # Expanded mock data with NIFTY and BANKNIFTY futures
        instruments = [
            {
                'instrument_key': 'NSE_INDEX|NIFTY 50',
                'tradingsymbol': 'NIFTY 50',
                'name': 'NIFTY 50',
                'expiry': '',
                'lot_size': 1,
                'instrument_type': 'INDEX'
            },
            {
                'instrument_key': 'NSE_INDEX|BANKNIFTY',
                'tradingsymbol': 'BANKNIFTY',
                'name': 'BANKNIFTY',
                'expiry': '',
                'lot_size': 1,
                'instrument_type': 'INDEX'
            },
            {
                'instrument_key': 'NSE_FO|NIFTY25JUNFUT',
                'tradingsymbol': 'NIFTY25JUNFUT',
                'name': 'NIFTY FUT',
                'expiry': '2025-06-26',
                'lot_size': 50,
                'instrument_type': 'FUT'
            },
            {
                'instrument_key': 'NSE_FO|NIFTY25JULFUT',
                'tradingsymbol': 'NIFTY25JULFUT',
                'name': 'NIFTY FUT',
                'expiry': '2025-07-31',
                'lot_size': 50,
                'instrument_type': 'FUT'
            },
            {
                'instrument_key': 'NSE_FO|NIFTY25AUGFUT',
                'tradingsymbol': 'NIFTY25AUGFUT',
                'name': 'NIFTY FUT',
                'expiry': '2025-08-28',
                'lot_size': 50,
                'instrument_type': 'FUT'
            },
            {
                'instrument_key': 'NSE_FO|BANKNIFTY25JUNFUT',
                'tradingsymbol': 'BANKNIFTY25JUNFUT',
                'name': 'BANKNIFTY FUT',
                'expiry': '2025-06-26',
                'lot_size': 15,
                'instrument_type': 'FUT'
            },
            {
                'instrument_key': 'NSE_FO|BANKNIFTY25SEPFUT',
                'tradingsymbol': 'BANKNIFTY25SEPFUT',
                'name': 'BANKNIFTY FUT',
                'expiry': '2025-09-25',
                'lot_size': 15,
                'instrument_type': 'FUT'
            },
        ]
        
        df = pd.DataFrame(instruments)
        str_cols = ['instrument_key', 'tradingsymbol', 'name', 'expiry']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        logging.info("Instruments loaded successfully.")
        return df
    except ApiException as e:
        st.error(f"Upstox API exception: {str(e)}")
        logging.error(f"Upstox API exception in load_instruments: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        logging.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# ========== PRICE FETCHING ========== #
def fetch_prices(_api_client, instrument_key):
    try:
        api_instance = upstox_client.MarketQuoteApi(_api_client)
        # Placeholder for actual API call
        # quote = api_instance.get_full_market_quote(instrument_key)
        # return quote['data']
        
        # Mock data adjusted for NIFTY and BANKNIFTY
        if 'NIFTY' in instrument_key:
            price_range = (23000, 24000)
        else:  # BANKNIFTY
            price_range = (48000, 50000)
        mock_price = {
            'last_price': np.random.uniform(*price_range),
            'bid': np.random.uniform(price_range[0] - 50, price_range[0]),
            'ask': np.random.uniform(price_range[0], price_range[1] + 50),
            'volume': np.random.randint(1000, 10000),
            'oi': np.random.randint(5000, 20000),
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        }
        return mock_price
    except ApiException as e:
        st.error(f"Upstox API exception while fetching prices: {str(e)}")
        logging.error(f"Upstox API price fetch failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching prices: {str(e)}")
        logging.error(f"Price fetch error: {str(e)}")
        return None

async def fetch_price_async(api_client, instrument_key):
    return await asyncio.to_thread(fetch_prices, api_client, instrument_key)

# ========== CALCULATION FUNCTIONS ========== #
def calculate_pcr(options_df):
    """Calculate Put-Call Ratio for a specific expiry"""
    if not options_df.empty and 'oi' in options_df.columns:
        puts = options_df[options_df['tradingsymbol'].str.contains('PE', case=False, na=False)]['oi'].sum()
        calls = options_df[options_df['tradingsymbol'].str.contains('CE', case=False, na=False)]['oi'].sum()
        return puts / calls if calls != 0 else 0
    return 0

def calculate_sma(series, length=20):
    """Calculate Simple Moving Average"""
    return series.rolling(window=length).mean()

def calculate_rsi(series, length=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(price_history):
    """Add technical indicators (SMA, RSI)"""
    price_history['SMA20'] = calculate_sma(price_history['Spot'], length=20)
    price_history['RSI'] = calculate_rsi(price_history['Spot'], length=14)
    return price_history

# ========== MAIN DASHBOARD ========== #
def main():
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### Dashboard Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
        show_analytics = st.checkbox("Show Advanced Analytics", value=False)
        st.markdown("### Help")
        st.write("1. Select an asset and futures contract.\n2. View real-time prices, PCR, and trends.\n3. Use settings to customize refresh rate and analytics.")

    # Initialize Upstox client
    api_client = setup_upstox_client()
    if not api_client:
        return

    # Load instruments
    instruments_df = load_instruments(api_client)
    if instruments_df.empty:
        st.error("No instrument data available. Please check your API connection.")
        return

    # Asset selection
    available_assets = instruments_df['name'].str.extract(r'(NIFTY|BANKNIFTY)').dropna()[0].unique()
    selected_asset = st.selectbox("Select Asset", available_assets, index=0)

    # Find spot and futures instruments
    nifty_spot = instruments_df[
        (instruments_df['tradingsymbol'].str.contains(selected_asset, case=False)) &
        (instruments_df['instrument_key'].str.contains('NSE_INDEX', case=False))
    ].iloc[0] if not instruments_df.empty else None

    nifty_futures = instruments_df[
        (instruments_df['tradingsymbol'].str.contains(f'{selected_asset}.*FUT', case=False, regex=True)) &
        (instruments_df['instrument_key'].str.contains('NSE_FO', case=False))
    ].copy()

    if nifty_futures.empty:
        st.warning(f"No {selected_asset} futures contracts found.")
        return

    # Convert and sort futures
    try:
        nifty_futures['expiry'] = pd.to_datetime(nifty_futures['expiry'])
    except Exception as e:
        st.error(f"Failed to parse expiry dates: {str(e)}. Please check the instrument data.")
        logging.error(f"Expiry date parsing failed: {str(e)}")
        return
    nifty_futures = nifty_futures.sort_values('expiry')
    nifty_futures['expiry_display'] = nifty_futures['expiry'].dt.strftime('%d-%b-%Y')

    # ===== FUTURES TABLE VIEW ===== #
    with st.expander("Futures Contracts", expanded=True):
        st.subheader(f"All {selected_asset} Futures Contracts")
        search_term = st.text_input("Search futures contracts", key="search_futures")
        filtered_futures = nifty_futures[
            nifty_futures['tradingsymbol'].str.contains(search_term, case=False, na=False)
        ] if search_term else nifty_futures

        if len(filtered_futures) == 0:
            st.warning("No futures contracts match the search criteria.")
            return
        elif len(filtered_futures) == 1:
            st.info("Only one futures contract available.")
            num_rows = 1
        else:
            num_rows = st.slider(
                "Rows to show", 
                min_value=1, 
                max_value=len(filtered_futures), 
                value=min(30, len(filtered_futures)),
                key=f"rows_{uuid.uuid4()}"
            )

        display_columns = ['tradingsymbol', 'expiry_display', 'lot_size']
        st.dataframe(
            filtered_futures[display_columns].head(num_rows),
            height=400,
            column_config={
                "tradingsymbol": "Contract",
                "expiry_display": "Expiry",
                "lot_size": "Lot Size"
            }
        )

    # ===== DRILL-DOWN SELECTION ===== #
    with st.expander("Contract Analysis", expanded=True):
        st.subheader("Analyze Specific Contract")
        selected_contract = st.selectbox(
            "Choose a futures contract",
            options=nifty_futures['tradingsymbol'].unique(),
            key=f"contract_{uuid.uuid4()}"
        )

        if selected_contract:
            selected_future = nifty_futures[
                nifty_futures['tradingsymbol'] == selected_contract
            ].iloc[0]

            # Fetch prices asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            spot_data, fut_data = loop.run_until_complete(asyncio.gather(
                fetch_price_async(api_client, nifty_spot['instrument_key']) if nifty_spot is not None else asyncio.sleep(0, result=None),
                fetch_price_async(api_client, selected_future['instrument_key'])
            ))

            if not spot_data or not fut_data:
                st.error("Failed to fetch price data. Check your API connection or try again.")
                logging.warning(f"Price fetch failed for {selected_contract}")
                return

            # Calculate premium/discount
            premium = fut_data['last_price'] - spot_data['last_price']
            premium_pct = (premium / spot_data['last_price']) * 100

            # ===== PRICE COMPARISON ===== #
            st.subheader("Price Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{selected_asset} Spot", 
                         f"{spot_data['last_price']:,.2f}",
                         help=f"Updated: {spot_data['timestamp']}")
            with col2:
                st.metric(f"Selected Future\n({selected_future['expiry_display']})", 
                         f"{fut_data['last_price']:,.2f}",
                         help=f"Updated: {fut_data['timestamp']}")
            with col3:
                delta_color = "inverse" if premium < 0 else "normal"
                st.metric("Premium/Discount", 
                         f"{premium:,.2f}",
                         f"{premium_pct:.2f}%",
                         delta_color=delta_color)
            with col4:
                st.metric("Futures Volume", f"{fut_data['volume']:,}")

            # ===== PCR ANALYSIS ===== #
            if st.checkbox("Show Put-Call Ratio Analysis", value=True):
                st.subheader("Put-Call Ratio Analysis")
                # Dynamic mock options data based on selected asset and expiry
                strike_price = 23500 if 'NIFTY' in selected_asset else 48500
                expiry_date = selected_future['expiry'].strftime('%Y-%m-%d')
                # Generate random OI to vary PCR (range adjusted for realism)
                call_oi = np.random.randint(30000, 70000)
                put_oi = np.random.randint(30000, 90000)
                # Adjust OI based on expiry (nearer expiries have higher OI)
                expiry_datetime = pd.to_datetime(expiry_date).tz_localize('Asia/Kolkata')
                current_datetime = datetime.now(pytz.timezone('Asia/Kolkata'))
                days_to_expiry = (expiry_datetime - current_datetime).days
                oi_multiplier = 1.5 if days_to_expiry < 60 else 1.0 if days_to_expiry < 120 else 0.7
                call_oi = int(call_oi * oi_multiplier)
                put_oi = int(put_oi * oi_multiplier)
                # Adjust OI slightly for BANKNIFTY (typically higher volatility)
                if 'BANKNIFTY' in selected_asset:
                    call_oi = int(call_oi * 1.2)
                    put_oi = int(put_oi * 1.3)
                
                nifty_options = pd.DataFrame([
                    {'tradingsymbol': f'{selected_asset}{expiry_date[-5:-3]}{strike_price}CE', 'oi': call_oi, 'expiry': expiry_date},
                    {'tradingsymbol': f'{selected_asset}{expiry_date[-5:-3]}{strike_price}PE', 'oi': put_oi, 'expiry': expiry_date},
                ])
                
                pcr = calculate_pcr(nifty_options)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Put-Call Ratio", f"{pcr:.2f}", help="Ratio of put to call open interest")
                    st.write("**Market Sentiment:**")
                    if pcr > 1.2:
                        st.error("Strong Bearish (PCR > 1.2)")
                    elif pcr > 0.8:
                        st.warning("Mild Bearish (0.8 < PCR ≤ 1.2)")
                    elif pcr > 0.6:
                        st.success("Mild Bullish (0.6 < PCR ≤ 0.8)")
                    else:
                        st.error("Strong Bullish (PCR ≤ 0.6)")
                with col2:
                    dates = pd.date_range(end=datetime.now(pytz.timezone('Asia/Kolkata')), periods=30)
                    pcr_history = pd.DataFrame({
                        'Date': dates,
                        'PCR': np.random.uniform(0.5, 1.5, 30) * (pcr / 1.0)  # Scale PCR trend around current PCR
                    })
                    fig = px.line(pcr_history, x='Date', y='PCR', 
                                 title=f'PCR Trend for {selected_future["expiry_display"]} Expiry',
                                 template="plotly_dark")
                    fig.add_hline(y=1, line_dash="dash", line_color="red")
                    fig.update_layout(hovermode="x unified", dragmode="zoom")
                    st.plotly_chart(fig, use_container_width=True)

            # ===== PRICE TREND VISUALIZATION ===== #
            st.subheader("Price Relationship")
            days = 30
            dates = pd.date_range(end=datetime.now(pytz.timezone('Asia/Kolkata')), periods=days)
            spot_prices = spot_data['last_price'] * (1 + np.random.normal(0, 0.005, days).cumsum())
            fut_prices = fut_data['last_price'] * (1 + np.random.normal(0, 0.005, days).cumsum())
            price_history = pd.DataFrame({
                'Date': dates,
                'Spot': spot_prices,
                'Futures': fut_prices,
                'Premium': fut_prices - spot_prices
            })

            if show_analytics:
                price_history = calculate_technical_indicators(price_history)
                fig1 = px.line(price_history.melt(id_vars='Date', value_vars=['Spot', 'Futures', 'SMA20', 'RSI']), 
                              x='Date', y='value', color='variable',
                              title=f'Spot vs {selected_contract} Price Trend with Indicators',
                              template="plotly_dark")
            else:
                fig1 = px.line(price_history.melt(id_vars='Date', value_vars=['Spot', 'Futures']), 
                              x='Date', y='value', color='variable',
                              title=f'Spot vs {selected_contract} Price Trend',
                              template="plotly_dark")
            fig1.update_layout(hovermode="x unified", dragmode="zoom")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.area(price_history, x='Date', y='Premium',
                          title='Futures Premium/Discount Trend',
                          template="plotly_dark")
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            fig2.update_layout(hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
