import asyncio
import datetime
import logging
import os
import aiohttp
import asyncpg
import pandas as pd
import yfinance as yf
import ta
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TechnicalAgent")


class TechnicalAgent:
    def __init__(self):
        self.massive_api_key  = os.getenv("MASSIVE_API_KEY")
        self.db_dsn           = os.getenv("DATABASE_URL")
        self.massive_base_url = "https://api.massive.com/v2"
        self.pool             = None

        # ── Swing cache (yfinance) ─────────────────────────────────────
        # Warm-up: başlangıçta 60 günlük tam veri çekilir → bellekte tutulur
        # Incremental: her 30 dakikada sadece son 2 günlük veri çekilir → eklenir
        # yfinance ücretsiz, limit yok, 30m için 60 gün geçmiş veri verir
        self._swing_cache: dict[str, pd.DataFrame] = {}
        self._swing_warmup_done: set[str]           = set()

    # ─────────────────────────────────────────────────────────────────
    # DB POOL
    # ─────────────────────────────────────────────────────────────────
    async def get_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.db_dsn,
                min_size=2,
                max_size=5,
                command_timeout=30
            )
        return self.pool

    # ─────────────────────────────────────────────────────────────────
    # VERİ ÇEKME — 0DTE (Massive API)
    # ─────────────────────────────────────────────────────────────────
    async def fetch_ohlcv_massive(self, symbol: str, multiplier: str, timespan: str) -> pd.DataFrame:
        """
        0DTE için Massive API — 5m mumlar, 5 günlük veri.
        EMA9 + EMA21 için 21 mum yeterli; 5 gün ~390 mum verir.
        """
        end_date   = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=5)

        _to   = end_date.strftime('%Y-%m-%d')
        _from = start_date.strftime('%Y-%m-%d')

        endpoint = (
            f"{self.massive_base_url}/aggs/ticker/{symbol}"
            f"/range/{multiplier}/{timespan}/{_from}/{_to}"
        )
        headers = {"Authorization": f"Bearer {self.massive_api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"[Massive] {symbol} yanıt kodu {response.status}")
                        return pd.DataFrame()
                    data = await response.json()
                    if 'results' in data and len(data['results']) > 0:
                        df = pd.DataFrame(data['results'])
                        df = df.rename(columns={
                            'v': 'volume', 'o': 'open', 'c': 'close',
                            'h': 'high',   'l': 'low',  't': 'timestamp'
                        })
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        logger.info(f"✅ [Massive] {symbol} {multiplier}{timespan} — {len(df)} mum")
                        return df
                    logger.warning(f"⚠️ [Massive] {symbol} sonuç boş.")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"[Massive] Veri çekme hatası ({symbol}): {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────
    # VERİ ÇEKME — SWING (yfinance)
    # ─────────────────────────────────────────────────────────────────
    def fetch_ohlcv_yfinance(self, symbol: str, period: str = "60d") -> pd.DataFrame:
        """
        yfinance — 30m mumlar, ücretsiz, limit yok.
        period="60d" → warm-up  (~780 mum, SMA200 için yeterli)
        period="2d"  → incremental (son 2 gün, overlap için +1 gün tampon)
        Not: yfinance sync çalışır, asyncio.to_thread ile sarılır.
        """
        try:
            ticker = yf.Ticker(symbol)
            df_raw = ticker.history(period=period, interval="30m")

            if df_raw.empty:
                logger.warning(f"[yfinance] {symbol} veri boş döndü.")
                return pd.DataFrame()

            df = df_raw.reset_index().rename(columns={
                "Datetime": "timestamp",
                "Open":     "open",
                "High":     "high",
                "Low":      "low",
                "Close":    "close",
                "Volume":   "volume",
            })[[ "timestamp", "open", "high", "low", "close", "volume"]]

            # Timezone bilgisini kaldır (asyncpg için)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.sort_values("timestamp").reset_index(drop=True)

            gun_sayisi = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days + 1
            gunluk_ort = round(len(df) / max(gun_sayisi, 1), 1)
            logger.info(
                f"✅ [yfinance] {symbol} 30m — {len(df)} mum | "
                f"{gun_sayisi} gün | ort {gunluk_ort} mum/gün"
            )
            return df
        except Exception as e:
            logger.error(f"[yfinance] Veri çekme hatası ({symbol}): {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────
    # SWING CACHE YÖNETİMİ
    # ─────────────────────────────────────────────────────────────────
    async def get_swing_df(self, symbol: str) -> pd.DataFrame:
        """
        İlk çağrıda  : 60 günlük tam veri → cache'e yaz (warm-up)
        Sonraki çağrı: son 2 günlük veri  → cache'e ekle (incremental)
        yfinance sync olduğu için asyncio.to_thread ile çalıştırılır.
        Duplicate mumları temizle, cache'i son 400 mumla sınırla.
        """
        if symbol not in self._swing_warmup_done:
            # ── WARM-UP ───────────────────────────────────────────────
            logger.info(f"🔄 [Cache] {symbol} warm-up başlıyor (60 gün)...")
            df = await asyncio.to_thread(self.fetch_ohlcv_yfinance, symbol, "60d")
            if df.empty:
                logger.error(f"❌ [Cache] {symbol} warm-up başarısız.")
                return pd.DataFrame()
            self._swing_cache[symbol] = df
            self._swing_warmup_done.add(symbol)
            logger.info(f"✅ [Cache] {symbol} warm-up tamamlandı — {len(df)} mum bellekte.")

        else:
            # ── INCREMENTAL ───────────────────────────────────────────
            yeni_df = await asyncio.to_thread(self.fetch_ohlcv_yfinance, symbol, "2d")
            if not yeni_df.empty:
                mevcut   = self._swing_cache.get(symbol, pd.DataFrame())
                combined = pd.concat([mevcut, yeni_df], ignore_index=True)

                # Duplicate timestamp temizle (son değeri koru)
                combined = (
                    combined
                    .drop_duplicates(subset="timestamp", keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                # Bellek tasarrufu: son 400 mum (SMA200 × 2 tampon)
                if len(combined) > 400:
                    combined = combined.tail(400).reset_index(drop=True)

                self._swing_cache[symbol] = combined
                logger.info(
                    f"🔁 [Cache] {symbol} güncellendi — "
                    f"toplam {len(combined)} mum (incremental +{len(yeni_df)})"
                )

        return self._swing_cache.get(symbol, pd.DataFrame())

    # ─────────────────────────────────────────────────────────────────
    # TREND TESPİTİ
    # ─────────────────────────────────────────────────────────────────
    def detect_trend(self, df: pd.DataFrame, mode: str = "0dte") -> str:
        """
        0DTE (5m)  : EMA9 + EMA21 — intraday momentum
        Swing (30m): EMA21 + SMA50 + SMA200 (SMA200 yoksa iki katmanlı çalışır)
        """
        close = df['close']

        if mode == "0dte":
            if len(df) < 21:
                return "VERİ_YETERSİZ"
            ema9   = ta.trend.ema_indicator(close, window=9)
            ema21  = ta.trend.ema_indicator(close, window=21)
            curr_p = close.iloc[-1]
            e9, e21 = ema9.iloc[-1], ema21.iloc[-1]

            if curr_p > e9 > e21:    return "BOGA_GUÇLU"
            elif curr_p > e9:        return "BOGA_ZAYIF"
            elif curr_p < e9 < e21:  return "AYI_GUÇLU"
            elif curr_p < e9:        return "AYI_ZAYIF"
            return "YATAY"

        else:  # swing — 30m
            if len(df) < 50:
                return "VERİ_YETERSİZ"

            ema21  = ta.trend.ema_indicator(close, window=21)
            sma50  = ta.trend.sma_indicator(close, window=50)
            curr_p = close.iloc[-1]
            e21, s50 = ema21.iloc[-1], sma50.iloc[-1]

            s200 = None
            if len(df) >= 200:
                s200 = ta.trend.sma_indicator(close, window=200).iloc[-1]

            if s200 is not None:
                if curr_p > e21 > s50 > s200:    return "BOGA_GUÇLU"
                elif curr_p > e21 and e21 > s50:  return "BOGA_ORTA"
                elif curr_p > e21:                return "BOGA_ZAYIF"
                elif curr_p < e21 < s50 < s200:   return "AYI_GUÇLU"
                elif curr_p < e21 and e21 < s50:  return "AYI_ORTA"
                elif curr_p < e21:                return "AYI_ZAYIF"
            else:
                if curr_p > e21 > s50:    return "BOGA_GUÇLU"
                elif curr_p > e21:        return "BOGA_ZAYIF"
                elif curr_p < e21 < s50:  return "AYI_GUÇLU"
                elif curr_p < e21:        return "AYI_ZAYIF"

            return "YATAY"

    # ─────────────────────────────────────────────────────────────────
    # FİBONACCİ
    # ─────────────────────────────────────────────────────────────────
    def calculate_fibonacci(self, df: pd.DataFrame, mode: str = "0dte") -> dict:
        """
        0DTE  → son 50 mum  (5m  × 50 = ~4 saatlik intraday swing)
        Swing → son 100 mum (30m × 100 = ~2 iş günü)
        Oranlar: Standart retracement + Gartley/Bat/Crab uzantıları
        """
        lookback = 50 if mode == "0dte" else 100
        recent   = df.tail(lookback)

        hi   = recent['high'].max()
        lo   = recent['low'].min()
        diff = hi - lo

        if diff == 0:
            return {}

        ratios = {
            "Fib_1.000":   1.000,
            "Fib_0.886":   0.886,
            "Fib_0.807":   0.807,
            "Fib_0.786":   0.786,
            "Fib_0.707":   0.707,
            "Fib_0.618":   0.618,
            "Fib_0.500":   0.500,
            "Fib_0.382":   0.382,
            "Fib_0.214":   0.214,
            "Fib_0.000":   0.000,
            "Fib_-0.118": -0.118,
            "Fib_-0.216": -0.216,
            "Fib_-0.270": -0.270,
            "Fib_-0.414": -0.414,
            "Fib_-0.618": -0.618,
            "Fib_-0.786": -0.786,
            "Fib_-1.000": -1.000,
        }

        fibs_raw = {label: round(hi - (diff * r), 2) for label, r in ratios.items()}
        # Fiyat sırasına göre büyükten küçüğe sırala (swing high → swing low → uzantılar)
        fibs = dict(sorted(fibs_raw.items(), key=lambda x: x[1], reverse=True))
        curr_price                = df['close'].iloc[-1]
        yakin_seviye, yakin_fiyat = self._fib_proximity(curr_price, fibs)

        return {
            "swing_high":   round(hi, 2),
            "swing_low":    round(lo, 2),
            "lookback_mum": lookback,
            "seviyeler":    fibs,
            "yakin_seviye": yakin_seviye,
            "yakin_fiyat":  yakin_fiyat,
        }

    def _fib_proximity(self, price: float, fibs: dict, tolerance: float = 0.003) -> tuple:
        """Fiyatın %0.3 yakınındaki Fib seviyesini döndürür."""
        for label, level in fibs.items():
            if level > 0 and abs(price - level) / level < tolerance:
                return label, level
        return None, None

    # ─────────────────────────────────────────────────────────────────
    # MUM FORMASYONLARI
    # ─────────────────────────────────────────────────────────────────
    def detect_candle_patterns(self, df: pd.DataFrame, mode: str = "0dte") -> str:
        if len(df) < 3:
            return "VERİ_YETERSİZ"

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        body       = abs(curr['close'] - curr['open'])
        wick_total = curr['high'] - curr['low']
        upper_wick = curr['high'] - max(curr['open'], curr['close'])
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        avg_body   = df['close'].diff().abs().mean()

        if wick_total == 0:
            return "NORMAL"

        if (body / wick_total) < 0.1:
            return "DOJI"

        if body > avg_body * 2 and upper_wick < body * 0.1 and lower_wick < body * 0.1:
            return "MARUBOZU"

        if lower_wick > body * 2 and upper_wick < body * 0.5:
            return "CEKIC"

        if upper_wick > body * 2 and lower_wick < body * 0.5:
            return "KAYAN_YILDIZ"

        if (curr['close'] > curr['open']
                and prev['close'] < prev['open']
                and curr['close'] > prev['open']
                and curr['open']  < prev['close']):
            return "YUTAN_BOGA"

        if (curr['close'] < curr['open']
                and prev['close'] > prev['open']
                and curr['close'] < prev['open']
                and curr['open']  > prev['close']):
            return "YUTAN_AYI"

        return "NORMAL"

    # ─────────────────────────────────────────────────────────────────
    # RSI UYUMSUZLUK
    # ─────────────────────────────────────────────────────────────────
    def check_rsi_divergence(self, df: pd.DataFrame, mode: str = "0dte") -> str:
        """
        0DTE: 14 mum geriye — kısa vadeli
        Swing: 30 mum geriye — daha güvenilir
        """
        lookback = 14 if mode == "0dte" else 30

        if len(df) < lookback + 2 or 'RSI' not in df.columns:
            return "YOK"

        window   = df.tail(lookback)
        prices   = window['close'].values
        rsi_vals = window['RSI'].values
        mid      = lookback // 2

        # Bullish: fiyat lower low, RSI higher low
        if (prices[mid:].min() < prices[:mid].min()
                and rsi_vals[mid + prices[mid:].argmin()] > rsi_vals[prices[:mid].argmin()]):
            return "POZITIF"

        # Bearish: fiyat higher high, RSI lower high
        if (prices[mid:].max() > prices[:mid].max()
                and rsi_vals[mid + prices[mid:].argmax()] < rsi_vals[prices[:mid].argmax()]):
            return "NEGATIF"

        return "YOK"

    # ─────────────────────────────────────────────────────────────────
    # ANA HESAPLAMA
    # ─────────────────────────────────────────────────────────────────
    async def calculate_indicators(self, df: pd.DataFrame, mode: str = "0dte") -> dict | None:
        min_rows = 21 if mode == "0dte" else 50
        if df.empty or len(df) < min_rows:
            logger.warning(f"Yetersiz veri: {len(df)} mum ({mode} için min {min_rows})")
            return None

        try:
            df        = df.copy()
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)

            trend      = self.detect_trend(df, mode)
            formation  = self.detect_candle_patterns(df, mode)
            divergence = self.check_rsi_divergence(df, mode)
            fib        = self.calculate_fibonacci(df, mode)

            curr_price = round(float(df['close'].iloc[-1]), 2)
            curr_rsi   = round(float(df['RSI'].iloc[-1]),   2)

            fib_yakin_fiyat = fib.get("yakin_fiyat")

            result = {
                "mode":             mode,
                "price":            curr_price,
                "rsi":              curr_rsi,
                "trend":            trend,
                "formation":        formation,
                "divergence":       divergence,
                "fib_swing_high":   round(float(fib.get("swing_high", 0) or 0), 2) or None,
                "fib_swing_low":    round(float(fib.get("swing_low",  0) or 0), 2) or None,
                "fib_yakin_seviye": fib.get("yakin_seviye"),
                "fib_yakin_fiyat":  round(float(fib_yakin_fiyat), 2) if fib_yakin_fiyat else None,
                "fib_data":         fib.get("seviyeler", {}),
            }

            logger.info(
                f"📊 [{mode.upper()}] {curr_price} | RSI {curr_rsi} | "
                f"Trend: {trend} | Form: {formation} | "
                f"Div: {divergence} | Fib: {fib.get('yakin_seviye', '-')}"
            )
            return result

        except Exception as e:
            logger.error(f"Hesaplama hatası ({mode}): {e}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # DB
    # ─────────────────────────────────────────────────────────────────
    async def ensure_tables(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            for table, default_mode in [("technical_0dte", "0dte"), ("technical_swing", "swing")]:
                await conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {table} (
                        id               SERIAL PRIMARY KEY,
                        timestamp        TIMESTAMPTZ DEFAULT NOW(),
                        symbol           TEXT NOT NULL,
                        timeframe        TEXT NOT NULL,
                        mode             TEXT DEFAULT '{default_mode}',
                        price            NUMERIC(10,2),
                        rsi              NUMERIC(6,2),
                        trend            TEXT,
                        formation        TEXT,
                        divergence       TEXT,
                        fib_swing_high   NUMERIC(10,2),
                        fib_swing_low    NUMERIC(10,2),
                        fib_yakin_seviye TEXT,
                        fib_yakin_fiyat  NUMERIC(10,2),
                        fib_data         JSONB
                    )
                ''')
                await conn.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table}_symbol_ts
                    ON {table}(symbol, timestamp DESC)
                ''')
        # Mevcut tablolarda eski NUMERIC kolonlarını NUMERIC(10,2)'ye güncelle
        numeric_cols = {
            'price':           'NUMERIC(10,2)',
            'rsi':             'NUMERIC(6,2)',
            'fib_swing_high':  'NUMERIC(10,2)',
            'fib_swing_low':   'NUMERIC(10,2)',
            'fib_yakin_fiyat': 'NUMERIC(10,2)',
        }
        async with pool.acquire() as conn:
            for table in ['technical_0dte', 'technical_swing']:
                for col, typ in numeric_cols.items():
                    try:
                        await conn.execute(
                            f'ALTER TABLE {table} ALTER COLUMN {col} '
                            f'TYPE {typ} USING round({col}::numeric, 2)'
                        )
                    except Exception:
                        pass  # Kolon zaten doğru tipteyse sessizce geç
        logger.info("✅ DB tabloları hazır.")

    async def save_to_db(self, symbol: str, timeframe: str, data: dict, mode: str = "0dte"):
        if not data:
            return
        table = "technical_0dte" if mode == "0dte" else "technical_swing"
        try:
            # Precision: tüm numeric alanları kayıt öncesi round'la
            def r2(v): return round(float(v), 2) if v is not None else None

            # Fib data: fiyat sırasına göre büyükten küçüğe sırala
            fib_raw  = data.get("fib_data", {})
            fib_data = dict(sorted(fib_raw.items(), key=lambda x: x[1], reverse=True))

            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(f'''
                    INSERT INTO {table}
                        (symbol, timeframe, mode, price, rsi, trend, formation, divergence,
                         fib_swing_high, fib_swing_low, fib_yakin_seviye, fib_yakin_fiyat, fib_data)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)
                ''',
                    symbol, timeframe, mode,
                    r2(data["price"]), r2(data["rsi"]),
                    data["trend"], data["formation"], data["divergence"],
                    r2(data.get("fib_swing_high")), r2(data.get("fib_swing_low")),
                    data.get("fib_yakin_seviye"), r2(data.get("fib_yakin_fiyat")),
                    json.dumps(fib_data)
                )
            logger.info(f"💾 [{mode.upper()}] {symbol} {timeframe} → {table}")
        except Exception as e:
            logger.error(f"DB kayıt hatası ({symbol} {mode}): {e}")

    # ─────────────────────────────────────────────────────────────────
    # ANALİZ ÇALIŞTIRICI
    # ─────────────────────────────────────────────────────────────────
    async def run_0dte(self, symbol: str):
        """0DTE: Massive'den 5m veri çek → hesapla → kaydet."""
        df = await self.fetch_ohlcv_massive(symbol, "5", "minute")
        if df.empty:
            return

        # Piyasa kapalı kontrolü — son mum 15 dk'dan eskiyse yazma
        son_mum = df['timestamp'].iloc[-1]
        if son_mum.tzinfo is None:
            son_mum = son_mum.tz_localize('UTC')
        fark = (pd.Timestamp.now(tz='UTC') - son_mum).total_seconds() / 60
        if fark > 15:
            logger.info(f"⏸️ [0DTE] {symbol} son mum {round(fark)} dk önce — piyasa kapalı, atlanıyor.")
            return

        data = await self.calculate_indicators(df, mode="0dte")
        await self.save_to_db(symbol, "5m", data, mode="0dte")

    async def run_swing(self, symbol: str):
        """
        Swing: yfinance cache'den 30m veri al → hesapla → kaydet.
        İlk çağrıda warm-up (60 gün), sonrası incremental (2 gün).
        30m mum + 30dk scheduler = max 60dk tolerans.
        """
        df = await self.get_swing_df(symbol)
        if df.empty:
            return

        # Piyasa kapalı kontrolü — son mum 60 dk'dan eskiyse yazma
        son_mum = df['timestamp'].iloc[-1]
        if son_mum.tzinfo is None:
            son_mum = son_mum.tz_localize('UTC')
        fark = (pd.Timestamp.now(tz='UTC') - son_mum).total_seconds() / 60
        if fark > 60:
            logger.info(f"⏸️ [SWING] {symbol} son mum {round(fark)} dk önce — piyasa kapalı, atlanıyor.")
            return

        data = await self.calculate_indicators(df, mode="swing")
        await self.save_to_db(symbol, "30m", data, mode="swing")

    # ─────────────────────────────────────────────────────────────────
    # SCHEDULER
    # ─────────────────────────────────────────────────────────────────
    async def run_scheduler(self):
        logger.info("🚀 TechnicalAgent başlatıldı. SPY & QQQ takibi aktif.")
        logger.info("   0DTE  → Massive API (5m, her 5 dakika)")
        logger.info("   Swing → yfinance   (30m, cache, 27. ve 57. dakika)")

        await self.ensure_tables()

        # Swing warm-up: başlangıçta hemen çalıştır
        logger.info("🔄 Swing warm-up başlatılıyor...")
        await asyncio.gather(
            self.run_swing("SPY"),
            self.run_swing("QQQ"),
            return_exceptions=True
        )
        logger.info("✅ Swing warm-up tamamlandı.")

        # 0DTE: her 5 dakikada bir
        m_0dte  = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58]
        # Swing: n8n'den 3 dakika önce
        m_swing = [27, 57]

        last_run = -1

        while True:
            now = datetime.datetime.now()

            if now.minute != last_run:
                tasks = []
                for sym in ["SPY", "QQQ"]:
                    if now.minute in m_0dte:
                        tasks.append(self.run_0dte(sym))
                    if now.minute in m_swing:
                        tasks.append(self.run_swing(sym))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                last_run = now.minute

            await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────
# BAŞLATMA
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = TechnicalAgent()
    try:
        asyncio.run(agent.run_scheduler())
    except KeyboardInterrupt:
        logger.info("⛔ Agent durduruldu.")
