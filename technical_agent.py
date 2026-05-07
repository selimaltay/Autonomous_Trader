import asyncio
import datetime
import logging
import os
import json

import asyncpg
import pandas as pd
import yfinance as yf
import ta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TechnicalAgent")

# ─────────────────────────────────────────────────────────────────────────────
# SABITLER
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS        = ["SPY", "QQQ"]
TABLE_0DTE     = "technical_0dte"
TABLE_SWING    = "technical_swing"

# Çalışma saatleri (TSİ = UTC+3)
TSI            = datetime.timezone(datetime.timedelta(hours=3))
SAAT_BASLANGIC = 15   # 15:00 TSİ
SAAT_BITIS     = 24   # 24:00 TSİ

# Scheduler dakikaları
MINUTES_0DTE   = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58]
MINUTES_SWING  = [27, 57]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────────────
class TechnicalAgent:

    def __init__(self):
        self.db_dsn = os.getenv("DATABASE_URL")
        if not self.db_dsn:
            raise RuntimeError("DATABASE_URL ortam değişkeni tanımlı değil!")
        self.pool = None

        self._swing_cache: dict[str, pd.DataFrame] = {}
        self._swing_warmup_done: set[str]           = set()

    # ─────────────────────────────────────────────────────────────────────────
    # DB POOL
    # ─────────────────────────────────────────────────────────────────────────
    async def get_pool(self) -> asyncpg.Pool:
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.db_dsn,
                min_size=2,
                max_size=5,
                command_timeout=30,
            )
        return self.pool

    # ─────────────────────────────────────────────────────────────────────────
    # TABLO OLUŞTUR
    # ─────────────────────────────────────────────────────────────────────────
    async def ensure_tables(self):
        pool = await self.get_pool()
        ddl = """
            CREATE TABLE IF NOT EXISTS {table} (
                id               SERIAL PRIMARY KEY,
                timestamp        TIMESTAMPTZ DEFAULT NOW(),
                symbol           TEXT        NOT NULL,
                timeframe        TEXT        NOT NULL,
                mode             TEXT        NOT NULL,
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
            );
            CREATE INDEX IF NOT EXISTS idx_{safe}_sym_ts
                ON {table}(symbol, timestamp DESC);
        """
        async with pool.acquire() as conn:
            for table in [TABLE_0DTE, TABLE_SWING]:
                safe = table.replace("-", "_")
                await conn.execute(ddl.format(table=table, safe=safe))
        logger.info(f"✅ Tablolar hazır: {TABLE_0DTE}, {TABLE_SWING}")

    # ─────────────────────────────────────────────────────────────────────────
    # VERİ ÇEKME
    # ─────────────────────────────────────────────────────────────────────────
    def _fetch_yfinance(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        try:
            df_raw = yf.Ticker(symbol).history(period=period, interval=interval)
            if df_raw.empty:
                logger.warning(f"[yfinance] {symbol} {interval} boş döndü.")
                return pd.DataFrame()

            df = (
                df_raw
                .reset_index()
                .rename(columns={
                    "Datetime": "timestamp",
                    "Date":     "timestamp",
                    "Open":     "open",
                    "High":     "high",
                    "Low":      "low",
                    "Close":    "close",
                    "Volume":   "volume",
                })[["timestamp", "open", "high", "low", "close", "volume"]]
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_localize(None)

            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"✅ [yfinance] {symbol} {interval} — {len(df)} mum")
            return df

        except Exception as e:
            logger.error(f"[yfinance] Hata ({symbol} {interval}): {e}")
            return pd.DataFrame()

    def fetch_5m(self, symbol: str) -> pd.DataFrame:
        return self._fetch_yfinance(symbol, interval="5m", period="5d")

    def fetch_30m(self, symbol: str, period: str = "60d") -> pd.DataFrame:
        return self._fetch_yfinance(symbol, interval="30m", period=period)

    # ─────────────────────────────────────────────────────────────────────────
    # SWING CACHE
    # ─────────────────────────────────────────────────────────────────────────
    async def get_swing_df(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._swing_warmup_done:
            logger.info(f"🔄 [Cache] {symbol} warm-up (60 gün)...")
            df = await asyncio.to_thread(self.fetch_30m, symbol, "60d")
            if df.empty:
                logger.error(f"❌ [Cache] {symbol} warm-up başarısız.")
                return pd.DataFrame()
            self._swing_cache[symbol] = df
            self._swing_warmup_done.add(symbol)
            logger.info(f"✅ [Cache] {symbol} — {len(df)} mum bellekte.")
        else:
            yeni = await asyncio.to_thread(self.fetch_30m, symbol, "2d")
            if not yeni.empty:
                combined = (
                    pd.concat([self._swing_cache.get(symbol, pd.DataFrame()), yeni])
                    .drop_duplicates(subset="timestamp", keep="last")
                    .sort_values("timestamp")
                    .tail(400)
                    .reset_index(drop=True)
                )
                self._swing_cache[symbol] = combined
                logger.info(f"🔁 [Cache] {symbol} güncellendi — {len(combined)} mum.")

        return self._swing_cache.get(symbol, pd.DataFrame())

    # ─────────────────────────────────────────────────────────────────────────
    # TREND
    # ─────────────────────────────────────────────────────────────────────────
    def detect_trend(self, df: pd.DataFrame, mode: str) -> str:
        close = df["close"]

        if mode == "0dte":
            if len(df) < 21:
                return "VERİ_YETERSİZ"
            e9  = ta.trend.ema_indicator(close, window=9).iloc[-1]
            e21 = ta.trend.ema_indicator(close, window=21).iloc[-1]
            p   = close.iloc[-1]
            if p > e9 > e21:   return "BOGA_GUÇLU"
            elif p > e9:       return "BOGA_ZAYIF"
            elif p < e9 < e21: return "AYI_GUÇLU"
            elif p < e9:       return "AYI_ZAYIF"
            return "YATAY"

        else:  # swing
            if len(df) < 50:
                return "VERİ_YETERSİZ"
            e21  = ta.trend.ema_indicator(close, window=21).iloc[-1]
            s50  = ta.trend.sma_indicator(close, window=50).iloc[-1]
            p    = close.iloc[-1]
            s200 = None
            if len(df) >= 200:
                s200 = ta.trend.sma_indicator(close, window=200).iloc[-1]

            if s200 is not None:
                if p > e21 > s50 > s200:    return "BOGA_GUÇLU"
                elif p > e21 and e21 > s50: return "BOGA_ORTA"
                elif p > e21:               return "BOGA_ZAYIF"
                elif p < e21 < s50 < s200:  return "AYI_GUÇLU"
                elif p < e21 and e21 < s50: return "AYI_ORTA"
                elif p < e21:               return "AYI_ZAYIF"
            else:
                if p > e21 > s50:   return "BOGA_GUÇLU"
                elif p > e21:       return "BOGA_ZAYIF"
                elif p < e21 < s50: return "AYI_GUÇLU"
                elif p < e21:       return "AYI_ZAYIF"
            return "YATAY"

    # ─────────────────────────────────────────────────────────────────────────
    # FİBONACCİ
    # ─────────────────────────────────────────────────────────────────────────
    FIB_RATIOS = {
        "Fib_1.000":   1.000, "Fib_0.886":  0.886, "Fib_0.807":  0.807,
        "Fib_0.786":   0.786, "Fib_0.707":  0.707, "Fib_0.618":  0.618,
        "Fib_0.500":   0.500, "Fib_0.382":  0.382, "Fib_0.214":  0.214,
        "Fib_0.000":   0.000, "Fib_-0.118": -0.118, "Fib_-0.216": -0.216,
        "Fib_-0.270": -0.270, "Fib_-0.414": -0.414, "Fib_-0.618": -0.618,
        "Fib_-0.786": -0.786, "Fib_-1.000": -1.000,
    }

    def calculate_fibonacci(self, df: pd.DataFrame, mode: str) -> dict:
        lookback = 50 if mode == "0dte" else 100
        recent   = df.tail(lookback)
        hi, lo   = recent["high"].max(), recent["low"].min()
        diff     = hi - lo

        if diff == 0:
            return {}

        fibs = dict(sorted(
            {label: round(hi - diff * r, 2) for label, r in self.FIB_RATIOS.items()}.items(),
            key=lambda x: x[1], reverse=True
        ))

        curr = df["close"].iloc[-1]
        yakin_seviye, yakin_fiyat = None, None
        for label, level in fibs.items():
            if level > 0 and abs(curr - level) / level < 0.003:
                yakin_seviye, yakin_fiyat = label, level
                break

        return {
            "swing_high":   round(hi, 2),
            "swing_low":    round(lo, 2),
            "seviyeler":    fibs,
            "yakin_seviye": yakin_seviye,
            "yakin_fiyat":  yakin_fiyat,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MUM FORMASYONLARI
    # ─────────────────────────────────────────────────────────────────────────
    def detect_candle(self, df: pd.DataFrame) -> str:
        if len(df) < 3:
            return "VERİ_YETERSİZ"

        prev       = df.iloc[-2]
        curr       = df.iloc[-1]
        body       = abs(curr["close"] - curr["open"])
        wick_total = curr["high"] - curr["low"]
        upper_wick = curr["high"] - max(curr["open"], curr["close"])
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        avg_body   = df["close"].diff().abs().mean()

        if wick_total == 0:
            return "NORMAL"
        if body / wick_total < 0.1:
            return "DOJI"
        if body > avg_body * 2 and upper_wick < body * 0.1 and lower_wick < body * 0.1:
            return "MARUBOZU"
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            return "CEKIC"
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            return "KAYAN_YILDIZ"
        if (curr["close"] > curr["open"]
                and prev["close"] < prev["open"]
                and curr["close"] > prev["open"]
                and curr["open"]  < prev["close"]):
            return "YUTAN_BOGA"
        if (curr["close"] < curr["open"]
                and prev["close"] > prev["open"]
                and curr["close"] < prev["open"]
                and curr["open"]  > prev["close"]):
            return "YUTAN_AYI"
        return "NORMAL"

    # ─────────────────────────────────────────────────────────────────────────
    # RSI DİVERJANS
    # ─────────────────────────────────────────────────────────────────────────
    def check_divergence(self, df: pd.DataFrame, mode: str) -> str:
        lookback = 14 if mode == "0dte" else 30
        if len(df) < lookback + 2 or "RSI" not in df.columns:
            return "YOK"

        w      = df.tail(lookback)
        prices = w["close"].values
        rsi    = w["RSI"].values
        mid    = lookback // 2

        p_first, p_second = prices[:mid], prices[mid:]
        r_first, r_second = rsi[:mid],    rsi[mid:]

        if (p_second.min() < p_first.min()
                and r_second[p_second.argmin()] > r_first[p_first.argmin()]):
            return "POZITIF"

        if (p_second.max() > p_first.max()
                and r_second[p_second.argmax()] < r_first[p_first.argmax()]):
            return "NEGATIF"

        return "YOK"

    # ─────────────────────────────────────────────────────────────────────────
    # ANA HESAPLAMA
    # ─────────────────────────────────────────────────────────────────────────
    async def calculate(self, df: pd.DataFrame, mode: str) -> dict | None:
        min_rows = 21 if mode == "0dte" else 50
        if df.empty or len(df) < min_rows:
            logger.warning(f"⚠️ Yetersiz veri: {len(df)} mum (min {min_rows})")
            return None

        try:
            df = df.copy()
            df["RSI"] = ta.momentum.rsi(df["close"], window=14)

            trend      = self.detect_trend(df, mode)
            formation  = self.detect_candle(df)
            divergence = self.check_divergence(df, mode)
            fib        = self.calculate_fibonacci(df, mode)

            result = {
                "mode":             mode,
                "price":            round(float(df["close"].iloc[-1]), 2),
                "rsi":              round(float(df["RSI"].iloc[-1]),   2),
                "trend":            trend,
                "formation":        formation,
                "divergence":       divergence,
                "fib_swing_high":   round(float(fib["swing_high"]), 2) if fib.get("swing_high") else None,
                "fib_swing_low":    round(float(fib["swing_low"]),  2) if fib.get("swing_low")  else None,
                "fib_yakin_seviye": fib.get("yakin_seviye"),
                "fib_yakin_fiyat":  round(float(fib["yakin_fiyat"]), 2) if fib.get("yakin_fiyat") else None,
                "fib_data":         fib.get("seviyeler", {}),
            }

            logger.info(
                f"📊 [{mode.upper()}] {result['price']} | RSI {result['rsi']} | "
                f"Trend: {trend} | Form: {formation} | Div: {divergence} | "
                f"Fib: {fib.get('yakin_seviye', '-')}"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Hesaplama hatası ({mode}): {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # DB KAYIT
    # ─────────────────────────────────────────────────────────────────────────
    async def save(self, symbol: str, timeframe: str, data: dict, mode: str):
        if not data:
            logger.warning(f"⚠️ [{mode.upper()}] {symbol} — kaydedilecek veri yok.")
            return

        table    = TABLE_0DTE if mode == "0dte" else TABLE_SWING
        fib_data = dict(sorted(
            data.get("fib_data", {}).items(),
            key=lambda x: x[1], reverse=True
        ))

        def r2(v):
            return round(float(v), 2) if v is not None else None

        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                        (symbol, timeframe, mode, price, rsi, trend, formation, divergence,
                         fib_swing_high, fib_swing_low, fib_yakin_seviye, fib_yakin_fiyat, fib_data)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)
                    """,
                    symbol, timeframe, mode,
                    r2(data["price"]),              r2(data["rsi"]),
                    data["trend"],                  data["formation"],   data["divergence"],
                    r2(data.get("fib_swing_high")), r2(data.get("fib_swing_low")),
                    data.get("fib_yakin_seviye"),   r2(data.get("fib_yakin_fiyat")),
                    json.dumps(fib_data),
                )
            logger.info(f"💾 [{mode.upper()}] {symbol} {timeframe} → {table} ✅")
        except Exception as e:
            logger.error(f"❌ DB kayıt hatası ({symbol} {mode}): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # ÇALIŞTIRICILAR
    # ─────────────────────────────────────────────────────────────────────────
    async def run_0dte(self, symbol: str):
        df = await asyncio.to_thread(self.fetch_5m, symbol)
        if df.empty:
            return
        data = await self.calculate(df, mode="0dte")
        await self.save(symbol, "5m", data, mode="0dte")

    async def run_swing(self, symbol: str):
        df = await self.get_swing_df(symbol)
        if df.empty:
            return
        data = await self.calculate(df, mode="swing")
        await self.save(symbol, "30m", data, mode="swing")

    # ─────────────────────────────────────────────────────────────────────────
    # SCHEDULER
    # ─────────────────────────────────────────────────────────────────────────
    async def run_scheduler(self):
        logger.info("🚀 TechnicalAgent başlatıldı.")
        logger.info(f"   Semboller    : {SYMBOLS}")
        logger.info(f"   Çalışma saati: {SAAT_BASLANGIC}:00 - {SAAT_BITIS}:00 TSİ (UTC+3)")
        logger.info(f"   0DTE  → {TABLE_0DTE}  | 5m  | dakikalar: {MINUTES_0DTE}")
        logger.info(f"   Swing → {TABLE_SWING} | 30m | dakikalar: {MINUTES_SWING}")

        await self.ensure_tables()

        logger.info("🔄 Başlangıç warm-up...")
        await asyncio.gather(*[self.run_swing(s) for s in SYMBOLS], return_exceptions=True)
        logger.info("✅ Başlangıç warm-up tamamlandı.")

        last_run      = -1
        saat_disi_log = False

        while True:
            now  = datetime.datetime.now(TSI)
            saat = now.hour

            # ── Çalışma saati dışı ───────────────────────────────────
            if not (SAAT_BASLANGIC <= saat < SAAT_BITIS):
                if not saat_disi_log:
                    logger.info(
                        f"😴 TSİ {now.strftime('%H:%M')} — çalışma saati dışında "
                        f"({SAAT_BASLANGIC}:00-{SAAT_BITIS}:00 TSİ). Bekleniyor..."
                    )
                    saat_disi_log = True
                    last_run      = -99
                await asyncio.sleep(30)
                continue

            # ── Saat aralığına yeni girildi → gün başı warm-up ───────
            if last_run == -99:
                logger.info(
                    f"⏰ TSİ {now.strftime('%H:%M')} — çalışma saati başladı, "
                    f"gün başı warm-up yapılıyor..."
                )
                self._swing_warmup_done.clear()
                self._swing_cache.clear()
                await asyncio.gather(*[self.run_swing(s) for s in SYMBOLS], return_exceptions=True)
                logger.info("✅ Gün başı warm-up tamamlandı.")
                saat_disi_log = False
                last_run      = -1

            # ── Normal çalışma ────────────────────────────────────────
            if now.minute != last_run:
                tasks = []
                for sym in SYMBOLS:
                    if now.minute in MINUTES_0DTE:
                        tasks.append(self.run_0dte(sym))
                    if now.minute in MINUTES_SWING:
                        tasks.append(self.run_swing(sym))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                last_run = now.minute

            await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = TechnicalAgent()
    try:
        asyncio.run(agent.run_scheduler())
    except KeyboardInterrupt:
        logger.info("⛔ Agent durduruldu.")
