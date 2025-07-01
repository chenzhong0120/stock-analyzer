#下載中文字體
!wget -O TaipeiSansTCBeta-Regular.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download

# 匯入必要的函式庫
import yfinance as yf  # 用於獲取股票數據的 Yahoo Finance API
import pandas as pd    # 數據處理和分析
import numpy as np     # 數值計算
import matplotlib as mpl
import matplotlib.pyplot as plt  # 繪圖
import matplotlib.font_manager as fontManager
import warnings        # 警告控制
from matplotlib.font_manager import fontManager
from datetime import datetime, timedelta  # 日期時間處理
from sklearn.preprocessing import MinMaxScaler # 將資料縮放到 0~1 之間，加快模型收斂速度
from tensorflow.keras.models import Sequential # 建立線性堆疊的模型
from tensorflow.keras.layers import LSTM, Dense # LSTM：處理時間序列資料，Dense：輸出層

fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')

class StockAnalyzer:  # 股票分析器類別
    def __init__(self):  # 初始化方法
        self.stock_data = None  # 儲存股票數據
        self.symbol = None      # 儲存股票代碼

    def get_stock_data(self, symbol, period="1y"):  # 獲取股票數據方法
        """獲取股票數據"""
        try:  # 嘗試執行以下程式碼
            self.symbol = symbol  # 保存股票代碼
            stock = yf.Ticker(symbol)  # 建立 Yahoo Finance Ticker 物件
            self.stock_data = stock.history(period=period)  # 獲取歷史股價數據

            if self.stock_data.empty:  # 檢查是否成功獲取數據
                return False, "無法獲取股票數據，請檢查股票代碼"  # 回傳失敗訊息

            return True, "數據獲取成功"  # 回傳成功訊息
        except Exception as e:  # 捕捉任何異常
            return False, f"獲取數據時發生錯誤: {str(e)}"  # 回傳錯誤訊息

    def calculate_technical_indicators(self):  # 計算技術指標方法
        """計算技術指標"""
        if self.stock_data is None or self.stock_data.empty:  # 檢查數據是否存在
            return None  # 無數據則回傳 None

        df = self.stock_data.copy()  # 複製股票數據避免修改原始數據

        # 移動平均線計算
        df['MA5'] = df['Close'].rolling(window=5).mean()    # 5日移動平均線
        df['MA20'] = df['Close'].rolling(window=20).mean()  # 20日移動平均線
        df['MA60'] = df['Close'].rolling(window=60).mean()  # 60日移動平均線

        # RSI (相對強弱指標) 計算
        delta = df['Close'].diff()  # 計算每日價格變化
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # 計算14日平均上漲幅度
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # 計算14日平均下跌幅度
        rs = gain / loss  # 計算相對強度
        df['RSI'] = 100 - (100 / (1 + rs))  # 計算RSI指標

        # MACD (指數平滑異同移動平均線) 計算
        exp1 = df['Close'].ewm(span=12).mean()  # 12日指數移動平均
        exp2 = df['Close'].ewm(span=26).mean()  # 26日指數移動平均
        df['MACD'] = exp1 - exp2  # MACD線 = 快線 - 慢線
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()  # 9日MACD信號線
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']  # MACD柱狀圖

        # 布林帶 (Bollinger Bands) 計算
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()  # 中軌：20日移動平均
        bb_std = df['Close'].rolling(window=20).std()  # 計算20日標準差
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)  # 上軌：中軌 + 2倍標準差
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)  # 下軌：中軌 - 2倍標準差

        return df  # 回傳包含所有技術指標的數據框

    def analyze_trend(self, df):  # 分析趨勢方法
        """分析趨勢"""
        if df is None or df.empty:  # 檢查數據是否存在
            return "無法分析趨勢"  # 無數據則回傳錯誤訊息

        latest = df.iloc[-1]  # 取得最新一筆數據
        prev = df.iloc[-2] if len(df) > 1 else latest  # 取得前一筆數據

        # 價格趨勢分析
        price_change = latest['Close'] - prev['Close']  # 計算價格變化
        price_change_pct = (price_change / prev['Close']) * 100  # 計算價格變化百分比

        # 移動平均線趨勢分析
        ma_trend = ""  # 初始化趨勢字串
        if pd.notna(latest['MA5']) and pd.notna(latest['MA20']):  # 檢查MA5和MA20是否有效
            if latest['MA5'] > latest['MA20']:  # 短期均線在長期均線上方
                ma_trend = "短期上升趨勢"  # 上升趨勢
            else:  # 短期均線在長期均線下方
                ma_trend = "短期下降趨勢"  # 下降趨勢

        # RSI 分析
        rsi_analysis = ""  # 初始化RSI分析字串
        if pd.notna(latest['RSI']):  # 檢查RSI是否有效
            if latest['RSI'] > 70:  # RSI大於70
                rsi_analysis = "超買區間"  # 表示超買
            elif latest['RSI'] < 30:  # RSI小於30
                rsi_analysis = "超賣區間"  # 表示超賣
            else:  # RSI在30-70之間
                rsi_analysis = "正常區間"  # 表示正常

        # MACD 分析
        macd_analysis = ""  # 初始化MACD分析字串
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):  # 檢查MACD指標是否有效
            if latest['MACD'] > latest['MACD_Signal']:  # MACD線在信號線上方
                macd_analysis = "多頭信號"  # 看多信號
            else:  # MACD線在信號線下方
                macd_analysis = "空頭信號"  # 看空信號

        return {  # 回傳分析結果字典
            'price_change': price_change,  # 價格變化
            'price_change_pct': price_change_pct,  # 價格變化百分比
            'ma_trend': ma_trend,  # 移動平均線趨勢
            'rsi_analysis': rsi_analysis,  # RSI分析結果
            'rsi_value': latest['RSI'] if pd.notna(latest['RSI']) else 0,  # RSI數值
            'macd_analysis': macd_analysis,  # MACD分析結果
            'current_price': latest['Close']  # 當前收盤價
        }

    def generate_signals(self, df):  # 生成交易信號方法
        """生成交易信號"""
        if df is None or df.empty:  # 檢查數據是否存在
            return []  # 無數據則回傳空列表

        signals = []  # 初始化信號列表
        latest = df.iloc[-1]  # 取得最新一筆數據

        # 檢查各項指標，使用 pd.notna() 避免 Series 布林錯誤
        if pd.notna(latest['RSI']):  # 檢查RSI是否有效
            if latest['RSI'] < 30:  # RSI小於30（超賣）
                signals.append("🔥 RSI 超賣信號 - 考慮買入")  # 加入買入信號
            elif latest['RSI'] > 70:  # RSI大於70（超買）
                signals.append("⚠️ RSI 超買信號 - 考慮賣出")  # 加入賣出信號

        if pd.notna(latest['MA5']) and pd.notna(latest['MA20']):  # 檢查移動平均線是否有效
            if latest['MA5'] > latest['MA20'] and latest['Close'] > latest['MA5']:  # 多頭排列且價格在MA5上方
                signals.append("📈 均線多頭排列 - 上升趨勢")  # 加入上升趨勢信號
            elif latest['MA5'] < latest['MA20'] and latest['Close'] < latest['MA5']:  # 空頭排列且價格在MA5下方
                signals.append("📉 均線空頭排列 - 下降趨勢")  # 加入下降趨勢信號

        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):  # 檢查MACD是否有效
            if latest['MACD'] > latest['MACD_Signal'] > 0:  # MACD金叉且在零軸上方
                signals.append("🚀 MACD 金叉且在零軸上方 - 強勢買入")  # 加入強勢買入信號
            elif latest['MACD'] < latest['MACD_Signal'] < 0:  # MACD死叉且在零軸下方
                signals.append("💥 MACD 死叉且在零軸下方 - 強勢賣出")  # 加入強勢賣出信號

        if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):  # 檢查布林帶是否有效
            if latest['Close'] > latest['BB_Upper']:  # 價格突破上軌
                signals.append("⚡ 價格突破布林帶上軌 - 強勢突破")  # 加入突破信號
            elif latest['Close'] < latest['BB_Lower']:  # 價格跌破下軌
                signals.append("🔴 價格跌破布林帶下軌 - 超賣反彈機會")  # 加入反彈機會信號

        return signals if signals else ["📊 當前無明顯交易信號"]  # 回傳信號列表，若無信號則回傳預設訊息

    def print_analysis_report(self, df, analysis, signals):
        """印出分析報告"""
        print("\n" + "=" * 60)  # 印出分隔線
        print(f"📊 {self.symbol} 股票分析報告")  # 印出報告標題
        print("=" * 60)  # 印出分隔線

        # 基本資訊
        latest = df.iloc[-1]  # 取得最新一筆數據
        print(f"📅 分析日期: {latest.name.strftime('%Y-%m-%d')}")  # 印出分析日期
        print(f"💰 當前價格: ${analysis['current_price']:.2f}")  # 印出當前價格
        print(f"📈 價格變化: ${analysis['price_change']:.2f} ({analysis['price_change_pct']:.2f}%)")  # 印出價格變化

        # 技術指標
        print(f"\n🔍 技術指標分析:")  # 印出技術指標標題
        print(f"   • 移動平均線: {analysis['ma_trend']}")  # 印出移動平均線分析
        print(f"   • RSI: {analysis['rsi_value']:.1f} - {analysis['rsi_analysis']}")  # 印出RSI分析
        print(f"   • MACD: {analysis['macd_analysis']}")  # 印出MACD分析

        # 交易信號
        print(f"\n🎯 交易信號:")  # 印出交易信號標題
        for signal in signals:  # 遍歷每個信號
            print(f"   {signal}")  # 印出信號內容

        # 成交量
        if 'Volume' in df.columns:  # 檢查成交量欄位是否存在
            avg_volume = df['Volume'].tail(20).mean()  # 計算20日平均成交量
            latest_volume = latest['Volume']  # 取得最新成交量
            volume_ratio = latest_volume / avg_volume  # 計算成交量比率
            print(f"\n📊 成交量分析:")  # 印出成交量分析標題
            print(f"   • 最新成交量: {latest_volume:,.0f}")  # 印出最新成交量
            print(f"   • 20日平均成交量: {avg_volume:,.0f}")  # 印出平均成交量
            print(f"   • 成交量比率: {volume_ratio:.2f}x")  # 印出成交量比率

        print("=" * 60)  # 印出結尾分隔線

    def plot_analysis(self, df):  # 繪製分析圖表方法
        """繪製分析圖表"""
        if df is None or df.empty:  # 檢查數據是否存在
            print("無數據可繪製")  # 印出錯誤訊息
            return  # 結束方法

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))  # 建立4個子圖，垂直排列
        fig.suptitle(f'{self.symbol} 股票技術分析', fontsize=16)  # 設定主標題

        # 價格和移動平均線圖表
        axes[0].plot(df.index, df['Close'], label='收盤價', linewidth=2)  # 繪製收盤價線
        if 'MA5' in df.columns:  # 檢查MA5欄位是否存在
            axes[0].plot(df.index, df['MA5'], label='MA5', alpha=0.7)  # 繪製5日均線
        if 'MA20' in df.columns:  # 檢查MA20欄位是否存在
            axes[0].plot(df.index, df['MA20'], label='MA20', alpha=0.7)  # 繪製20日均線
        if 'MA60' in df.columns:  # 檢查MA60欄位是否存在
            axes[0].plot(df.index, df['MA60'], label='MA60', alpha=0.7)  # 繪製60日均線
        axes[0].set_title('股價走勢與移動平均線')  # 設定子圖標題
        axes[0].legend()  # 顯示圖例
        axes[0].grid(True, alpha=0.3)  # 顯示網格

        # 布林帶圖表
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):  # 檢查布林帶欄位是否都存在
            axes[1].plot(df.index, df['Close'], label='收盤價', linewidth=2)  # 繪製收盤價線
            axes[1].plot(df.index, df['BB_Upper'], label='布林帶上軌', alpha=0.7)  # 繪製上軌
            axes[1].plot(df.index, df['BB_Middle'], label='布林帶中軌', alpha=0.7)  # 繪製中軌
            axes[1].plot(df.index, df['BB_Lower'], label='布林帶下軌', alpha=0.7)  # 繪製下軌
            axes[1].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)  # 填充上下軌之間區域
        axes[1].set_title('布林帶')  # 設定子圖標題
        axes[1].legend()  # 顯示圖例
        axes[1].grid(True, alpha=0.3)  # 顯示網格

        # RSI圖表
        if 'RSI' in df.columns:  # 檢查RSI欄位是否存在
            axes[2].plot(df.index, df['RSI'], label='RSI', color='purple')  # 繪製RSI線
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超買線(70)')  # 繪製超買線
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超賣線(30)')  # 繪製超賣線
            axes[2].axhline(y=50, color='gray', linestyle='-', alpha=0.5)  # 繪製中線
        axes[2].set_title('RSI 相對強弱指標')  # 設定子圖標題
        axes[2].set_ylim(0, 100)  # 設定Y軸範圍
        axes[2].legend()  # 顯示圖例
        axes[2].grid(True, alpha=0.3)  # 顯示網格

        # MACD圖表
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):  # 檢查MACD欄位是否都存在
            axes[3].plot(df.index, df['MACD'], label='MACD', color='blue')  # 繪製MACD線
            axes[3].plot(df.index, df['MACD_Signal'], label='Signal', color='red')  # 繪製信號線
            axes[3].bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.3)  # 繪製柱狀圖
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 繪製零軸線
        axes[3].set_title('MACD')  # 設定子圖標題
        axes[3].legend()  # 顯示圖例
        axes[3].grid(True, alpha=0.3)  # 顯示網格

        plt.tight_layout()  # 自動調整子圖間距
        plt.show()  # 顯示圖表

    def prepare_lstm_data(self, df, sequence_length=30):
        """準備 LSTM 訓練資料"""
        if df is None or df.empty or 'Close' not in df.columns:
          return None, None, None
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
          X.append(scaled_data[i-sequence_length:i, 0])
          y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    def train_lstm_model(self, X, y):
        """建立並訓練 LSTM 模型"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        return model

    def predict_future_price(self, df, model, scaler, sequence_length=30, days=10):
        """用訓練好的模型預測未來幾天價格"""
        last_sequence = df['Close'].values[-sequence_length:].reshape(-1, 1)
        scaled_sequence = scaler.transform(last_sequence)
        X_pred = [scaled_sequence[:, 0]]
        X_pred = np.array(X_pred).reshape(1, sequence_length, 1)

        future_prices = []
        for _ in range(days):
          pred = model.predict(X_pred, verbose=0)
          future_prices.append(pred[0][0])

          # 更新序列
          new_input = np.append(X_pred[0][:, 0][1:], pred[0][0])
          X_pred = np.array([new_input]).reshape(1, sequence_length, 1)

        return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

def main():  # 主程式函數
    analyzer = StockAnalyzer()  # 建立股票分析器實例

    print("🚀 歡迎使用股票分析系統!")  # 印出歡迎訊息
    print("支援格式: 台股請加.TW (如:2330.TW), 美股直接輸入代碼 (如:AAPL)")  # 印出使用說明

    while True:  # 無限迴圈，直到使用者退出
        try:  # 嘗試執行以下程式碼
            symbol = input("\n請輸入股票代碼 (輸入 'quit' 退出): ").strip().upper()  # 取得使用者輸入並轉大寫

            if symbol == 'QUIT':  # 檢查是否要退出
                print("👋 感謝使用股票分析系統!")  # 印出告別訊息
                break  # 跳出迴圈

            if not symbol:  # 檢查輸入是否為空
                print("❌ 請輸入有效的股票代碼")  # 印出錯誤訊息
                continue  # 繼續下一次迴圈

            print(f"📊 正在獲取 {symbol} 的數據...")  # 印出正在獲取數據的訊息

            # 獲取數據
            success, message = analyzer.get_stock_data(symbol)  # 呼叫獲取股票數據方法
            if not success:  # 檢查是否成功獲取數據
                print(f"❌ {message}")  # 印出錯誤訊息
                continue  # 繼續下一次迴圈

            print("✅ 數據獲取成功！")  # 印出成功訊息
            print("🔄 正在計算技術指標...")  # 印出正在計算指標的訊息

            # 計算技術指標
            df = analyzer.calculate_technical_indicators()  # 呼叫計算技術指標方法

            # 分析趨勢
            analysis = analyzer.analyze_trend(df)  # 呼叫分析趨勢方法

            # 生成信號
            signals = analyzer.generate_signals(df)  # 呼叫生成交易信號方法

            # 顯示報告
            analyzer.print_analysis_report(df, analysis, signals)  # 呼叫印出分析報告方法

            # 詢問是否顯示圖表
            show_chart = input("\n是否顯示技術分析圖表? (y/n): ").strip().lower()  # 取得使用者輸入
            if show_chart in ['y', 'yes', '是']:  # 檢查使用者是否同意顯示圖表
                analyzer.plot_analysis(df)  # 呼叫繪製分析圖表方法

            # 訓練 LSTM 並預測未來10天
            X, y, scaler = analyzer.prepare_lstm_data(df)
            model = analyzer.train_lstm_model(X, y)
            future_prices = analyzer.predict_future_price(df, model, scaler, days=10)

            print("\n🔮 預測未來 10 天的價格：")
            for i, price in enumerate(future_prices, 1):
                print(f"第 {i} 天預測價：${price:.2f}")

        except KeyboardInterrupt:
            print("\n👋 程序已中斷，感謝使用!")
            break

        except Exception as e:
            print(f"❌ 發生未預期的錯誤: {str(e)}")
            continue



if __name__ == "__main__":  # 檢查是否為主程式執行
    main()  # 呼叫主程式函數
