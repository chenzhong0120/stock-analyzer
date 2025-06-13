#上學期
import gradio as gr
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 計算 RSI 指標
def calculate_rsi(data, window=14):
    delta = data.diff()#計算每日價格變動
    gain = (delta.where(delta > 0, 0)).fillna(0)# 將上漲部分取值，其他為 0
    loss = (-delta.where(delta < 0, 0)).fillna(0)# 將下跌部分取值，其他為 0
    avg_gain = gain.rolling(window=window, min_periods=1).mean()# 計算平均漲幅
    avg_loss = loss.rolling(window=window, min_periods=1).mean()# 計算平均跌幅
    rs = avg_gain / avg_loss# 計算相對強弱
    rsi = 100 - (100 / (1 + rs))# 計算 RSI 指標
    return rsi#返回 RSI

# 抓取股票數據並進行分析
def fetch_stock_data(stock_code):
    stock_code = stock_code.strip().upper()# 去除空格並轉換為大寫
    try:
        # 從 Yahoo Finance 下載最近一個月的股票數據
        stock_data = yf.download(stock_code, period="1mo")

        # 獲取股票的基本資訊
        stock_info = yf.Ticker(stock_code).info#yfinance 提供的 Ticker 物件，抓取該股票的基本資訊，如市值、市盈率等。
        market_cap = stock_info.get('marketCap', '資料缺失')# 市值 ，get() 函數用來從上行所下載的資料字典中取值，如果找不到對應的鍵，則返回第二個參數 '資料缺失'。
        pe_ratio = stock_info.get('trailingPE', '資料缺失')# 市盈率
        if pe_ratio == '資料缺失':
          pe_ratio = '無市盈率資料，可能因為虧損或資料未更新'# 處理市盈率缺失的情況

        # 檢查是否有抓取到數據
        if stock_data.empty: #empty 是 pandas.DataFrame 物件的一個屬性，它會返回一個布林值：True表沒有數據，False表有數據。
            return f"無法取得代碼 {stock_code} 的資料，請確認股票代碼是否正確，或稍後再試。", None #當返回ture時輸出此內容，返回None時表示沒有有效的數據可以顯示

        # 計算每日漲跌幅
        stock_data['每日漲跌幅 (%)'] = stock_data['Close'].pct_change() * 100 #pct_change() 是 pandas 的一個函數，用來計算數列中每個元素相對於前一元素的變化百分比。

        # 將RSI指標存入字典中
        stock_data['RSI'] = calculate_rsi(stock_data['Close']) #將前面定義的 calculate_rsi ，存入 stock_data['RSI'] 欄位中。

        # 新增移動平均線
        stock_data['5日移動平均線'] = stock_data['Close'].rolling(window=5).mean() #rolling(window=n) 用來創建一個大小為 n 的數據點，並通過 mean() 計算這個視窗內數據的平均值。
        stock_data['10日移動平均線'] = stock_data['Close'].rolling(window=10).mean()

        # 篩選大幅波動的日期
        large_changes = stock_data[stock_data['每日漲跌幅 (%)'].abs() > 5] #篩選出漲跌幅超過 5% 的日期，定義為市場波動較大的日期。
        large_changes_html = large_changes.to_html() if not large_changes.empty else "無大幅波動的日期"#將大幅波動的日期轉換為 HTML 表格，方便顯示，若沒有大幅波動則另外輸出

        # 生成數據表格
        stock_table = stock_data[['Close', '每日漲跌幅 (%)', '5日移動平均線', '10日移動平均線', 'RSI']].to_html()#將其他數據轉換為HTML表格格式。
        # 繪圖
        fig, ax = plt.subplots(figsize=(8, 5)) #在Matplotlib中繪製一個圖.並設置大小.寬8英尺高5英尺
        for i in range(1, len(stock_data)): #從1開始，到stock_data的長度（len(stock_data)）结束。起始值1是為了要和前一天的數值做比較
            if stock_data['每日漲跌幅 (%)'][i] > 0: #判斷漲跌是否大於0
                ax.plot(stock_data.index[i-1:i+1], stock_data['Close'][i-1:i+1], color="green")  # 上漲用綠色,i-1.i+1是為了求前一天和今天的數據,close指收盤價
            else:
                ax.plot(stock_data.index[i-1:i+1], stock_data['Close'][i-1:i+1], color="red")  # 下跌用紅色
        ax.plot(stock_data.index, stock_data['5日移動平均線'], label="5日移動平均線", color="orange") #ax.plot在ax上繪製數據,五日移動平均值是縱軸,label為曲線的標籤
        ax.plot(stock_data.index, stock_data['10日移動平均線'], label="10日移動平均線", color="green")
        ax.set_title(f"{stock_code} 股票分析") #標題為yahoo股票分析
        ax.set_xlabel("日期")
        ax.set_ylabel("價格")
        ax.legend() #顯示圖利
        plt.xticks(rotation=45)# x 軸日期旋轉以避免重疊
        plt.tight_layout() #調整圖片大小

        # 儲存圖表為圖片
        img_path = "/tmp/stock_chart.png"
        plt.savefig(img_path)

        # 修正的 summary 部分
        summary = f"""
        市值：{market_cap}<br>
        市盈率：{pe_ratio}<br>
        大幅波動的日期：<br>{large_changes_html}<br>
        RSI指標：<br>{stock_data['RSI'].tail(1).values[0]:.2f}
        """

        return summary, img_path

    except Exception as e:
        return f"發生錯誤：{e}", None

# Gradio 介面設定
interface = gr.Interface(
    fn=fetch_stock_data,
    inputs=gr.Textbox(label="股票代碼", placeholder="請輸入股票代碼（例如：2330.TW）"),# 用戶輸入
    outputs=[gr.HTML(), gr.Image()],# 輸出為 HTML 和圖片
    title="股票分析系統",#界面頂部的名稱
    description="輸入股票代碼並查看最新的股票分析結果。",#界面的簡介
    theme="default",#指定了使用 Gradio 的默認主題
    examples=[["2330.TW"], ["AAPL"]],  # 提供一些示範股票代碼
    )

# 啟動 Gradio 介面
interface.launch()
Add main script
