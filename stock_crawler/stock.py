class Stock:
    def __init__(self):
        # 日期
        self.date = ""
        # 开盘
        self.opening = 0
        # 最高指数
        self.high = 0
        # 最低指数
        self.low = 0
        # 收盘
        self.closing = 0
        # 成交量(单位：亿手)
        self.volume = 0
        # 成交额（单位：亿元）
        self.turnover = 0

    def set_opening(self, opening):
        self.opening = opening

    def set_high(self, high):
        self.high = high

    def set_low(self, low):
        self.low = low

    def set_closing(self, closing):
        self.closing = closing

    def set_volume(self, volume):
        self.volume = volume

    def set_turnover(self, turnover):
        self.turnover = turnover