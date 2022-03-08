import tushare as ts

class Get_data:

    # 获取深证成份指数
    def __init__(self, start_date, end_date, ts_code):
        ts.set_token('752ba2cd8f48df61d08385e43ba808cd0e6d0e8a8ceca52f86648f67')                    # 设置token
        pro = ts.pro_api()                                                                          # 初始化pro接口
        self.df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)        # 返回深证成份指数数据,df为DataFrame类型（表格）数据