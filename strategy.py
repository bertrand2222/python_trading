from sys import float_repr_style
import yfinance as yf
import pandas as pd
import os
from constant import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, shgo, differential_evolution


symbol = "TTE"

# df = yf.download(symbol, period="20y", interval = "1d").bfill()
# df.to_pickle(os.path.join(DATA_PATH,"df_TTE.pkl"))

df_orig = pd.read_pickle(os.path.join(DATA_PATH,"df.pkl"))

NB_DAY_YEAR = int(365 * 5/ 7)


LEN_TURBO_ARRAY = 50
CAP_INIT = 5000
ORD = 1000
ORD_MIN = 100
FEES = 0.0045
STRIKE_LIM = 0.2
CALL = 1
PUT = -1

TAKE_PROFIT = 0.2
WIN_RATIO = 2

class TurboLine():
    def __init__(self,) -> None :
        
        self.activated = False
        self.strike = None
        self.direction = 1
        self.parity = 100
        self.value = 0
        self.quantity = 0

    def open(self, stock_price : float, amount : float = ORD, direction : int = CALL, parity : float = 100, stop_loss = None, win_ratio = WIN_RATIO, strike_lim = 0.2,):
        
        self.activated = True
        self.direction =  direction
        self.parity = parity
        self.strike = stock_price * (1 - strike_lim * direction)
        self.unit_price = self.direction * (stock_price - self.strike) / self.parity
        self.quantity = int(amount / self.unit_price)
        self.value = self.quantity * self.unit_price
        self.cost = self.value * (1+ FEES)

        if not stop_loss is None :
            self.use_stop_loss = True
            self.take_profit_value = self.value * (1 + stop_loss * win_ratio)
            self.stop_loss_value = self.value * (1 - stop_loss)
        else :
            self.use_stop_loss = False
            self.take_profit_value = self.value * (1 +  win_ratio)
            self.stop_loss_value = None
    def close(self) :
        
        self.activated = False
        self.quantity = 0
        self.value = 0
        
    def update_value(self, stock_price : float = None) :
        self.unit_price = self.direction * (stock_price - self.strike) / self.parity
        self.value = self.quantity  * self.unit_price

class Wallet():

    def __init__(self, init_capital : float = CAP_INIT, stop_loss :float = 0.2 , win_ratio : float = WIN_RATIO, strike_lim : float = 0.2, record_cap = True,) -> None:
        self.turbo_line_list = [TurboLine() for i in range(LEN_TURBO_ARRAY)]
        self.disp_capital = init_capital
        self.total_capital = init_capital
        self.stop_loss = stop_loss
        self.win_ratio = win_ratio
        self.strike_lim = strike_lim
        self.record_cap = record_cap
        self.loss = 0
        self.gain = 0
        self.nb_loss = 0
        self.nb_win = 0



    def open_line(self, stock_price : float, amount : float = ORD, direction : int = CALL, parity : float = 100):
        # print("Open line")
        
        l_free_found = False
        for l in self.turbo_line_list:
            if not l.activated :
                l.open(stock_price, amount, direction , parity, self.stop_loss, self.win_ratio, self.strike_lim)
                self.disp_capital -= l.value * (1 + FEES)
                l_free_found = True
                return(0)
        if not l_free_found:
            print('ERROR : not enought space alocated for turbo lines')
            return(1)
    
    def update_day(self, day):

        market = self.df.loc[day]
        
        # amount = min(self.disp_capital * 0.1,ORD)
        amount = self.disp_capital * 0.1
        # amount = ORD
        if amount > ORD_MIN *(1+ FEES):
            # amount = max(self.disp_capital * 0.1, ORD)
            if self.open_line(market['Open'], amount = amount, direction= CALL) : return(1)
            # if self.open_line(market['Open'], amount = amount/2, direction= PUT) : return(1)
        if market['Close'] > market['Open']:
            self.update_movement(market['Low'])
            self.update_movement(market['High'])
        else :
            self.update_movement(market['High'])
            self.update_movement(market['Low'])
        
        self.update_movement(market['Close'])
        
        if self.record_cap :
            self.df.loc[day, "Cap_disp"] = self.disp_capital
            self.df.loc[day, "Cap"] = self.total_capital
            self.df.loc[day, "nb_position"] = len([l for l in self.turbo_line_list if l.activated ])
            self.df.loc[day, "loss"] = self.loss
            self.df.loc[day, "gain"] = self.gain
            self.df.loc[day, "nb_loss"] = self.nb_loss
            self.df.loc[day, "nb_win"] = self.nb_win
            
        return(0)

    def update_movement(self, price) :
        
        
        for l in self.turbo_line_list:
            if not l.activated : continue

            l.update_value(price)
            
            if l.use_stop_loss :
                if l.value < l.stop_loss_value :
                    # stop loss
                    colect = l.stop_loss_value * (1 - FEES)
                    self.disp_capital += colect
                    self.loss += l.cost - colect
                    self.nb_loss += 1
                    l.close()
                    continue
            if l.value > l.take_profit_value :
                # take profit
                colect = l.take_profit_value * (1 - FEES)
                self.disp_capital += colect
                self.gain += (colect - l.cost)
                self.nb_win += 1
                l.close()
                continue
            
            if (price - l.strike) * l.direction <= 0 :
                # strike 
                self.loss += l.cost
                self.nb_loss += 1
                l.close()
                continue
            
            
        self.total_capital = self.disp_capital + sum([l.value for l in self.turbo_line_list])
        
        
    def eval_capital(self, days = NB_DAY_YEAR) :
        
        self.df = df_orig[-days:].copy()
        self.df["Cap"] = self.disp_capital
        self.df["Cap_disp"] = self.disp_capital
        self.df["nb_position"] = 0

        for d in self.df.index :
            if self.update_day(d) : 
                print("ERROR at day {}".format(d))
                exit(0)
        self.df["Cap_position"] = self.df["Cap"] - self.df["Cap_disp"]
        self.df['relative_gain'] = self.df["gain"] - self.df["loss"]
        return(  self.total_capital)


def eval_strategy(ar ):
    w = Wallet(CAP_INIT , stop_loss = ar[0], win_ratio = ar[1], strike_lim = ar[2], record_cap = False,)
    return( - w.eval_capital(NB_DAY_YEAR))

if __name__ == '__main__':

    print(CAP_INIT)

    #### optimisation

    #          stopp loss , win_ratio, strike
    # bounds = [(0.06,0.9), (0.1, 3), (0.04, 0.2)]
    # res = differential_evolution(eval_strategy , bounds,)
    # print(res)

    # w = Wallet(CAP_INIT, stop_loss = res.x[0], win_ratio = res.x[1] , strike_lim = res.x[2], )
    # w = Wallet(CAP_INIT, stop_loss = 0.1 , win_ratio = 3 ,strike_lim = 0.1,  )
    # print("profit %f"%(w.win_ratio * w.stop_loss))
    # cap = w.eval_capital(NB_DAY_YEAR)
    # print(cap)
    # # w.df['Open_evol'] = w.df['Open'] / w.df['Open'].iloc[0]
    # # w.df["Cap_evol"] = w.df['Cap'] / CAP_INIT
    # # w.df[['Cap_evol']].plot()

    df_p = df_orig[-NB_DAY_YEAR:]
    sma = df_p['Open'].rolling(window=R_WIN_SIZE).mean()
    std = df_p['Open'].rolling(window=R_WIN_SIZE).std()
    df_pp = df_p['Open'][R_WIN_SIZE:]
    band_1 = sma + 2*std 
    band_2 = sma - 2*std
    bdw = 4 * std/sma
    # w.df['bb'] = w.df['Close'].sub(sma).divide()
    fig, axs = plt.subplots(2,1)
    
    bdw.plot(ax=axs[0])
    # w.df["bb"].plot(ax=axs[1])
    # w.df["loss"].plot(ax=axs[2])
    df_pp.plot(ax=axs[1])
    band_1.plot(ax=axs[1])
    band_2.plot(ax=axs[1])
    # print(w.df[['Open','relative_gain']])
    
    plt.show()
