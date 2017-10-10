from datetime import datetime, timedelta
from status_data import *
import pandas as pd
import queue
from util import *
from kyroller.engine import Engine
from kyroller.strategy import Strategy
from kyroller.types import *
from ky import api as api
from func import *
import random
import bisect
from decimal import *
import os.path
import zmq
import json as json
import constant as cons
import numpy as np
import signal
import sys
import logging
from operator import itemgetter
from functools import reduce
import random
from MLP import Neural_Networks as brain
from Agent import Agent

Brain = brain(
    n_actions=3,
    n_features=1000,
    neurons_per_layer=np.array([32]),
    learning_rate=0.00025,
    restore=False,
    output_graph=True,
)
RL = Agent(
    brain=Brain,
    n_actions=3,
    observation_space_shape=(1000,),
    reward_decay=0.9,
    replace_target_iter=100,
    memory_size=2000,
    MAX_EPSILON=0.9,
    LAMBDA=0.001,
)


FD = 50000000  # 委买一金额超过5kw
NUM_OF_SHARES = 100
INIT_PRICE = np.zeros(5, dtype=float)
EPISODE = 0

TC = 0.001  # 交易费 千分之一

# quote_work既是30秒quote事件的处理主体,单线程。


def sigHandler_qw(signum, frame):
    # print "got signal", signum
    if signum == signal.SIGINT:
        cons.log.info('quote_worker got Ctrl +C , quit')
        os._exit(1)


class StrategyCY(Strategy):

    def __init__(self, **kwargs):
        Strategy.__init__(self, **kwargs)
        # print(**kwargs)
        self.init_dqn()
        self.total_steps = 0
        self.episode = 0

    def init(self, subs, start_time=None, end_time=None):
        '''
            初始化策略,跨日期全局的初始化。
            每日初始化见nit_curdate(curdate_str)
        '''
        #print('init called',start_time,end_time)

        # 用于发送消息到mesdequer服务
        context2 = zmq.Context()  #
        self.state.msg_push_socket = context2.socket(zmq.PUSH)
        self.state.msg_push_socket.connect(cons.MSG_PUSH_STR)

        self.init_state()
        if start_time is None:
            self.init_curdate(datetime.now().strftime('%Y-%m-%d'))

        self.state.num_trade = 0
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' end init')

    def init_dqn(self):
        self.reward = 0
        self.action = -1

        self.this_state = np.zeros(1000, dtype=float)
        self.pre_state = np.zeros(1000, dtype=float)

        self.state1 = np.zeros(200, dtype=float)
        self.state2 = np.zeros(200, dtype=float)
        self.state3 = np.zeros(200, dtype=float)
        self.state4 = np.zeros(200, dtype=float)
        self.state5 = np.zeros(200, dtype=float)

        self.state_len_per_share = 200

        self.step = 0
        # r = 0

        # is_init = [True, True,True,True,True]

        # price of 5 shares in a timestep
        self.price = np.zeros(5, dtype=float)
        #
        self.volumn = np.array([1000, 1000, 1000, 1000, 1000])
        # count 5 steps for 1 step
        self.cnt = 0
        self.stock_number = 5

        # 流动资金
        self.fund_pool = 50000

        # 交易粒度100股
        self.num_of_shares = 100

    def init_state(self):
        self.state.sslist = {}
        self.state.fixedsslist = {}  # fixedsslistf[date][code] = fixedsslist['2017-04-05'][4000 codes]
        self.state.tradestatuslist = {}  # tradestatuslist[date][code] = tradestatuslist['2017-04-05'][4000 codes]
        self.state.evil_mbar = {}  # 把坏数据的代码放过来 {code:reason}
        self.state.evil_30s = {}  # 把坏数据的代码放过来 {code:reason}
        self.state.evil_3s = {}  # 把坏数据的代码放过来 {code:reason}

        self.state.orderextra_dict_by_oid = {}  # order_id:OrderUnit

        # 目前持有的品种列表
        self.init_symbol_code_list = []
        self.init_positions_dict = {}
        self.init_positions = self.account.get_positions()
        for pos in self.account.get_positions():
            self.init_symbol_code_list.append(pos.symbol)
            self.init_positions_dict[pos.symbol] = pos

        # for symbol in self.init_symbol_code_list:
            # print('初始仓位：股票 %s: %s股'%(symbol,self.account.get_position(symbol)))
        print(self.init_symbol_code_list)

        self.state.env_timestamp = 0

    def init_curdate(self, curdate_str):

        self.state.fixedsslist[curdate_str] = init_all_stock_info(curdate_str)
        self.state.tradestatuslist[curdate_str] = init_all_stock_trade_status(curdate_str)

    def on_market_event(self, e: MarketEvent):
        print('大盘事件', e)
        time_str = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
        date_str = time.strftime("%Y-%m-%d", time.localtime(e.timestamp))

        if e.status == MarketStatus.ND:  # 盘前和盘后要把盘中产生的数据重置。fixedsslist固定数据不用管。
            # reset()
            self.init_state()
            self.init_curdate(date_str)

        if e.status == MarketStatus.PZ:  # 开盘信号
            self.state.marketstatus = 'PZ'

        if e.status == MarketStatus.PH:  # 收盘信号
            for symbol in self.init_symbol_code_list:
                print('收盘仓位：股票 %s: %s' % (symbol, self.account.get_position(symbol)))

            print('account 总值:', self.account.total_value, ' 其中股票市值:', self.account.market_value)

    def on_bar(self, bar: Bar):
        pass

    def on_tick(self, tick: Tick):
        '''
            新的 Tick 到达时的触发,下面是 一个简单策略
        '''
        self.state.env_timestamp = tick.timestamp
        self.state.time_str = time.strftime("%H:%M:%S", time.localtime(self.state.env_timestamp))
        self.state.date_str = time.strftime("%Y-%m-%d", time.localtime(self.state.env_timestamp))

        localtick = localize_tick(tick)
        # if tick.timestamp % 60 ==0 and  tick['code'] == '601318':
        #     print(localtick['time'])

        if not localtick['symbol'] in self.init_symbol_code_list:
            # print('OK')
            return

        #print(datetime.now().strftime('%H:%M:%S'),' before validate')
        #***每个tick都要做的validate
        if not self.validate_env_for_tick_30s(localtick):
            return
        #print(datetime.now().strftime('%H:%M:%S'),' after validate')

        code = tick['code']
        if tick['volume'] == 0 or tick['amount'] == 0:
            # print(tick['time'])
            return
        self.state.sslist[code].avg_price = tick['amount'] / tick['volume']
        # self.do_orders(localtick)
        # self.trade_avg_price(localtick)
        self.ss(localtick)

    # @staticmethod
    def shift_left(self, *state):
        state = state[0]
        for i in range(200 - 1):
            state[i] = state[i + 1]
        state[200 - 1] = 0
        # print (len(state))
        return state

    def ss(self, tick):
        # initialize the price
        if (self.step == 0):
            for i in range(0, 5):
                self.price[i] = INIT_PRICE[i]

        # state
        for i in range(len(self.init_symbol_code_list)):
            if (self.init_symbol_code_list[i][2:] == tick['code']):

                if i == 0:
                    self.state1 = self.shift_left(self.state1)
                    self.state1[199] = float(tick['price']) - float(self.price[i])
                    # print(self.state1[199])
                    # print (self.state1)
                if i == 1:
                    self.state2 = self.shift_left(self.state2)
                    self.state2[199] = float(tick['price']) - float(self.price[i])
                    # print (self.state2)
                if i == 2:
                    self.state3 = self.shift_left(self.state3)
                    self.state3[199] = float(tick['price']) - float(self.price[i])
                    # print (self.state3)
                if i == 3:
                    self.state4 = self.shift_left(self.state4)
                    self.state4[199] = float(tick['price']) - float(self.price[i])
                    # print (tick['time'],self.state4)
                    # print(1)
                if i == 4:
                    self.state5 = self.shift_left(self.state5)
                    self.state5[199] = float(tick['price']) - float(self.price[i])
                    # print (self.state5)

                # update price
                self.price[i] = tick['price']

        # count ticks
        # update the state, take an action and retreive reward every 5 ticks
        self.cnt += 1
        if self.cnt == 5:
            self.cnt = 0
            self.step += 1
            # print (self.step)
            self.pre_state = self.this_state
            self.this_state = np.concatenate((self.state1, self.state2, self.state3, self.state4, self.state5), axis=0)
            # print(self.this_state)

            # 第一次 此时 只有self.this_state 第二次才保存
            if self.action != -1:
                RL.store_memory(self.pre_state, self.action, self.reward, self.this_state)
                self.total_steps += 1
                if self.total_steps > 50:
                    RL.learn()

            self.action = RL.choose_action(self.this_state)

            # final reward
            if tick['time'][0:2] == '15':
                self.episode += 1
                self.reward = self.get_final_reward() + self.fund_pool
                self.pre_state = self.this_state
                self.this_state = None
                RL.store_memory(self.pre_state, self.action, self.reward, self.this_state)
                self.total_steps += 1
                print('episode end', self.episode, 'final_reward', self.reward)
                self.init_dqn()

            else:
                # instaneous reward
                self.reward, self.action = self.get_reward(self.action)
                print('reward:', self.reward, 'action:', self.action, 'fund:', self.fund_pool, self.volumn)
            # print (self.fund_pool, self.volumn)
            # if a == 0:
            #     r = self.bid(0)
            # if a == 1:
            #     r = 0
            # if a == 2:
            #     r = self.ask(0)
            #
            # if a == 3:
            #     r = self.bid(1)
            # if a == 4:
            #     r = 0
            # if a == 5:
            #     r = self.ask(1)
            #
            # if a == 6:
            #     r = self.bid(2)
            # if a == 7:
            #     r = 0
            # if a == 8:
            #     r = self.ask(2)
            #
            # if a == 9:
            #     r = self.bid(3)
            # if a == 10:
            #     r = 0
            # if a == 11:
            #     r = self.ask(3)
            #
            # if a == 12:
            #     r = self.bid(4)
            # if a == 13:
            #     r = 0
            # if a == 14:
            #     r = self.ask(4)

    def get_final_reward(self):
        r = 0
        for i in range(5):
            if self.volumn[i] < 1000:
                r -= (1000 - self.volumn[i]) * self.price[i]
            elif self.volumn[i] > 1000:
                r += (self.volumn[i] - 1000) * self.price[i]
            else:
                r += 0
        return r

    def get_reward(self, a):
        r = 0
        if a in range(0, 15, 3):
            # buy
            r = self.bid(int(a / 3))
            if r == 0:
                a += 1

        elif a in range(1, 15, 3):
            # hold
            r = 0
        else:
            # sell
            r = self.ask(int((a - 2) / 3))
            if r == 0:
                a -= 1
        return r, a

    # buy 100 stocks
    def bid(self, i):
        bid = self.price[i] * NUM_OF_SHARES
        if bid <= self.fund_pool:
            self.fund_pool -= bid
            self.fund_pool -= TC * bid
            self.volumn[i] += NUM_OF_SHARES
            return (0 - bid - TC * bid)
        return 0
    # sell 100 stocks

    def ask(self, i):
        ask = self.price[i] * NUM_OF_SHARES
        if self.volumn[i] >= NUM_OF_SHARES:
            self.volumn[i] -= NUM_OF_SHARES
            self.fund_pool += ask
            self.fund_pool -= TC * ask
            return ask + TC * ask
        return 0

    def on_order_confirmed(self, order, src_status):
        # print(self.state.time_str,' 订单：%s 被确认' % (order.cid))
        code = order.symbol[2:]
        cid = order.cid

        # self.state.sslist[code].intraday_orderlist.setdefault(cid, order)

    def on_order_canceled(self, order, src_status):
        '''
        委托单被撤单成功
        '''
        print(order.cid + ' has been canceld', 'red', 0)
        oe = self.state.orderextra_dict_by_oid[order.cid]
        if oe.side_type == 1:  # 平仓单
            self.state.orderextra_dict_by_oid[oe.opposite_order_oid].pair_status = 1  # 对应的开仓单不再有平仓单

    def on_order_rejected(self, order, src_status):
        '''
        委托单被拒绝，开单失败
        '''
        print(order.cid + ' has been rejected', 'red', 0)

    def on_order_status_changed(self, order, src_status):
        pass

    def on_order_traded(self, order, src_status):
        # code = order.symbol[2:]
        cid = order.cid
        #self.state.sslist[code].intraday_orderlist.setdefault(cid, order)
        # oe = self.state.orderextra_dict_by_oid[order.cid]
        # diff_volume = order.filled_volume - self.state.orderextra_dict_by_oid[cid].order.filled_volume
        # diff_amount = order.filled_amount - self.state.orderextra_dict_by_oid[cid].order.filled_amount

        # if oe.side_type == 1: #平仓单

        #     self.state.orderextra_dict_by_oid[oe.opposite_order_oid].volume_reverse_filled = self.state.orderextra_dict_by_oid[oe.opposite_order_oid].volume_reverse_filled + diff_volume
        #     self.state.orderextra_dict_by_oid[oe.opposite_order_oid].amount_reverse_filled = self.state.orderextra_dict_by_oid[oe.opposite_order_oid].amount_reverse_filled + diff_amount

        # print(self.state.time_str,'订单：%s 有成交,量%s, 现总市值 %s' % (order.cid, diff_volume,self.account.total_value))

    def on_exerpt(self, rpt):
        cid = rpt.order_cid
        #self.state.sslist[code].intraday_orderlist.setdefault(cid, order)
        oe = self.state.orderextra_dict_by_oid[cid]

        if oe.side_type == 1:  # 平仓单
            if rpt.volume > 0:
                self.state.orderextra_dict_by_oid[oe.opposite_order_oid].volume_reverse_filled = self.state.orderextra_dict_by_oid[oe.opposite_order_oid].volume_reverse_filled + rpt.volume
                self.state.orderextra_dict_by_oid[oe.opposite_order_oid].amount_reverse_filled = self.state.orderextra_dict_by_oid[oe.opposite_order_oid].amount_reverse_filled + rpt.amount
            elif rpt.volume < 0:
                pass
                print('error')

        print(self.state.time_str, '订单：%s 类型%s 有成交,量%s, 现总市值 %s' % (cid, oe.side_type, rpt.volume, self.account.total_value))

        print('on_exerpt', rpt)

    def do_orders(self, tick):
        '''
        处理本品种的order
        '''

        code = tick['code']

        #name = self.state.fixedsslist[today_str][code].name
        str_trade_time = datetime.fromtimestamp(tick['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        preclose_backadj = tick['preclose_backadj']
        price = tick['price']
        upper_limit = tick['upper_limit']
        lower_limit = tick['lower_limit']
        symbol = tick['symbol']

        #print('entering single stock do_order iteration, orderextralist has %s orders'%(len(self.state.orderextra_list)))
        for key in list(self.state.sslist[code].intraday_oedict):
            order_extra = self.state.sslist[code].intraday_oedict[key]
            # need do nothing
            if order_extra.order.status == OrderStatus.WAITING_TO_ORDER or \
                    order_extra.order.status == OrderStatus.ORDER_PENDING or \
                    order_extra.order.status == OrderStatus.REJECTED or \
                    order_extra.order.status == OrderStatus.CANCELED:
                # order_extra.order.status == OrderStatus.REJECTED_T or  \

                continue

            # check if timeout: cancel order if lives over 60s
            if order_extra.order.status == OrderStatus.CONFIRMED or \
                    order_extra.order.status == OrderStatus.PART_FILLED:
                if self.state.env_timestamp - order_extra.order.create_time > 60:
                    self.cancel_order(order_extra.order.cid)
                    print(self.state.time_str, ' 订单：%s 超时，发出撤单' % (order_extra.order.cid))

            if order_extra.order.status == OrderStatus.CONFIRMED:
                continue

            # 处理开仓单
            if order_extra.side_type == 0:  # i am open_side order
                if order_extra.pair_status == 1:  # open
                    if (price * order_extra.order.filled_volume - order_extra.order.filled_amount) / order_extra.order.filled_amount > 0.02 or \
                            (price * order_extra.order.filled_volume - order_extra.order.filled_amount) / order_extra.order.filled_amount < -0.02:  # 盈利或止损平仓

                        if order_extra.order.status == OrderStatus.ALL_FILLED or order_extra.order.status == OrderStatus.PART_CANCELED:  # 全部成交，或已撤部分成交，可以平仓
                            # r_volume 要注意，= 原始开仓量 - 已经平掉的量
                            r_volume = order_extra.order.filled_volume - order_extra.volume_reverse_filled

                            if r_volume <= 0:  # 已平
                                order_extra.pair_status == 3
                                print('订单%s 已平' % (order_extra.order.cid))

                            if order_extra.order.order_side == OrderSide.BID:
                                r_orderside = OrderSide.ASK
                            elif order_extra.order.order_side == OrderSide.ASK:
                                r_orderside = OrderSide.BID
                            r_order_price = tick['price']
                            # 新建平仓订单
                            r_order = self.mk_order(
                                symbol=symbol,
                                order_side=r_orderside,
                                price=r_order_price,
                                order_type=OrderType.LIMIT,
                                volume=r_volume)
                            if r_order:
                                oe = OrderExtra(r_order, 1, order_extra.order.cid)
                                self.state.orderextra_dict_by_oid[r_order.cid] = oe
                                self.state.sslist[code].intraday_oedict.setdefault(r_order.cid, oe)
                                order_extra.pair_status = 2  # 该OrderPair的状态调整为2
                                print('平仓委托单被创建，订单号：' + r_order.cid)

                            else:
                                print('平仓订单创建失败， volume：%s' % (r_volume))
                                return
                        elif order_extra.order.status == OrderStatus.PART_FILLED:  # 部分成交，要先撤单，下一轮进来再平仓
                            self.cancel_order(order_extra.order.cid)
                            print(self.state.time_str, ' 订单：%s 部分成交，止盈。第一步先撤单' % (order_extra.order.cid))
                        else:
                            print('error in 订单状态超出处理范围 ')

                    else:  # 平仓条件没有达到
                        pass
            elif order_extra.side_type == 1:  # 我是平仓单 close_side order
                if order_extra.pair_status == 2:  # being closed
                    if (price * order_extra.order.filled_volume - order_extra.order.filled_amount) / order_extra.order.filled_amount > 0.02 or \
                            (price * order_extra.order.filled_volume - order_extra.order.filled_amount) / order_extra.order.filled_amount < -0.02:  # 平仓不利
                        self.cancel_order(order_extra.order.cid)
                        # todo：在收到撤单回执后，要把orderpair的状态改为1
                        print(self.state.time_str, ' 订单：%s 平仓不利，平仓单撤单。' % (order_extra.order.cid))
                    else:  # 无需动作
                        pass
            else:
                # error
                print('没有开仓或平仓标识的订单cid：', order_extra.order.cid)

    def trade_avg_price(self, tick):
        if not tick['status'] == 'PZ':
            return
        code = tick['code']
        trade_timestamp = tick['timestamp']
        today_str = time.strftime("%Y-%m-%d", time.localtime(trade_timestamp))
        name = self.state.fixedsslist[today_str][code].name
        str_trade_time = datetime.fromtimestamp(tick['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        preclose_backadj = tick['preclose_backadj']
        price = tick['price']
        upper_limit = tick['upper_limit']
        lower_limit = tick['lower_limit']
        symbol = tick['symbol']

        avg_price = self.state.sslist[code].avg_price

        # 这里加股票限制：只看已经持仓的代码

        account_volume = self.account.get_volume_of_symbol(symbol)
        #print(datetime.now().strftime('%H:%M:%S'),' after get volume')
        account_free_volume = self.account.get_free_volume_of_symbol(symbol)

        if self.state.num_trade > 20:
            print('交易次数达上限')
            return

        if price > avg_price:  # 下开仓多单

            if self.state.sslist[code].avg_price_status == 0 and price > preclose_backadj:
                # todo:需要考虑是否有持有足够的仓位卖掉？？
                # if self.account.balance / self.account.total_value < 0.25:  #只允许操作一半的cash
                #     print('%s 买入已达上限，跳过此次开仓买入' % tick['time'])
                #     return
                if self.account.get_volume_of_symbol(symbol) < 2 * self.init_positions_dict[symbol].volume - 100:  # 允许买入
                    volume = 100
                    order = self.mk_order(
                        symbol=symbol,
                        order_side=OrderSide.BID,
                        price=tick['price'] + 0.1,  # 加一毛钱防止不能成交
                        order_type=OrderType.LIMIT,
                        volume=volume)
                    if order:
                        print('开仓买入委托单被创建，订单号：' + order.cid)
                        oe = OrderExtra(order, 0)
                        self.state.orderextra_dict_by_oid[order.cid] = oe
                        self.state.sslist[code].intraday_oedict.setdefault(order.cid, oe)
                    else:
                        print('开仓买入订单创建失败')
                        return
            self.state.sslist[code].avg_price_status = 1
            # print(1,tick['time'])
        elif price < avg_price:  # 下开仓空单

            # todo:需要考虑是否有持有足够的仓位买回来？？
            if self.state.sslist[code].avg_price_status == 1 and price < preclose_backadj:
                if self.account.get_free_volume_of_symbol(symbol) > 100:  # 有仓位可卖
                    volume = 100
                    order = self.mk_order(
                        symbol=symbol,
                        order_side=OrderSide.ASK,
                        price=tick['price'] - 0.1,  # 加一毛钱防止不能成交
                        order_type=OrderType.LIMIT,
                        volume=volume)
                    if order:
                        oe = OrderExtra(order, 0)
                        self.state.orderextra_dict_by_oid[order.cid] = oe
                        self.state.sslist[code].intraday_oedict.setdefault(order.cid, oe)
                        print('开仓卖出委托单被创建，订单号：%s 卖出%s 100股。卖出前共有%s股，其中%s 可卖' % (order.cid, symbol, self.account.get_position(symbol), self.account.get_free_volume_of_symbol(symbol)))

                    else:
                        print('开仓卖出订单创建失败，account_free_volume:%s ,account_volume:%s' % (account_free_volume, account_volume))
                        return
            self.state.sslist[code].avg_price_status = 0
            # print(0,tick['time'])

    def validate_env_for_tick_30s(self, tick):
        code = tick['code']
        trade_timestamp = tick['timestamp']
        today_str = time.strftime("%Y-%m-%d", time.localtime(trade_timestamp))

        if code in self.state.evil_30s:  # 本股票代码已经不能运行了
            return False

        # 每个股票都先有StockStats
        if not (code in self.state.sslist.keys()):
            self.state.sslist[code] = StockStats(code)

        # 检查当日的固定属性文件是否已经读入
        if not (today_str in self.state.fixedsslist):
            print('validate_env_for_tick_30s:固定属性文件未读入,date=', today_str)
            return False

        # 检查本只股票固定属性是否已经初始化
        if not (code in self.state.fixedsslist[today_str]):
            self.state.evil_30s[code] = 'validate_env_for_tick_30s:固定属性未初始化,date=' + code
            print('validate_env_for_tick_30s:固定属性未初始化,code:', code)
            return False

        # 检查今天所有股票的统计数据文件是否已经读入
        if not (today_str in self.state.tradestatuslist):
            print('validate_env_for_tick_30s:统计数据文件未读入,date=', today_str)
            return False

        # 检查本只股票的统计数据是否已经初始化
        if not (code in self.state.tradestatuslist[today_str]):
            self.state.evil_30s[code] = 'validate_env_for_tick_30s:统计数据未初始化,date=' + code
            print('validate_env_for_tick_30s:统计数据未初始化,code:', code)
            return False

        if (not tick['bids']) and (not tick['asks']):  # 这是一个坏的tick，可能引起异常
            self.state.evil_30s[code] = 'validate_env_for_tick_30s:tick没有bids和asks信息,code:' + code
            print('tick validate:没有bids和asks信息,code:', code)
            return

        return True


if __name__ == '__main__':
    curdate_str = '2017-05-24'
    #sdk = api.Api('https://api.kuaiyudian.com/rpc')

    filename = 'data/' + '2017-05-24' + '.json'
    if not os.path.isfile(filename):
        sdk = api.Api('http://data-api.kuaiyutech.com/rpc2')
        re = sdk.get_dailybars(end='2017-05-24', num_days=1)

        lastdaybarlist = [x[0] for x in re]

        all_stock_yesterday = pd.DataFrame(lastdaybarlist)

        all_stock_yesterday['code'] = all_stock_yesterday['symbol'].str[2:]
        all_stock_yesterday['closePrice'] = all_stock_yesterday['closePrice'].astype(np.float64)

        all_stock_yesterday.to_json(filename, orient='records')
        print('网络请求完成:' + filename)
    all_stock_yesterday = pd.read_json(filename, orient='records', dtype=False)
    # print(all_stock_yesterday)

    # 给每只股票1000股的仓位
    tmp_balance = 0
    positions = []

    stock_list = []
    for symbol in all_stock_yesterday['symbol']:
        stock_list.append(symbol)
    # 随机获取5支股票
    sublist = random.sample(stock_list, 5)

    # sublist = ['sz300463','sz002737','sz300244','sz002405','sz300068']
    subdf = all_stock_yesterday[all_stock_yesterday['symbol'].isin(sublist)]

    i = 0
    for index, row in subdf.iterrows():
        pos = Position(symbol=row['symbol'], volume=1000, price=row['closePrice'])
        positions.append(pos)
        tmp_balance = tmp_balance + 1000 * row['closePrice']
        INIT_PRICE[i] = row['openPrice']
        i += 1
    print(INIT_PRICE)

    sub_str = ','.join(map(lambda x: 'tick_5s.' + x, sublist))
    # print(subdf)
    # print(sub_str)

    # 一半现金一半股票
    account = BackTestAccount(balance=tmp_balance, positions=positions)
    print('account 总值:', account.total_value, ' 其中股票市值:', tmp_balance)

    st = StrategyCY()
    #engine = Engine(server='127.0.0.1:3000',strategy=st,token='x8874545454545')
    # engine.plot_assets()
    engine = Engine(strategy=st)
    engine.run_rollback(subs=sub_str, account=account, start='2017-05-24', end='2017-05-24')
    #engine.run_realtime(subs='tick_30s.all', account=account)
