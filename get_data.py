import requests
import json
import os
import pandas as pd
import logging
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
import random


class SuperLottery:
    def __init__(self):
        self.SAVE_DIR = 'data/super_lottery'
        self.SAVE_FILE_NAME = 'super_lottery_total.xlsx'
        self.build_directory()
        self.build_logger()
        self.start_page = 1
        self.end_page = 28

    def build_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Building logger...")
        return

    def build_directory(self):
        if not os.path.exists(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)
        if not os.path.exists(os.path.join(self.SAVE_DIR, 'partitions')):
            os.mkdir(os.path.join(self.SAVE_DIR, 'partitions'))
        return

    def request_txt(self, url):
        r = requests.get(url)
        return r.text

    def parse_json(self, txt):
        return json.loads(txt)

    def deal_with_one_page(self, r_list):
        res = []
        for rl in r_list:
            lotteryDrawNum, lotteryDrawResult = rl['lotteryDrawNum'], rl['lotteryDrawResult']
            res.append([lotteryDrawNum, lotteryDrawResult])
        return res

    def save_partitions(self, one_page_results, page_num):
        df = pd.DataFrame(one_page_results, columns=['num', 'result'])
        partition_dir = os.path.join(self.SAVE_DIR, 'partitions')
        df.to_excel('{}/{}.xlsx'.format(partition_dir, page_num))
        self.logger.info("Finishing writing page {} into {}".format(page_num, partition_dir))
        return

    def fun(self):
        self.logger.info("Starting getting data...")
        for page_num in tqdm(range(self.start_page, self.end_page + 1)):
            url_pattern = 'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?gameNo=85&provinceId=0&pageSize=30&isVerify=1&pageNo={}&startTerm=16000&endTerm=21081'.format(
                page_num)
            r_text = self.request_txt(url_pattern)
            r_json = self.parse_json(r_text)
            r_json = r_json['value']['list']
            one_page_results = self.deal_with_one_page(r_json)
            self.save_partitions(one_page_results, page_num)
            time.sleep(random.uniform(1.1, 3.1))
        return

    def combine_partitions(self):
        add_parts = filter(lambda x: x.split('.')[1] == 'xlsx', os.listdir(os.path.join(self.SAVE_DIR, 'partitions')))
        add_parts = [os.path.join(self.SAVE_DIR, 'partitions', x) for x in add_parts]
        df = pd.DataFrame([], columns=['num', 'result'])
        for ap in add_parts:
            curr_df = pd.read_excel(ap)
            df = pd.concat([df, curr_df], axis=0)
        df.to_excel(os.path.join(self.SAVE_DIR, self.SAVE_FILE_NAME))
        return

    def history_fun(self):
        years_list = [y for y in range(2007, 2016)]
        df = pd.DataFrame([], columns=['num', 'result'])
        for year in tqdm(years_list):
            url = 'https://kjh.55128.cn/dlt-history-{}.htm'.format(year)
            year_txt = self.request_txt(url)
            soup = BeautifulSoup(year_txt, 'lxml')
            css = 'body > section.main > div.detail-wrapper > div.row > div.col-xs-12 > div.list > div.list-content > table > tbody > tr'
            rows_content = soup.select(css)
            rows = []
            for row in rows_content:
                row_soup = row.select('td')
                row_num = row_soup[1].text[2:]
                row_result = ' '.join([x.text.strip() for x in row_soup[2].select('span')])
                rows.append([row_num, row_result])
            curr_df = pd.DataFrame(rows, columns=['num', 'result'])
            df = pd.concat([df, curr_df], axis=0)
            self.logger.info('Finishing getting year {}'.format(year))
        df.to_excel(os.path.join(self.SAVE_DIR, 'year2007_2015.xlsx'))
        return


def main():
    sl = SuperLottery()
    # sl.fun()
    # sl.combine_partitions()
    sl.history_fun()


if __name__ == '__main__':
    main()
