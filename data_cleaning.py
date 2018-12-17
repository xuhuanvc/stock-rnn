import re
import time
import datetime
import requests
import csv
import sys

def find_crumb_store(lines):
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")

def get_page_data(ticker):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (ticker, ticker)
    r = requests.get(url)
    cookie = {'B': r.cookies['B']}
    lines = r.content.decode('unicode-escape').strip(). replace('}', '\n')
    return cookie, lines.split('\n')

def get_cookie_crumb(ticker):
    cookie, lines = get_page_data(ticker)
    crumb = (find_crumb_store(lines)).split(':')[2].strip('"')
    return cookie, crumb

def get_data(ticker, date_begin, date_end, cookie, crumb):
    csvname = '%s.csv' % (ticker)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (ticker, date_begin, date_end, crumb)
    response = requests.get(url, cookies=cookie)
    with open (csvname, 'wb') as csvfile:
        for writer in response.iter_content(1024):
            csvfile.write(writer)

def download_quotes(ticker):
    # our specific dates
    date_begin = int(datetime.datetime(1995, 4, 25).timestamp())
    date_end = int(datetime.datetime(2018, 4, 25).timestamp())

    # no need to check these two ugly things
    cookie, crumb = get_cookie_crumb(ticker)

    # main function to grab data based on ticker and our specific dates
    get_data(ticker, date_begin, date_end, cookie, crumb)

if __name__ == '__main__':
    try:
        download_quotes(sys.argv[1])
        print("%s Download Complete" % sys.argv[1])
    except:
        print("%s Download Error" % tickers[i])