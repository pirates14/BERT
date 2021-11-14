import sys, os
import requests
import selenium
from selenium import webdriver
import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pickle, progressbar, json, glob, time
from tqdm import tqdm
from urllib.request import Request, urlopen
import urllib3

urllib3.disable_warnings()

###### 날짜 저장 ##########
date = str(datetime.now())
date = date[:date.rfind(':')].replace(' ', '_')
date = date.replace(':', '시') + '분'
sleep_sec = 0.5

####### 언론사별 본문 위치 태그 파싱 함수 ###########
print('본문 크롤링에 필요한 함수를 로딩하고 있습니다...\n' + '-' * 100)


def crawling_main_text(url):
    print(url)
    if not url:
        return
    lists = []
    global text
    req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False)
    req.encoding = None
    soup = BeautifulSoup(req.content, 'html.parser')
    # 연합뉴스
    if ('://yna' in url) | ('app.yonhapnews' in url):
        main_article = soup.find('div', {'class': 'story-news article'})
        if main_article == None:
            main_article = soup.find('div', {'class': 'article-txt'})

        text = main_article.text

    # 경향신문
    elif 'khan' in url or 'oid=032' in url:
        error = ''
        text = None
        try:
            text = soup.find('div', {'id': 'articleBody'}).text
        except Exception as e:
            error = 'e'
            # return

        if not text:
            try:
                text = soup.find('div', {'id': 'container'}).text
            except Exception as e:
                error = 'e'
                # return
        if not text:
            try:
                text = soup.find('div', {'class': 'end_body_wrp'}).text
            except Exception as e:
                error = 'e'
                # return
        if error == 'e':
            return

    # 중앙일보
    elif 'joongang' in url or 'joins' in url:
        try:
            text = soup.find('div', {'id': 'article_body'}).text
            print(url)
        except Exception as e:
            return


    # 그 외
    else:
        text == None
    lists = text.split('.')
    return lists


press_nm = input('언론사 입력:(경향신문, 중앙일보)  :')

############### 브라우저를 켜고 검색 키워드 입력 ####################
query = input('검색할 키워드  : ')
news_num = int(input('수집 뉴스의 수(숫자만 입력) : '))

print('\n' + '=' * 100 + '\n')

print('브라우저를 실행시킵니다(자동 제어)\n')
chrome_path = 'C:/chromedriver/chromedriver.exe'
browser = webdriver.Chrome(chrome_path)

news_url = 'https://search.naver.com/search.naver?where=news&query={}'.format(query)
browser.get(news_url)
time.sleep(sleep_sec)

######### 언론사 선택 및 confirm #####################
print('설정한 언론사를 선택합니다.\n')

search_opn_btn = browser.find_element_by_xpath('//a[@class="btn_option _search_option_open_btn"]')
search_opn_btn.click()
time.sleep(sleep_sec)

bx_press = browser.find_element_by_xpath(
    '//div[@role="listbox" and @class="api_group_option_sort _search_option_detail_wrap"]//li[@class="bx press"]')

# 기준 두번 째(언론사 분류순) 클릭하고 오픈하기
press_tablist = bx_press.find_elements_by_xpath('.//div[@role="tablist" and @class="option"]/a')
press_tablist[1].click()
time.sleep(sleep_sec)

# 첫 번째 것(언론사 분류선택)
bx_group = bx_press.find_elements_by_xpath(
    './/div[@class="api_select_option type_group _category_select_layer"]/div[@class="select_wrap _root"]')[0]

press_kind_bx = bx_group.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[0]
press_kind_btn_list = press_kind_bx.find_elements_by_xpath('.//ul[@role="tablist" and @class="lst_item _ul"]/li/a')

for press_kind_btn in press_kind_btn_list:

    # 언론사 종류를 순차적으로 클릭(좌측)
    press_kind_btn.click()
    time.sleep(sleep_sec)

    # 언론사선택(우측)
    press_slct_bx = bx_group.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[1]
    # 언론사 선택할 수 있는 클릭 버튼
    press_slct_btn_list = press_slct_bx.find_elements_by_xpath('.//ul[@role="tablist" and @class="lst_item _ul"]/li/a')
    # 언론사 이름들 추출
    press_slct_btn_list_nm = [psl.text for psl in press_slct_btn_list]

    # 언론사 이름 : 언론사 클릭 버튼 인 딕셔너리 생성
    press_slct_btn_dict = dict(zip(press_slct_btn_list_nm, press_slct_btn_list))

    # 원하는 언론사가 해당 이름 안에 있는 경우
    # 1) 클릭하고
    # 2) 더이상 언론사분류선택 탐색 중지
    if press_nm in press_slct_btn_dict.keys():
        print('<{}> 카테고리에서 <{}>를 찾았으므로 탐색을 종료합니다'.format(press_kind_btn.text, press_nm))

        press_slct_btn_dict[press_nm].click()
        time.sleep(sleep_sec)

        break

################ 뉴스 크롤링 ########################

print('\n크롤링을 시작합니다.')
#####동적 제어로 페이지 넘어가며 크롤링

news_dict = {'text': []}
idx = 1
cur_page = 1

pbar = tqdm(total=news_num, leave=True)

while idx < news_num:

    table = browser.find_element_by_xpath('//ul[@class="list_news"]')
    li_list = table.find_elements_by_xpath('./li[contains(@id, "sp_nws")]')
    area_list = [li.find_element_by_xpath('.//div[@class="news_area"]') for li in li_list]
    a_list = [area.find_element_by_xpath('.//a[@class="news_tit"]') for area in area_list]

    for n in a_list[:min(len(a_list), news_num - idx + 1)]:
        lists = []
        n_url = ''
        n_url = n.get_attribute('href')
        if n_url == '':
            continue
        lists = crawling_main_text(n_url)
        idx += 1
        if not lists:
            continue
        else:
            for line in lists:
                if not line:
                    continue

                line.strip()
                line.replace('구글 지도화면 갈무리', '').replace('\n', '')
                line = ' '.join(line.split())
                if line.find(query) > 0:
                    news_dict['text'].append(line)
                    print(n.get_attribute('title'))
                    print(line)
                else:
                    pass

            pbar.update(1)

    if idx < news_num:
        cur_page += 1
        pages = browser.find_element_by_xpath('//div[@class="sc_page_inner"]')
        print(pages.find_elements_by_xpath('.//a'))
        check = [p for p in pages.find_elements_by_xpath('.//a') if p.text == str(cur_page)]
        if not check:
            pbar.close()
            print('\n더이상 기사가 없습니다.\n')
            time.sleep(0.7)
            browser.close()
            break
        next_page_url = [p for p in pages.find_elements_by_xpath('.//a') if p.text == str(cur_page)][0].get_attribute(
            'href')
        browser.get(next_page_url)
        time.sleep(sleep_sec)
    else:
        pbar.close()
        print('\n브라우저를 종료합니다.\n' + '=' * 100)
        time.sleep(0.7)
        browser.close()
        break

#### 데이터 전처리하기 ######################################################

print('데이터프레임 변환\n')

news_df = DataFrame(news_dict)

folder_path = os.getcwd()
xlsx_file_name = '{}뉴스_본문_{}개_{}_{}.xlsx'.format(press_nm, news_num, query, date)

news_df.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))

os.startfile(folder_path)

print('=' * 100 + '\n결과물의 일부')
news_df