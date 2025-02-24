#Difference between v2 and v1:
# This one only searches for character tag, and only outputs a max of 200 results


from requests_html import HTMLSession
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import time
from nltk import tokenize

def get_works(search_tags, char):
    works = []
    for search_tag in search_tags[char]:
        #url = f'https://archiveofourown.org/works/search?work_search%5Bquery%5D=&work_search%5Btitle%5D=&work_search%5Bcreators%5D=&work_search%5Brevised_at%5D=&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bword_count%5D=&work_search%5Blanguage_id%5D=&work_search%5Bfandom_names%5D=&work_search%5Brating_ids%5D=&work_search%5Bcharacter_names%5D=&work_search%5Brelationship_names%5D={search_tag}&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=&work_search%5Bcomments_count%5D=&work_search%5Bbookmarks_count%5D=&work_search%5Bsort_column%5D=kudos_count&work_search%5Bsort_direction%5D=desc&commit=Search'
        url = f'https://archiveofourown.org/works/search?work_search%5Bquery%5D=&work_search%5Btitle%5D=&work_search%5Bcreators%5D=&work_search%5Brevised_at%5D=&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bword_count%5D=&work_search%5Blanguage_id%5D=&work_search%5Bfandom_names%5D=&work_search%5Brating_ids%5D=&work_search%5Bcharacter_names%5D={search_tag}&work_search%5Brelationship_names%5D=&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=&work_search%5Bcomments_count%5D=&work_search%5Bbookmarks_count%5D=&work_search%5Bsort_column%5D=kudos_count&work_search%5Bsort_direction%5D=desc&commit=Search'
        print("Searching for works with tag: " + search_tag)

        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        success = False
        retries = 0
        failure = False
        while not success and retries <= 5:
                try:
                    request_page = urlopen(req)
                    page_html = request_page.read()
                    request_page.close()
                    soup = BeautifulSoup(page_html, 'html.parser')
                    success = True
                except Exception as e:
                    if 'Retry-After' in e.headers:
                        wait = int(e.headers['Retry-After'])
                        success = False
                        print(f'Request limit reached.. Waiting {wait} seconds')
                        time.sleep(wait)
                    else:
                        wait = retries * 30
                        print(f'Error occured. Waiting {wait} secs and retrying.')
                        print(e)
                        time.sleep(wait)
                        success = False
                        retries += 1

        if not success and retries > 5:
            failure = True

        if not failure:
            results = soup.find('ol', class_='work index group').find_all('li', role='article')
            num_works = 0
            for result in results:
                new_url = 'https://archiveofourown.org' + result.a.get('href') + '?view_full_work=true&view_adult=true'
                if new_url not in works:
                    works.append(new_url)
                    num_works += 1
                if num_works >= 5:
                    break
                
            ao3_works = open(f'{char}.ao3_works.txt', 'w', encoding="utf-8")
            
            for work in works:
                ao3_works.write(work + '\n')

            ao3_works.close()
    return works

def scrape_works(works, char):
    PRINT_LIMIT = 189
    ao3 = open(f'../data/{char}_ao3.jsonl', 'w', encoding='utf-8')
    num_lines_printed = 0
    for url in works:
        if num_lines_printed < PRINT_LIMIT:
            print("Scraping work: " + url)
            success = False
            retries = 0
            failure = False
            while not success and retries <= 5:
                    
                    session = HTMLSession()
                    r = session.get(url)
                    if r.status_code == 200:
                        r.html.render(timeout=20)
                        soup = BeautifulSoup(r.content, 'html.parser')
                        success = True
                    elif r.status_code == 429:
                        if 'Retry-After' in r.headers:
                            wait = int(r.headers['Retry-After'])
                            success = False
                            print(f'Request limit reached.. Waiting {wait} seconds')
                            time.sleep(wait)
                        else:
                            wait = 300
                            print(f'Request limit reached, no Retry-After header was given. Waiting a default of {wait} seconds and retrying.')
                            time.sleep(wait)
                            success = False
                            retries += 1
                    else:
                        wait = retries * 30
                        print(f'Error occured. Waiting {wait} secs and retrying.')
                        print(r)
                        time.sleep(wait)
                        success = False
                        retries += 1

            if not success and retries > 5:
                failure = True

            if not failure:
                content = soup.find('div', id='chapters').find_all('p')
                for ptag in content:
                    if num_lines_printed < PRINT_LIMIT:
                        tokens = tokenize.sent_tokenize(ptag.get_text())
                        for token in tokens:
                            if char == 'emet':
                                if ('Emet-Selch' in token or 'Hades' in token or 'Emet' in token) and (len(token.split()) > 10):
                                    {"prompt":"","completion":" I am Emet-Selch."}
                                    output = token.strip().replace('\"', '\\"')
                                    ao3.write(f'{{"prompt":"", "completion":"{output}"}}\n')
                                    num_lines_printed += 1
                                    if num_lines_printed >= PRINT_LIMIT:
                                        break
                            elif char == 'hermione':
                                if ('Hermione' in token or 'Granger' in token) and (len(token.split()) > 10):
                                    output = token.strip().replace('\"', '\\"')
                                    ao3.write(f'{{"prompt":"", "completion":"{output}"}}\n')
                                    num_lines_printed += 1
                                    if num_lines_printed >= PRINT_LIMIT:
                                        break
                            elif char == 'gandalf':
                                if ('Gandalf' in token) and (len(token.split()) > 10):
                                    output = token.strip().replace('\"', '\\"')
                                    ao3.write(f'{{"prompt":"", "completion":"{output}"}}\n')
                                    num_lines_printed += 1
                                    if num_lines_printed >= PRINT_LIMIT:
                                        break
        

if __name__ == '__main__':
    search_tags = {
        'emet': [
            'Emet-Selch+%28Final+Fantasy+XIV%29'
        ],
        'hermione': [
            "Hermione+Granger"
        ],   
        'gandalf': [
            "Gandalf+%7C+Mithrandir"
        ]
        
    }
    
    char_input = input("""Run ao3 scraper on which character(s)?
    1: Emet-Selch
    2: Hermione Granger
    3: Gandalf
For multiple characters, separate with a comma (ex: Emet-Selch and Gandalf = 1,3)
""").split(',')

    if len(char_input) <= 0 or len(char_input) > 3:
        print("Incorrect input, try again.")
        exit()
    else:
        chars = []
        for element in char_input:
            if element.strip() == '1':
                chars.append('emet')
            elif element.strip() == '2':
                chars.append('hermione')
            elif element.strip() == '3':
                chars.append('gandalf')
            else:
                print("One of the inputted values was incorrect. Exiting..")
                exit()

    for char in chars:
        works = get_works(search_tags, char)
        scrape_works(works, char)
