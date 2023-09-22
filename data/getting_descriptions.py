import requests
from requests import utils
import rdflib
from bs4 import BeautifulSoup
import wikipediaapi as w

def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en', debug=False):
    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug: print(wikidata_id, url, json_response)

    entities = json_response.get('entities')
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:
                    # filter only the specified language
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return requests.utils.unquote(wiki_url)
                else:
                    # return all of the urls
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = requests.utils.unquote(wiki_url)
                    return wiki_urls
    return None

def get_description_from_wikipedia_url(wikipedia_url):
    description = ""
    if wikipedia_url != None:
        respond = requests.get(wikipedia_url)
        soup = BeautifulSoup(respond.text, features="lxml")

        # https://stackoverflow.com/questions/67605758/how-to-match-and-remove-wikipedia-refences-with-python-and-re
        for tag in soup.find_all(class_="reference"):
            tag.decompose()

        l = soup.find_all('p')

        description = BeautifulSoup(str(l[0]), features="lxml").get_text()
        if len(l) > 1:
            gfg = BeautifulSoup(str(l[1]), features="lxml")
            description += gfg.get_text()
            if len(l) > 2:
                gfg = BeautifulSoup(str(l[2]), features="lxml")
                description += gfg.get_text()

    description = description.replace('\n',"")
    return description

def get_label_from_wikidata_id(wikidata_id):
    g = rdflib.ConjunctiveGraph('SPARQLStore')
    g.open('https://query.wikidata.org/sparql')

    query="""
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT *
    WHERE {
    wd:""" + wikidata_id + """ rdfs:label ?label .
    FILTER (langMatches(lang(?label), "EN"))}
    LIMIT 1
    """
    label = None
    qresult = g.query(query)
    for row in qresult:
        label = row[0]
    return label

def get_description_from_label(label):
    description = ""
    wiki = w.Wikipedia('en')
    page = wiki.page(label)

    if page.exists():
        description = page.summary

    return description

def get_wikipedia_id_from_wikidata_url(wikipedia_url):
    response = requests.get(url=wikipedia_url)
    soup = BeautifulSoup(response.text, 'lxml')
    wikipedia_id = soup.find('li', {'id' : 't-wikibase'})
    wikipedia_id = wikipedia_id.a['href'].rsplit('/')[-1]
    return wikipedia_id

def get_description_from_wikiID(wikiID):
    respond = requests.get("https://www.wikidata.org/wiki/" + wikiID)
    soup = BeautifulSoup(respond.text, features="lxml")
    for tag in soup.find_all(class_="reference"):
        tag.decompose()
    if len(str(soup).split("name=\"description\"")) < 2:
        description = "" # No description
    else:
        description = str(soup).split("name=\"description\"")[1].split("<meta content=")[1].split("\"")[1]
    return description

def get_title_from_wikiID(wikiID):
    response = requests.get("https://www.wikidata.org/wiki/" + wikiID)
    soup = BeautifulSoup(response.text, 'lxml')
    title = response.text.split("<title>")[1].split(" -")[0]
    return title


file1 = open('WIKIDATA12k/entity2id.txt', 'r')
lines = file1.readlines()

with open('WIKIDATA12k_entity2name.txt','w') as file2:
    for line in lines:
        wikiID = line.split("\t")[0]
        title = get_title_from_wikiID(wikiID)
        file2.write(wikiID +"\t"+ title + "\n")

file3 = open('WIKIDATA12k/entity2id.txt', 'r')
lines = file3.readlines()

with open('WIKIDATA12k/entity2desc.txt','w') as file4:
    for line in lines:
        wikiID = line.split("\t")[0]

        # 1st method
        url = get_wikipedia_url_from_wikidata_id(wikiID)

        if url is not None:
            description = get_description_from_wikipedia_url(get_wikipedia_url_from_wikidata_id(wikiID))

        if url is None or (len(description) == 0 or len(description) == 1):
            # 2nd method
            description = get_description_from_wikiID(wikiID)

            #if len(description) == 0 or len(description) == 1:
                #print("The 2nd method failed for " + wikiID + "\n")

        file4.write(wikiID +"\t"+ description + "\n")

file5 = open('YAGO11k/entity2id.txt', 'r')
lines = file5.readlines()

def remove_underscore(text):
    return text.replace('_', ' ')

with open('YAGO11k_entity2name.txt','w') as file6:
    for line in lines:
        entity = line.split("\t")[0]
        new_entry = entity + "\t" + remove_underscore(entity)[1:-1] +"\n"
        file6.write(new_entry)

file7 = open('YAGO11k/entity2id.txt', 'r')
lines = file7.readlines()

needs_manual_extraction = {"Washington,_D.C.", "Russia", "Kanye_West"}

with open('YAGO11k/entity2desc.txt','w') as file8:
    for line in lines:
        entity = line.split("\t")[0]
        if entity[1:-1] in needs_manual_extraction:
            continue

        wikipedia_url = "https://en.wikipedia.org/wiki/{}".format(entity[1:-1])

        # 1st method
        description = get_description_from_wikipedia_url(wikipedia_url)

        if len(description) == 0 or len(description) == 1:
            print("The first method failed for " + entity + "\n")

            # 2nd method
            wiki = w.Wikipedia('en')
            page = wiki.page(entity[1:-1])

            if page.exists():
                description = page.summary
            else:
                print("The second method failed for " + entity + "\n")

                # 3rd method
                description = get_description_from_label(get_label_from_wikidata_id(get_wikipedia_id_from_wikidata_url(wikipedia_url)))

                if len(description) == 0 or len(description) == 1:
                    print("The third method failed for " + entity + "\n")

                    # 4th method
                    label = entity[1:-1].replace("_"," ")
                    wiki = w.Wikipedia('en')
                    page = wiki.page(label)
                    if page.exists():
                        description = page.summary
                    else:
                        print("The forth method failed for " + entity + "\n")

        file8.write(entity +"\t"+ description + "\n")
