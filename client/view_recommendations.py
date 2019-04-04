import requests
import webbrowser
import json
from lxml import html


def get_recomendations(session_sequence):
    
    service_endpoint = "http://52.236.150.53:80/score"

    input_data = {"viewed_vehicles":session_sequence}
    input_data = json.dumps(input_data)
    
    headers = {'Content-Type':'application/json'}

    response = requests.post(service_endpoint, input_data, headers=headers)
    
    return response.json()

def open_results(results):

        c=0

        for i in results:

                page = requests.get(i['page'])
                tree = html.fromstring(page.content)

                if tree.xpath('//title/text()')[0] != "Used Cars for Sale":
                        c+=1
                        webbrowser.open_new_tab(i['page'])

        return c

clicks = ["df16czj", "va65wpt", "lf16ypo", "ay14ygl", "ym08nha"]

c = open_results(get_recomendations(clicks))
print(c)
