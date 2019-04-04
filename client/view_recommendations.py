import requests
import webbrowser
import json

def get_recomendations(session_sequence):
    
    service_endpoint = "http://52.236.150.53:80/score"

    input_data = {"viewed_vehicles":session_sequence}
    input_data = json.dumps(input_data)
    
    headers = {'Content-Type':'application/json'}

    response = requests.post(service_endpoint, input_data, headers=headers)
    
    return response.json()

def open_results(results):
    for i in results[:10]:
        webbrowser.open_new_tab(i['page'])



clicks = ['ck12bce', 'bf17jya']

open_results(get_recomendations(clicks))
