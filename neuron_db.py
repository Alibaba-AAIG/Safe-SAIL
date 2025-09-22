import os
import json

def get_neurondb(neuron_dict_path="data/neuron_db/en_db.json"):
    with open(neuron_dict_path,'r') as file:
        neuron_db = json.load(file)
    return neuron_db

def get_display_table(neuron_db, layer, neuron_indices, values):
    records = []
    for neuron_id, value in zip(neuron_indices, values):
        neuron_name = f'{layer}_{neuron_id}'
        if neuron_name in neuron_db:
            explanation = neuron_db[neuron_name]['explanation']
            correlation_score = neuron_db[neuron_name]['correlation_score']
            records.append([neuron_name, value, explanation, correlation_score])
    return records