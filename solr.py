import pandas as pd
import pysolr
from utils.parser import CovidParser

solr_instance = pysolr.Solr("http://localhost:8983/solr/cord", timeout=1600)

df = pd.read_csv('dataset-week1/metadata.csv', index_col=None)
docs=[]

for index, row in df.iterrows():
    print(index)
    document = CovidParser.parse(row)
    solr_instance.add([document], commit=False)

solr_instance.commit()
solr_instance.optimize()