#로컬에서는 이거 실행이 안돼요. 코랩에서 돌리는 코드입니다.
#!pip install konlpy

from konlpy.utils import pprint
from konlpy.tag import Okt
import pandas as pd
import csv

okt = Okt()
tokenset = []

# read TSV files
with open('/content/dataset - data_set_raw.tsv') as f:
  tr = csv.reader(f, delimiter='\t')
  for row in tr:
    token = okt.morphs(row[0])
    for i in token:
      tokenset.append(i)

print(tokenset)

with open('out.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(tokenset)
