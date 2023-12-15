# Baca file csv
import pandas as pd
from IPython.display import display
from convert import *
# Baca file csv (Data Bully Dimas TI)
data = pd.read_csv('data.csv')
display(data)

# Preposessing text
import re
def preposessing(text):
    # Mengubah teks menjadi lowercase
    text = text.lower().split()
    text2 = []
    for kata in text:
        if kata.find("@") !=-1 or kata.find("http") !=-1 or kata.find("#") !=-1:
            continue
        else:
            text2.append(kata)
    text = ""
    for kata in text2:
        if kata !="":
            text = text + " " + kata
    # Menghapus karakter selain huruf dan angka
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Menghapus spasi yang berlebihan
    text = re.sub(r'\d+', '', text)
    # Menghapus spasi yang berlebihan
    text = re.sub(r'[^\w\s]', '', text)
    # Menghapus spasi ganda
    text = re.sub(r'\s+', ' ', text)
    # Menghapus spasi di awal dan di akhir kalimat
    text = text.strip()
    # Menghapus karakter aneh
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Menghapus angka
    text = re.sub(r"\d+", "", text)
    # Menghapus enter
    text = re.sub(r"\n", "", text)
    # Kalau misal kata terakhir BANGETTTTT T nya ambil 1
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

def split(text,list_stopwords):
    text = text.split()
    text2 = []
    for kata in text:
        if kata in convert.keys():
            text2.append(convert[kata])
        else:
            if kata in list_stopwords:
                continue
            text2.append(kata)
    return text2

import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords

list_stopwords = set(stopwords.words('indonesian'))
print(list_stopwords)
data['full_text'] = data['full_text'].apply(preposessing).apply(lambda x : split(x,list_stopwords))

# Drop kolom tweet_url
data = data.drop(['tweet_url'], axis=1)

# Analisis Deskriptif
# Jumlah Total Data: Hitung jumlah tweet yang ada dalam dataset
print("Jumlah Total Data: ", len(data))
# Distribusi Kelas: Identifikasi berapa banyak tweet yang terkait dengan bullying dan berapa yang tidak
def clean_label(kata):
    if kata.find("No"):
        kata = "NoBully"
    else:
        kata = "Bully"
    return kata

data['Label'] = data['Label'].apply(clean_label)
print("Distribusi Kelas: ")
print(data['Label'].value_counts())
print("")
# Statistik Kata: Identifikasi kata-kata paling umum yang muncul
from collections import Counter
import itertools

# Menghitung jumlah kata
#data['full_text'] = data['full_text'].apply(Counter)
## Menghitung jumlah kata yang unik
#data['full_text'] = data['full_text'].apply(dict)
word_count = dict()
for index in range(len(data.index)):
    for kata in data.iloc[index,0]:
        if kata not in word_count:
            word_count[kata] = 0
        word_count[kata]+=1
word_count_final = sorted(word_count.items(), key=lambda x:x[1],reverse=True)
print("List 5 kata paling umum:")

file = open("text_frequency.txt","w")
for i in range(len(data.index)):
    file.write(f"{word_count_final[i][0]} : {word_count_final[i][1]}\n")

for i in range(5):
    print(f"{word_count_final[i][0]} : {word_count_final[i][1]}")
# Simpan hasil preposessing ke file csv
data.to_csv('clean.csv', index=False)