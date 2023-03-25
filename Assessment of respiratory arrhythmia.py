#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics
import seaborn as sns
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.stats import zscore


# In[2]:


KATALOG_PROJEKTU = os.path.join(os.getcwd(),"rytm_serca")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"dane")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
os.makedirs(KATALOG_DANYCH, exist_ok=True)


# In[3]:


SKAD_POBIERAC = ['./healthy_decades/', './HTX_LVH/',  './hypertension_RR_SBP/',
                 './hypertension/']
czytamy = SKAD_POBIERAC[3]
pliki = os.listdir('./hypertension')
pomin=5
kolumny = ['time','R_peak','Resp','SBP','cus']
df = pd.DataFrame(columns = ['plik', 'SDNN', 'RMSSD', 'pNN50', 'pNN20', 'Piki R Wdech', 'Piki R Wydech', 'przyspieszenie wdech', 
                             'przyspieszenie wydech', 'zwolnienie wdech', 'zwolnienie wydech', 'Średnie SBP', 'SBP spadek wdech', 
                             'SBP wzrost wdech', 'SBP spadek wydech', 'SBP wzrost wydech'])


# In[4]:


def load_serie(skad , co, ile_pomin = 0, kolumny =['RR_interval', 'Num_contraction']):
    csv_path = os.path.join(skad, co )
    seria = pd.read_csv(csv_path, sep='\t', header =None,
                        skiprows= ile_pomin, names= kolumny, low_memory=False)
    if skad == SKAD_POBIERAC[2]:
        seria = pd.read_csv(csv_path, sep='\t',  decimal=',' )
    return seria


# # Ładowanie danych

# In[5]:


dataframes_list = []
for i in pliki[1:len(pliki)]:
    seria = load_serie(skad = czytamy, co = i, ile_pomin= pomin, kolumny=kolumny)
    
    seria.time = pd.to_numeric(seria.time, errors='coerce')
    seria.Resp = pd.to_numeric(seria.Resp, errors='coerce')
    seria.SBP = pd.to_numeric(seria.SBP, errors='coerce')
    seria.R_peak = pd.to_numeric(seria.R_peak, errors='coerce')
    
    seria = seria[seria['time'].notnull()]
    seria = seria[seria['R_peak'].notnull()]
    seria = seria[seria['Resp'].notnull()]
    
    dataframes_list.append(seria)


# In[6]:


dataframes_list[1]


# # Wykres oddechu w czasie z R_peak

# In[7]:


seria = dataframes_list[2]
seria = seria.head(100000)
df2 = seria[seria['SBP'] > 0]

fig, ax = plt.subplots(figsize =(20, 10))
plt.plot_date(x = seria["time"], y = seria["Resp"], fmt="-", color = "green")
ax.scatter(df2['time'], df2['Resp'], color = 'blue', s =20, zorder = 10)


# # Pozbycie się "przerw" w rejestracji danych

# In[8]:


for i in range(0, len(dataframes_list)):
    time_list = [0 + x * 0.001 for x in range(len(dataframes_list[i]))]
    dataframes_list[i]["time"] = time_list


# # Sprawdzenie poprawności danych

# In[9]:


for i in range(0, len(dataframes_list)):
    seria = dataframes_list[i]
    print(seria['R_peak'].unique())


# In[10]:


for i in range(0, len(dataframes_list)):
    dataframes_list[i][(dataframes_list[i]["R_peak"] != 0) & (dataframes_list[i]["R_peak"] != 1)] = 0


# In[11]:


for i in range(0, len(dataframes_list)):
    seria = dataframes_list[i]
    print(seria['R_peak'].unique())


# # Wykres oddechu w czasie w krótszym czasie wraz z Rpeak

# In[66]:


seria1 = dataframes_list[5]
seria1 = seria1.head(300000)
df21 = seria1[seria1['R_peak'] == 1]

fig, ax = plt.subplots(figsize =(20, 10))
plt.plot_date(x = seria1["time"], y = seria1["Resp"], fmt="-", color = "green")
ax.scatter(df21['time'], df21['Resp'], color = 'blue', s =20, zorder = 10)


# # Właściwa pętla

# In[17]:


k = 0
df = pd.DataFrame(columns = ['plik', 'SDNN', 'RMSSD', 'pNN50', 'pNN20', 'Piki R Wdech', 'Piki R Wydech', 'przyspieszenie wdech', 
                             'przyspieszenie wydech', 'zwolnienie wdech', 'zwolnienie wydech'])
for i in range(0, len(pliki)-1):
    k =+ 1
    seria = dataframes_list[i]
    
    df_headers = list(seria.columns)
    for k in df_headers[6:]:
        seria[k] = pd.to_numeric(seria[k], errors='coerce')
        
    # Wygładzenie oddechu z wykorzystaniem średniej kroczącej
    window_size = 1000                                             # mówi ile obserwacji bierzemy do naszej średniej
    numbers_series = pd.Series(seria["Resp"])
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    seria = seria.iloc[window_size-1: , :]
    final_list = pd.Series(final_list)
    seria.loc[:, "Resp_sm"] = final_list.values
    
    #Wskaźnik RR i NN 
    RR = pd.DataFrame(seria[seria['R_peak'] == 1].time.diff())     # rożnica pomiędzy następującymi po sobie eleentami
    RR = RR[1:]
    RR[np.abs(zscore(RR["time"])) > 2] = np.median(RR)
    RR['RRt1'] =  RR.time[1:]
    RR = RR[1:]
    RR = RR.assign(RRt1 = RR.RRt1.shift(-1)).drop(RR.index[-1])    # przesunięcie kolmuny RRt
    RR = RR.rename(columns = {'time' : 'RRt'})
    RR["NN"] = np.abs(RR["RRt"] - RR["RRt1"])
    
    #SDNN
    SDNN = np.std(RR['RRt'])
    
    # RMSSD
    RR['RRt^2'] = RR['RRt']**2
    RMSSD = (sum(RR['RRt^2']) / (len(RR['RRt^2']) - 1))**(1/2)
    
    # PNN50/20
    pNN50 = len(RR[RR["NN"] >= 0.05]) / len(RR)
    pNN20 = len(RR[RR["NN"] >= 0.02]) / len(RR)
    
    # średnie ciśnienie skurczowe
    mean_SBP = seria[seria['SBP'] > 0].SBP.mean()
    
    # Liczba wzrostów i spadków ciśnienia skurczowego SBP przy wdechu i wydechu
    SBP_diff = pd.DataFrame(seria[seria['SBP'] > 0].SBP.diff())     # rożnica pomiędzy następującymi po sobie eleentami
    SBP_diff["Resp"] = seria[seria['SBP'] > 0].Resp_sm
    
    # Liczba przypadków wzrostu i spadku ciśnienia w podziale na wdech i wydech
    SBP_spadek_wdech = len(SBP_diff[(SBP_diff['SBP'] < 0) & (SBP_diff['Resp'] >= 0)]) / len(SBP_diff)
    SBP_spadek_wydech = len(SBP_diff[(SBP_diff['SBP'] < 0) & (SBP_diff['Resp'] < 0)]) / len(SBP_diff)
    SBP_wzrost_wdech = len(SBP_diff[(SBP_diff['SBP'] >= 0) & (SBP_diff['Resp'] >= 0)]) / len(SBP_diff)
    SBP_wzrost_wydech = len(SBP_diff[(SBP_diff['SBP'] >= 0) & (SBP_diff['Resp'] < 0)]) / len(SBP_diff)
    
    # ilość wystąpień pików R przy wdechu i wydechy
    ileR_wdech = seria[seria['Resp_sm'] >= 0 ].R_peak.sum() / seria.R_peak.sum()
    ileR_wydech = seria[seria['Resp_sm'] < 0 ].R_peak.sum() / seria.R_peak.sum()
    
    
    # Ilość przyspieszeń (a) i zwolnień (d) rytmu serca przy wdechu i wydechu
    RR2 = seria[seria['R_peak'] > 0]
    RR3 = pd.DataFrame(RR2.time.diff())
    RR3['RRt1'] =  RR3.time[1:]
    RR3 = RR3[1:]
    RR3 = RR3.assign(RRt1 = RR3.RRt1.shift(-1)).drop(RR3.index[-1])              #przesunięcie komuny RRt
    RR3 = RR3.rename(columns = {'time' : 'RRt'})
    RR3['Resp_sm'] = RR2.Resp[1:]
    
    a = RR3[RR3['RRt1'] - RR3['RRt'] < 0 ]
    d = RR3[RR3['RRt1'] - RR3['RRt'] > 0 ]
    a_wdech = len(a[a['Resp_sm'] > 0]) / len(a)
    a_wydech = len(a[a['Resp_sm'] < 0]) / len(a)
    d_wdech = len(d[d['Resp_sm'] > 0]) / len(d)
    d_wydech = len(d[d['Resp_sm'] < 0]) / len(d)
    
    
    dane_zebrane = {'plik': pliki[i + 1], 'SDNN':SDNN, 'RMSSD':RMSSD, 'pNN50':pNN50, 'pNN20':pNN20, 
                    'Średnie SBP':mean_SBP, 'Piki R Wdech':ileR_wdech, 'Piki R Wydech':ileR_wydech,
                    'przyspieszenie wdech':a_wdech, 'przyspieszenie wydech':a_wydech, 'zwolnienie wdech':d_wdech, 
                    'zwolnienie wydech':d_wydech, 'SBP spadek wdech':SBP_spadek_wdech , 'SBP wzrost wdech':SBP_wzrost_wdech,
                    'SBP spadek wydech': SBP_spadek_wydech, 'SBP wzrost wydech': SBP_wzrost_wydech}
    
    df = df.append(dane_zebrane, ignore_index=True)
    


# In[18]:


for i in df_headers[6:]:
    df[i] = pd.to_numeric(df[i], errors='coerce')


# Dodanie informacji czy pacjent jest zdrowy czy nie

# In[19]:


df["Stan Zdrowia"] = np.where(df.plik.str.slice(start = 4, stop = 5) == " ", 
                              df.plik.str.slice(start = 5, stop = 6), df.plik.str.slice(start = 4, stop = 5)) 


# Wyświetlenie tabeli

# In[20]:


df


# In[67]:


df.to_excel("df.xlsx")     #Zapisz jako excel


# # CZĘŚĆ II PROJEKTU

# **********************************************************************************************************

# In[21]:


from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# # Standaryzacja danych

# In[22]:


df_stand = df


# In[23]:


df_stand.head()


# In[24]:


standar = np.array(df_stand.iloc[:, [1, 2, 3, 4, 15]], dtype=np.float64) #bierzemy tylko kolumny, które zawierają wartości nie procentowe


# In[25]:


np.mean(standar, axis=0)


# In[26]:


np.std(standar, axis=0)


# In[27]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(standar)
k = scaler.transform(df_stand.iloc[:, [1, 2, 3, 4, 15]])
k


# Jak widać po transofrmacji średnia dla poszczególnych kolumn wynosi 0, natomiast odchylenie standardowe równa się 1

# In[28]:


np.mean(k, axis=0)


# In[29]:


np.std(k, axis=0)


# Teraz należy przekształcić tabele na odpowiednią formę wejściową

# In[30]:


df_stand = pd.DataFrame(k, columns = ['SDNN', 'RMSSD', 'pNN50', 'pNN20', 'Średnie SBP'])

labels = ['Piki R Wdech', 'Piki R Wydech', 'przyspieszenie wdech', 'przyspieszenie wydech', 'zwolnienie wdech', 
          'zwolnienie wydech', 'SBP spadek wdech', 'SBP wzrost wdech', 'SBP spadek wydech', 'SBP wzrost wydech','Stan Zdrowia']

df_stand.insert(0, 'plik', list(df["plik"]))

for i in range(0, len(labels)):
    df_stand.insert(i+6, labels[i], list(df[labels[i]]))


# In[31]:


df_stand.head()


# # Podział danych  (Cross-validation)

# Kolejnym krokiem jest podział danych na dwie grupy: treningową, testową. Dzięki temu można wybrać najlepszy model klasyfikatora za pomocą walidacji krzyżowej i ocenić jego dokładność na zbiorze testowym.

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(df_stand.iloc[:, 1:16], df_stand.iloc[:,16], test_size=0.2, random_state=42)


# In[33]:


X_train.head()


# # PCA

# W naszym zbiorze danych występuje aż 15 zmiennych niezależnych. Część z nich w dużym stopniu jest na pewno ze sobą skorelowana, co nie potrzebnie sprawia, że nasz model może być przetrenowany. Należy użyć metody PCA, aby zredukować liczbę tych zmiennych, jako sposób na określenie liczby komponentów wybieram metodę minimalnego poziomu wariancji danych ustalony na 97%

# In[34]:


pca = PCA()
pca_Xtrain = pca.fit_transform(X_train)
pca_Xtest = pca.transform(X_test)


# In[35]:


per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1) 
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]


# In[36]:


fig, ax = plt.subplots(figsize =(20, 10))
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Procent wyjaśnionej wariancji')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# Na podstawie powyższego wykresu można stwierdzić, że optymalna liczba nowych zmiennych (komponentów) powinna być określona na poziomie 7. W celu dokłądniejszego określenia liczby zmiennych poniżej zostały zaprezentowane dodatkowa analiza:

# In[37]:


per_var_cumulated = per_var
for i in range(1, len(per_var_cumulated)):
    per_var_cumulated[i] = per_var_cumulated[i] + per_var_cumulated[i - 1]
per_var_cumulated


# In[38]:


fig, ax = plt.subplots(figsize =(20, 10))
plt.bar(x = range(1, len(per_var) + 1), height = per_var_cumulated, tick_label = labels)
plt.ylabel('Procent wyjaśnionej wariancji')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# Liczba zmiennych PCA poprzez analizę skumulowanej wariancji została określona na poziomie 5. Również ze zbiory per_var_cumulated możemy określić, że właśnie dla 5 komponentów poziom wariancji danych przekracza 97%. Warto również zobaczyć jak wyglądałby nasz model dla liczby parametrów 4. Dlatego też do dalszej części projektu przygotuję 2 zestawy danych, dla 4 i 5 komponentów. 

# #### Pięć komponentów

# In[39]:


pca = PCA(n_components = 5)
pca_Xtrain5 = pca.fit_transform(X_train)
pca_Xtest5 = pca.transform(X_test)


# In[40]:


pca_Xtrain5 = pd.DataFrame(pca_Xtrain5, columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'])
pca_Xtest5 = pd.DataFrame(pca_Xtest5, columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'])
pca_Xtrain5.head()


# #### Cztery komponenty

# In[41]:


pca = PCA(n_components = 4)
pca_Xtrain4 = pca.fit_transform(X_train)
pca_Xtest4 = pca.transform(X_test)


# In[42]:


pca_Xtrain4 = pd.DataFrame(pca_Xtrain4, columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4'])
pca_Xtest4 = pd.DataFrame(pca_Xtest4, columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4'])
pca_Xtrain4.head()


# Ostatecznie otrzymaliśmy tabele ze zredukowaną liczbą kolumn, pozwala nam to przejść do kolejnego etapu, a więc tworzenia klasyfikatora.

# # Model Klasyfikatora

# Na tym etapie skorzystam z różnych metod klasyfikacji uczeniem maszynowego, postaram się każdy z nich zoptymalizować tak aby w jak najlepszym stopniu klasyfikator ten były w stanie rozpoznać osobę chorą na arytmię oddechową oraz zdrową. W tym celu wykorzystam funkcję Stochastic Gradient Descent, dzięki której będę w stanie stworzyć optymalny model, który będzie w stanie przewidywać na podstawie inputów z pca, czy dany pacjent jst chory czy nie. Część tę podzieliłem na 2 etapy, czyli tyle ile mamy zbiorów danych. Wykorzystam również funkcję GridSearchCV, dzięki której jestem w stanie otrzymać optymalne parametry dla mojego modelu. Dodatkowo wykorzystam 2 podziały, z PCA dla 4 i 5 komponentów.

# ## PCA = 5 komponentów

# ### 1 etap - na podstawie zbiory trenującego (pca_Xtrain5)

# Na tym etapie dojdzie do strojenia hiperparametrów na podstawie zbioru trenującego, z wykorzystaniem metody kross-walidacji, dlatego też w funkcji GridSearchCV, za cv podstawiłem 5, co oznacza że w naszym zbiorze danych dojdzie do 5-krotnej walidacji krzyżowej.

# In[43]:


param_grid = {'alpha': list(np.arange(0.01, 10, 0.01)), 'loss': ['hinge', 'modified_huber', 'log'], 'penalty': ['l2', 'l1', 'elasticnet']}


# W pierwszej kolejności stworzyłem słownik zawierający różne wartości poszczególnych parametrów funkcji SGDClassifier

# In[44]:


sgd = SGDClassifier()


# In[45]:


grid_search5 = GridSearchCV(sgd, param_grid, cv=5)


# In[46]:


grid_search5.fit(pca_Xtrain5, y_train)


# In[48]:


print(grid_search5.best_params_)   #parametry dla których model jest najlepiej dopasowany


# In[49]:


sgd5 = SGDClassifier(alpha = 0.27, loss = 'log', penalty = 'l2')
sgd5.fit(pca_Xtrain5, y_train)


# ### 2 etap - na podstawie zbioru testowego (pca_Xtest5)

# In[50]:


prediction5 = sgd5.predict(pca_Xtest5)
prediction5 == y_test


# Jak widać stosunek wartości poprawnie do źle sklasyfikowanych wynosi 7:6, jest to liczba, która jest minimalnej większa od 0,5, co jest watością losową. Można więc stwierdzić, że nasz model nie jest w stanie przeiwidzieć, który pacjent jest chory, ponieważ jego poprawność jest na dość niskiem poziomie

# In[51]:


confusion_matrix(y_test, prediction5)


# Powyżej znajduje się macierz błedów, jak widać: true positive = 0, true negatives = 7, false positive = 2 oraz false negatives = 4

# ## PCA - 4 komponenty

# ### 1 etap - na podstawie zbiory trenującego (pca_Xtrain5)

# In[68]:


sgd4 = SGDClassifier()
grid_search4 = GridSearchCV(sgd4, param_grid, cv=5)


# In[69]:


grid_search4.fit(pca_Xtrain4, y_train)


# In[70]:


print(grid_search4.best_params_)


# In[71]:


sgd4 = SGDClassifier(alpha = 0.27, loss = 'log', penalty = 'l2')
sgd4.fit(pca_Xtrain4, y_train)


# ### 2 etap - na podstawie zbioru testowego (pca_Xtest5)

# In[72]:


prediction4 = sgd4.predict(pca_Xtest4)
prediction4 == y_test


# Tak jak w poprzednim przykładzie stosunek wartości poprawnie do źle sklasyfikowanych wynosi 7:6. Można więc stwierdzić, że nasz model również dla danych dla 4 komponentów nie jest w stanie przewidzieć, który pacjent jest chory, ponieważ jego poprawność jest na dość niskiem poziomie

# In[73]:


confusion_matrix(y_test, prediction4)


# Macierz błędu w tym przypadku wygląda identycznie jak w poprzednim przykładzie, można więc stwierdzić, że różnice dla danych PCA nie są istotne. 

# In[74]:


accuracy_score(y_test, prediction4)


# Dokłądnośćprognoz na poziomie około 0,54.

# In[75]:


len(X_train)


# # Podsumowanie

# Podsumowując, w drugiej częsci projektu zasosowałęm różne metody, które miały doprowadzić mnie do modelu, który będzie klasyfikował pacjentów na zdrowych oraz chorych na arytmię oddechową. W poprzedniej częsci, po poprawkach udało mi się otrzymać tabelę dla wszystkich pacjentów z różnymi wskaźnikami odnośnie ich zdrowia, jak np. Pnn50 czy RMSSD. Aby można było porównać dane na początku tego projektu zasosowałem standaryzację danych w oparciu o funkcję StandardScaler(), która według literatury jest najbardziej odpowiednią funkcją jeżeli chcemy w dalszej częsci zastosować PCA. Następnie wykorzystałem metodę PCA w celu redukcji danych, za przyjęty poziom przy wyborze liczby komponentów wybrałem 97% zmienności wariancji. Poziom ten odpowiadał pięciu komponentom, ale również w celu dodatkowej analizy stworzyłem drugi zbiór danych dla czterech komponentów. W trzecim kroku podzieliłem dane na część testową oraz treningową. Ostatni etap polegał na dopasowaniu modelu do danych treningowych oraz na sprawdzeniu poprawności prognoz w oparciu o dane testowe. W tym celu wykorzystałem funkcję SGDClassifier(), która klasyfikuje dane wartości, w oparciu o przyjęte parametry. Aby nasze prognozy były jak najbardziej trafne wytrenowałem model w oparciu o metodę kross walidacyjną z wykorzystaniem funkcji GridSearchCV, która dla zadanych parametrów wyszukuje najbardziej optymalnego modelu. 
# Na końcu sprawdziłem jak dopasowany model działa na danych testowych i okazało się, że skuteczność jego prognoz jest na bardzo niskim poziomie. Dokładność jego prognoz szacowała się na poziomie około 0,54. Również podział na dwie grupy PCA nie przynósł dodatkowych różnic.
# Przyczyną tak słabego wyniku mogą być błędy na etapie pre-processingu, a więc błędy w czyszczeniu danych. Również problemem może być mały zakres danych, w zbiorze testowym zawierało się tylko 49 pacjentów, możliwe, że jest to liczba zbyt mała, aby poprawnie wytrenować nasz model. Również możliwe jest, że parametry, które dobraliśmy nie są najbardziej optymalne. W celu lepszego sprawdzenia, należałoby w naszej siatce uwzględnić więcej możliwości. Taki krok oczywiście doprowadziłby do dłuższej pracy kodu. 

# In[ ]:




