import pandas as pd
import numpy as np

#데이터 탐색 함수
#head()
#tail()
#shape
#info
#describe()


exam = pd.read_csv("data/exam.csv")
exam.head(10)
exam.tail(10)
exam.shape

exam.describe()

type(exam)
var=[1,2,3]
type(var)
exam.head()


exam2 = exam.copy()
exam2=  exam2.rename(columns={"nclass" : "class"})
exam2

exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam.head()

exam2["test"] = np.where(exam2["total"] >= 200, "pass" , "fail")
exam2.head()

exam2["test"].value_counts()

import matplotlib.pyplot as plt
count_test=exam2["test"].value_counts()
count_test.plot.bar()
plt.show()
plt.clf()

df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
ax = df.plot.bar(x='lab', y='val')


exam2["test2"] = np.where(exam2["total"] >= 200, "A",
                np.where(exam2["total"] >= 100, "B", "C"))
exam2.head()


exam2["test2"].isin(["A", "C"])


import numpy as np

np.random.seed(2024)

a = np.random.choice(np.arange(1, 21), size=10, replace=False)
print(a)

#데이터 전처리 함수
#query()
#df[]
#sort_values()
#groupby()
#assign()
#agg()
#merge()
#concat()

exam = pd.read_csv("data/exam.csv")
exam.query("nclass==1")
# 조건에 맞는 행을 걸러내는 .query()
# exam[exam["nclass"] == 1]

exam.query("nclass==1")
exam.query("nclass1!=1")
exam.query("nclass!=3")
exam.query("math>50")
exam.query("math<50")
exam.query("english>=50")
exam.query("english<=80")
exam.query("nclass==1 & math >=50")
exam.query("nclass == 2 & english >= 80")
exam.query("math>=90 | enlgish >= 90 ")
exam.query("english<90 | science < 50")
exam.query("nclass == 1 | nclass == 3 | nclass == 5")
exam.query("nclass in [1, 3, 5]")


exam["nclass"]
exam[["id", "nclass"]]

exam.drop([columns='math'. 'english'])
exam.query("nclass == 1")[["math", "english"]]

exam2 = exam2.assign(
    total = exam["math"]) + exam["english"] + exam["science"], 
    mean = exam["math"]) + exam["english"] + exam["science"])/3)
    
    .sort_values("total", ascending = False)
exam2.head()


exam2 = pd.read_csv("data/exam.csv")

# 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \.agg(mean_math = ("math", "mean"))

exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"),
    sum_math = ("math, "sum""),
    median_math = ("math", "median"),
    n = ("nclass", "count"))

exam2.groupby("nclass").mean()


import pydataset

pydataset.data("mpg")


mpg. query ('category == "suv"') \
.assign(total = (mpg['hwy'] + mpg['cty']) / 2) \
.groupby ('manufacturer') \
.agg (mean_tot = ('total','mean')) \
.sort_values('mean_tot', ascending = False) \
.head ()


import pydataset

# pydataset을 사용하여 데이터 불러오기
mpg = pydataset.data("mpg")

# 'class' 열을 사용하여 'suv' 필터링
result = mpg.query('class == "suv"') \
            .assign(total=(mpg['hwy'] + mpg['cty']) / 2) \
            .groupby('manufacturer') \
            .agg(mean_tot=('total', 'mean')) \
            .sort_values('mean_tot', ascending=False) \
            .head()

print(result)

mpg.groupby(['manufacturer', 'drv']) \
.agg(mean_cty=('cty', 'mean'))

##############################################


test1 = pd.DataFrame({'id': [1,2,3,4,5]
                    'midterm' : [60,80,70,90,85]})
                    
test2 = pd.DataFrame({'id' : [1,2,3,4,5]
                    'final' : [70,83,65,95,80]})

total = pd.merge(test1, test2, how = 'left', on='id')
total
