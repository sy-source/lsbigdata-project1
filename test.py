
import numpy as np
import pandas as pd

df= pd.DataFrame({'제품'    :['사과', '딸기', '수박'],
                   '가격'    : [1800, 1500, 3000],
                    '판매량': [24, 38, 13]})
                    df
#과일의 가격 평균과 판매량 평균
sum(df['판매량']/3)
sum(df['가격']/3)

df

type(df)
type(df["name"])

sum(df["english"])/4

a = 10
b = 20
print("a==b", a == b)

