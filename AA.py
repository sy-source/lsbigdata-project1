import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.shape
mpg

import seaborn as sns
#!pip install seaborn
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4)) #그래프 사이즈 (짤려 보일 때)

sns.scatterplot(data=mpg, x="displ", y="hwy")
plt.show()
plt.clf()

sns.scatterplot(data=mpg, x="displ", y="hwy") \
    .set(xlim = [3,6], ylim=[10, 30])
plt.show()

sns.scatterplot(data=mpg, x="displ", y="hwy", hue = 'drv') \
 .set(xlim = [3,6], ylim=[10, 30])
 plt.show()
 
 #막대그래프
 
mpg.groupby("drv") \
    .agg(mean_hwy=('hwy', 'mean'))


mpg["drv"].unique()
 
 
mpg.groupby("drv", as_index=False) \
    .agg(mean_hwy=('hwy', 'mean'))
 
df_mpg = mpg.groupby("drv", as_index=False) \
    .agg(mean_hwy=('hwy', 'mean'))
df_mpg    

plt.clf()
sns.barplot(data=df_mpg.sort_values("mean_hwy", ascending = False), 
            x = "drv", y = "mean_hwy", hue = "drv")
plt.show()

df_mpg = mpg.groupby("drv", as_index = False) \
                    .agg(n = ("drv", "count"))
df_mpg

sns.barplot(data=df_mpg, x = 'drv', y= 'n')
plt.show()

sns.countplot(data=mpg, x= "drv")
plt.show()

import numpy as np
np.arange(33).sum()/33

np.unique((np.arange(33)-16)**2)


np.arrane(33).sum()/33
np.unique((np.arange(33)-16)**2) * (2/33)
sum(np.unique((np.arange(33)-16)**2) * (2/33))

x= np.arange(33)
sum(x)/33
sum((x-16)* 1/33)
(x-16)**2

np.unique((np.arange(33)-16)**2) * (2/33)
sum(np.unique((np.arange(33)-16)**2) * (2/33))

# E(X^2)
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X])
sum(x**2 * (1/33)) - 16**2


import numpy as np

# X의 값과 각각의 확률
values = np.array([0, 1, 2, 3])
probabilities = np.array([1/6, 2/6, 2/6, 1/6])

# 기대값 E(X) 계산
expected_value = np.sum(values * probabilities)

# E(X^2) 계산
expected_value_squared = np.sum(values**2 * probabilities)

# 분산 Var(X) 계산
variance = expected_value_squared - expected_value**2

x= np.arange(4)
x

pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

Ex=sum(x*pro_x)
Exx=sum(x**2 * pro_x)

Exx - Ex**2

x= np.arange(33)
sum(x)/33
sum((x-16)* 1/33)
(x-16)**2

np.unique((np.arange(33)-16)**2) * (2/33)
sum(np.unique((np.arange(33)-16)**2) * (2/33))

#연습문제

x= np.arange(99)
x

np.arange(1,51)
np.arange(49,0, -1)

# 1-50, 50-1 벡터
pro_x=np.concatenate((np.arange(1,51),np.arange(49,0,-1)))
pro_x=pro_x/2500                                                                                                
pro_x

Ex=sum(x*pro_x)
Exx=sum(x**2 * pro_x)

Exx = Ex**2

sum((x=Ex**2))

#연습 3

x=np.arange(4)*2
x

9.52**2 / 25
np.sqrt(9.52**2 / 25)


np.sqrt(
    
9.52**2 / 10
)








!pip install scipy.stats

# 확률질량함수(pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)

from scipy.stats import bernoulli
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)


# 이항분포 P(X = | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
from scipy.stats import binom
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)


import math
num_a= math.factorial(54)
num_b= math.factorial(26)
num_c = math.factorial(28)

result= num_a / (num_b * num_c)
result

math.comb(54, 26) # 한번에 해결완료..

#np.cumprod(np.arange(1,5))[-1]
#

log(a *b) = log(a) + log(b)
log(1 *2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

math.log(24)

math.log(math.factorial(54))
logf_54 = sum(np.log(np.arange(1, 55)))
logf_26 = sum(np.log(np.arange(1, 27)))
logf_28 = sum(np.log(np.arange(1, 29)))

np.exp(logf_54 - (logf_26 + logf_28))



nu_a= math.factorial(3)
nu_b= math.factorial(0)
nu_c = math.factorial(28)


math.comb(2,0) * 0.3**0 * (1-0.3) **2
math.comb(2,1) * 0.3**1 * (1-0.3) ** 1
math.comb(2,2) * 0.3**2 * (1-0.3) ** 0


#pmf :probability mass function (확률)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B (n=10, p=0.36)
# P(X = 4) = ?
binom.pmf(4, n=10, p=0.36)


binom.pmf(np.arange(5), n=10, p=0.36).sum()
np.arange(5)

binom.pmf(np.arange(5), n=10, p=0.36).sum()
binom.pmf(np.arange(3,9), n=10, p=0.36)

a= binom.pmf(np.arange(4), n=30, p=0.2).sum()
b= binom.pmf(np.arange(25,31), n=30, p= 0.2).sum()
a+b

c= binom.pmf(np.arange(4,25), n=30, p= 0.2).sum()
1- c

# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Berunlli(p=0.3)

bernoulli.rvs(p=0.3, size=1)
bernoulli.rvs(p=0.3)
# X ~ B (n=2, p=0.3)
bernoulli.rvs(0.3) +bernoulli.rvs(0.3)

binom.rvs(1, n=2, p= 0.3)

# X ~ B(30, 0.26)
# 표본 30개를 뽑기

binom.rvs(n=30, p=0.26, size=30)


binom.rvs(n=30, p=0.26, size=1)

30 * 0.26


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 30
p = 0.26

x = np.arange(0, n + 1)

pmf = binom.pmf(x, n, p)

# 교재 p.207
x=np.arange(31)
df = pd.DataFrame({"x" : x, "prob": prob_x})
df    

sns.barplot(data = df, x= "x", y = "prob")
plt.show()

# cdf: cumlative dist. function(누적확률분포 함수)
# F(X=x) = P(X<= x)

binom.cdf(4, n=30, p=0.26)

binom.cdf(np.arange(5,18), n=30, p=0.26)

binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

#

import numpy as np
import seaborn as sns

x_1=binom.rvs(n=30, p=0.26, size=1)
x_1
x= np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color = "blue")

plt.scatter(x_1, 0.002, color='red', zorder=1, s=5)
plt.show()
plt.clf()

#
import numpy as np
import seaborn as sns

x_1=binom.rvs(n=30, p=0.26, size=10)
x_1
x= np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color = "blue")

plt.scatter(x_1, np.repeat(0.002,10), color='red', zorder=10, s=10)
plt.axvline(x=7.8, color='green', linestyle ='--', linewidth=2)
plt.show()

plt.clf()

#
binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)

binom.ppf(0.5, n=30, p=0.26)
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)


1/np.sqrt(2* math.pi)
 
from scipy.stats import norm

norm.pdf(0, loc=0, scale=1)

norm.pdf(5, loc=3, scale=4)



k= np.linspace(-3, 3, 100)
y= norm.pdf(np.linspace(-3, 3, 5), loc=0, scale=1)

plt.scatter(k, y, color='red')
plt.show()


k= np.linspace(-5, 5, 100)
y= norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color="black")
plt.show()
plt.clf()


# mu (loc)

k= np.linspace(-5, 5, 100)
y= norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color="black")
plt.show()
plt.clf()

# sigma(scale) : 분포의 퍼짐을 결정하는 모수(표준편차)

k= np.linspace(-5, 5, 100)
y= norm.pdf(k, loc=0, scale=1)
y2= norm.pdf(k, loc=0, scale=2)
y3= norm.pdf(k, loc=0, scale=0.5)

plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()

norm.cdf(0, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

norm.cdf(1, loc=0, scale=1) + (1 - norm.cdf(3, loc=0, scale=1))

# X ~ N(3, 5^2)
# P(3 < X < 5) = ? -> 15.55%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)
#위 확률변수에서 표본 100개를 뽑기
x=norm.rvs(loc=3, scale=5, size=100)

# 평균 : 0, 표준편차 : 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인

x=norm.rvs(loc=0, scale=1, size=1000)
np.mean(x<0)

x=norm.rvs(loc=3, scale=2, size=1000)
x

sns.histplot(x)
plt.show()
plt.clf()

 
# μ=3, ,𝜎=2인 정규분포에서 생성된 샘플들의 히스토그램과 해당 정규분포의 
# PDF가 빨간색 선으로 된 그래프

x=norm.rvs(loc=3, scale=2, size=1000)
x

sns.histplot(x, stat = "density")

xmin, xmax = (x.min(), x.max())

x_values = np.linspace(xmin, xmax, 100)

pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()



























