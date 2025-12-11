import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Commented "pickle" lines were used to save time on data loading, but are not necessary for computation
data = pd.read_csv('data/cps_00005.csv/cps_00005.csv')
# #data.to_pickle("loaded.pkl")
# #data = pd.read_pickle("loaded.pkl")

#########################DATA CLEANING
#drop blanks/NIUs for sex, race, income, and educ
data = data[data['SEX'] != 9] #drop NIUs
data = data[data['RACE'] != 999]
data = data[data['INCWAGE'] != 99999999] #vals of 99999999 are NIU
data = data[data['EDUC'] != 999]

#make gender dummy variables and add them to the data set
sex_dummies = pd.get_dummies(data['SEX'], prefix='SEX', drop_first=True)
data = pd.concat([data, sex_dummies], axis=1)

#group together race subgroups into larger groups for analysis
def simplify_race(code):
    if code == 100:
        return 'White'
    elif code == 200:
        return 'Black'
    elif code in [651, 652]:
        return 'Asian'
    elif code >= 700:
        return 'Mixed/Other'
    else:
        return 'Other'
data['race_group'] = data['RACE'].apply(simplify_race)
df = pd.concat([data, pd.get_dummies(data['race_group'], prefix='race', drop_first=True)], axis=1)

#convert race to dummy variables (excluding one category to avoid multicollinearity)
race_dummies = pd.get_dummies(data['RACE'], prefix='RACE', drop_first=True)
data = pd.concat([data, race_dummies], axis=1)

#filter to ages 15-80, keeping just women in the data set
data = data[data['AGE'] > 15]
data = data[data['AGE'] < 80]
data = data[data['EDUC'] > 2]
data = data[data['SEX'] == 2] #keep just women


#recode the education variable into bins corresponding to their highest education level
def simplify_educ(code):
    if code in [10, 72]:
        return 'nonHSgrad'
    elif code == 73:
        return 'hsGrad'
    elif code in [80, 110]:
        return 'someCollegeNoBach'
    elif code == 111:
        return 'bachelors'
    elif code > 111:
        return 'moreThanBach'
    else:
        return 'Other'
data['educ_group'] = data['EDUC'].apply(simplify_educ)
#df = pd.concat([data, pd.get_dummies(data['educ_group'], prefix='educ', drop_first=True)], axis=1)

#create the treatment (states with abortion restrictions) and post (after the reversal) variables
treatedStates = [1,5,16,18,21,22,29,28,40,46,47] #alabama, arkasas, Idaho, Indiana, Kentucky, Louisiana, Missouri, Mississippi, Oklahoma, South Dakota,Tennessee
data['banned'] = data['STATEFIP'].isin(treatedStates).astype(int)
data['post'] = (data['YEAR'] >= 2022).astype(int)
data['post_banned'] = data['post'] * data['banned']

data['logIncome'] = np.log(data['INCWAGE'] + 1) #log of income to better fit the data
data['is_black'] = (data['RACE'] == 200).astype(int) #dummy var for race
data['post_banned_black'] = data['post_banned'] * data['is_black']

# data.to_pickle("cleaned.pkl")
# data = pd.read_pickle("cleaned.pkl")

#variables included for analyses
using = data[['INCWAGE', 'logIncome', 'banned', 'post', 'YEAR', 'STATEFIP', 'post_banned', 'is_black','post_banned_black', 'educ_group']].dropna()


#reg1 = smf.ols('INCWAGE ~ post_banned + C(educ_group)+ C(STATEFIP)+ C(YEAR)' , data=using).fit()
reg1 = smf.ols('INCWAGE ~ post + banned + post_banned + C(educ_group)' , data=using).fit()
print(reg1.summary())

#EDA
plt.figure()
plt.plot(data['YEAR'],data['INCWAGE'])
plt.show()

#Group by year and calculate the average income
avgWageBanned = data[data['banned'] == 1].groupby("YEAR")["INCWAGE"].mean()
avgWageNotBanned = data[data['banned'] == 0].groupby("YEAR")["INCWAGE"].mean()

#plot to check for paralell trends
plt.figure()
plt.plot(avgWageBanned.index, avgWageBanned.values, label = "States that did issue an abortion ban")
plt.gca().plot(avgWageNotBanned.index, avgWageNotBanned.values, label = "States that did not issue an abortion ban")
plt.gca().axvline(x=2022, color='grey', linestyle='--', label='Roe v. Wade was overturned')
plt.title("Average Income by Year")
plt.xlabel("Year")
plt.ylabel("Average Wage (INCWAGE)")
plt.legend()
plt.tight_layout()
plt.show()

sns.lineplot(data=df, x='YEAR', y='INCWAGE', hue='treated')

#reg2 = smf.ols('INCWAGE ~ post_banned+ C(STATEFIP)+ C(YEAR)' , data=using).fit()
reg2 = smf.ols('INCWAGE ~ post_banned + post_banned_black + is_black + C(STATEFIP) + C(YEAR)', data=using).fit()
print(reg2.summary())
