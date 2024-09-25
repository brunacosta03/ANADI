import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import numpy as np

import scikit_posthocs as sp

# import file
df = pd.read_csv('CO_data.csv', delimiter=',')

# 4.1

# Exercise 1
# Filter data for Portugal
portugal_data = df[df['country'] == 'Portugal']

# View Portugal's total CO2 emissions in the period 1900-2021
plt.figure(figsize=(14, 6))
plt.plot(portugal_data['year'], portugal_data['co2'])
plt.title('Total emissions CO2 in Portugal (1900-2021)')
plt.xlabel('Year')
plt.ylabel('Total emissions CO2 (in million tons)')
plt.grid(True)
plt.xticks(range(1900, 2022, 5))
plt.tight_layout()
plt.show()

# Max year CO2 in Portugal
i_max_co2_portugal = portugal_data['co2'].idxmax()
year_max_co2_portugal = portugal_data.loc[i_max_co2_portugal, 'year']
print("Portugal Max C02 in year = ", year_max_co2_portugal)

# Exercise 2
# Compare Portugal CO2 emissions from: cement, coal, flaring, gas, methane, nitrous oxide and oil using a graph
plt.figure(figsize=(14, 8))
plt.plot(portugal_data['year'], portugal_data['cement_co2'], label='Cement')
plt.plot(portugal_data['year'], portugal_data['coal_co2'], label='Coal')
plt.plot(portugal_data['year'], portugal_data['flaring_co2'], label='Flaring')
plt.plot(portugal_data['year'], portugal_data['gas_co2'], label='Gas')
plt.plot(portugal_data['year'], portugal_data['methane'], label='Methane')
plt.plot(portugal_data['year'], portugal_data['nitrous_oxide'], label='Nitrous Oxide')
plt.plot(portugal_data['year'], portugal_data['oil_co2'], label='Oil')
plt.title('Portugal CO2 Emissions by Source (1900-2021)')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (in million tons)')
plt.legend()
plt.grid(True)
plt.xticks(range(1900, 2022, 5))
plt.tight_layout()
plt.show()

# Exercise 3
spain_data = df[df['country'] == 'Spain']

portugal_data_co2_per_capita = portugal_data['co2'] / portugal_data['population']
spain_data_co2_per_capita = spain_data['co2'] / spain_data['population']

# Graph that compares CO2 emissions per capita between Portugal and Spain in the period 1900-2021.
plt.figure(figsize=(14, 8))
plt.plot(portugal_data['year'], portugal_data_co2_per_capita, label='Portugal')
plt.plot(spain_data['year'], spain_data_co2_per_capita, label='Spain')
plt.title('CO2 Emissions Per Capita: Portugal vs Spain (1900-2021)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions Per Capita (in millions tons)')
plt.legend()
plt.grid(True)
plt.xticks(range(1900, 2022, 5))
plt.tight_layout()
plt.show()

# Exercise 4
countries = ['United States', 'China', 'India', 'European Union (27)', 'Russia']
df_filtered = df[df['country'].isin(countries) & (df['year'].between(2000, 2021))]

# Filter only CO2 emissions originating from coal
df_filtered_coal = df_filtered[['country', 'year', 'coal_co2']]

# Graph that compares CO2 emissions from coal between several countries from 2000 to 2021
plt.figure(figsize=(14, 8))

for country in countries:
    country_data = df_filtered_coal[df_filtered_coal['country'] == country]
    plt.plot(country_data['year'], country_data['coal_co2'], label=country)

plt.title('CO2 Emissions Originated by Coal (2000-2021)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions Originated by Coal (in millions of tons)')
plt.legend()
plt.grid(True)
plt.xticks(range(2000, 2022, 1))
plt.tight_layout()
plt.show()

# Exercise 5
# Do the mean
means = df_filtered.groupby('country').mean()

# Select relevant columns and round
means = means[['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'methane', 'nitrous_oxide', 'oil_co2']]
means = means.round(3)

# Show result
print(means.to_string())

# 4.2
hungary_data = df[df['country'] == 'Hungary']
gdp_portugal = portugal_data[['year', 'gdp']]
gdp_hungary = hungary_data[['year', 'gdp']]

# Exercise 1 (related)
print('\n4.2.1 - gdp portugal > gdp hungary?\n')
print('H0 - GDP Hungary > GDP Portugal')
print('H1 - GDP Hungary not greater GDP Portugal\n')

seed_value = 100
years = pd.Series([i for i in range(1900, 2021)])
sample_years1 = years.sample(n=30, replace=False, random_state=seed_value)

gdp_portugal_sample = gdp_portugal[gdp_portugal['year'].isin(sample_years1)].dropna()
gdp_hungary_sample = gdp_hungary[gdp_hungary['year'].isin(sample_years1)].dropna()

common_years = set(gdp_portugal_sample['year']).intersection(set(gdp_hungary_sample['year']))

# filtered by common years so analysis stays related
gdp_portugal_sample = gdp_portugal_sample[gdp_portugal_sample['year'].isin(common_years)]
gdp_hungary_sample = gdp_hungary_sample[gdp_hungary_sample['year'].isin(common_years)]

alpha = 0.05
print('Alpha: ', alpha)
print()

# n = 26 so shapiro is used
print('shapiro portugal: ', stats.shapiro(gdp_portugal_sample['gdp']).pvalue)
print('shapiro hungary: ', stats.shapiro(gdp_hungary_sample['gdp']).pvalue)
print()

# since shapiro is low wilcoxon is used
result = stats.wilcoxon(
    gdp_portugal_sample['gdp'],
    gdp_hungary_sample['gdp'],
    alternative='greater'
)

print('Wilcoxon Proof value: ', result.pvalue)
if result.pvalue < alpha:
    print("Reject H0.")
    print("GDP Portugal > GDP Hungary.")
else:
    print("Do not reject H0.")
    print("We can't affirm GDP Portugal > GDP Hungary")

# Exercise 2 (independent, different year samples)
print('\n4.2.2 Different Sample Year Test\n')
print('H0 - GDP Hungary > GDP Portugal')
print('H1 - GDP Hungary not greater GDP Portugal\n')

alpha = 0.05
print('Alpha: ', alpha)
print()

seed_value2 = 55
sampleyears2 = years.sample(n=12, replace=False, random_state=seed_value2)
seed_value3 = 85
sampleyears3 = years.sample(n=12, replace=False, random_state=seed_value3)

gdp_portugal_sample = gdp_portugal[gdp_portugal['year'].isin(sampleyears2)].dropna()
gdp_hungary_sample = gdp_hungary[gdp_hungary['year'].isin(sampleyears3)].dropna()

# length < 30 for both, so we use shapiro
print('shapiro portugal: ', stats.shapiro(gdp_portugal_sample['gdp']).pvalue)
print('shapiro hungary: ', stats.shapiro(gdp_hungary_sample['gdp']).pvalue)
print()

# shapiro is high, so it is normalized this time (ttest_ind)
# levene to test for variances
statistic, p_value_levene = stats.levene(gdp_portugal_sample['gdp'], gdp_hungary_sample['gdp'])

print('levene: ', p_value_levene)
print()

equal_var_bool = not (p_value_levene < alpha)

t_statistic, p_value = stats.ttest_ind(gdp_portugal_sample['gdp'],
                                       gdp_hungary_sample['gdp'],
                                       alternative='greater',
                                       equal_var=equal_var_bool)

print('ttest_ind Proof value: ', p_value)

if p_value < alpha:
    print("Reject H0.")
    print("GDP Portugal > GDP Hungary.")
else:
    print("Do not reject H0.")
    print("We can't affirm GDP Portugal > GDP Hungary")

# Exercise 3 (related, many samples, friedman)
print('\n4.2.3 Significant differences between regions\n')
print('H0 - CO2 Emission differences are not relevant')
print('H1 - CO2 Emissions have significant differences\n')

alpha = 0.05
print('Alpha: ', alpha)
print()

co2_data = df[['country', 'co2', 'year']]

co2_data_sample_years_2 = co2_data[co2_data['year'].isin(sampleyears2)]

regions = ['United States', 'Russia', 'China', 'India', 'European Union (27)']

usa_sample = co2_data_sample_years_2[co2_data_sample_years_2['country'] == 'United States'].dropna()
russia_sample = co2_data_sample_years_2[co2_data_sample_years_2['country'] == 'Russia'].dropna()
china_sample = co2_data_sample_years_2[co2_data_sample_years_2['country'] == 'China'].dropna()
india_sample = co2_data_sample_years_2[co2_data_sample_years_2['country'] == 'India'].dropna()
europe_sample = co2_data_sample_years_2[co2_data_sample_years_2['country'] == 'European Union (27)'].dropna()

# Find common years
common_years = (set(usa_sample['year'])
                .intersection(set(russia_sample['year']))
                .intersection(set(china_sample['year']))
                .intersection(set(india_sample['year']))
                .intersection(set(europe_sample['year'])))

# Filter samples by common years (to ensure test stays related)
usa_sample_filtered = usa_sample[usa_sample['year'].isin(common_years)]
russia_sample_filtered = russia_sample[russia_sample['year'].isin(common_years)]
china_sample_filtered = china_sample[china_sample['year'].isin(common_years)]
india_sample_filtered = india_sample[india_sample['year'].isin(common_years)]
europe_sample_filtered = europe_sample[europe_sample['year'].isin(common_years)]

# related analysis so we use friedman test
f_statistic, p_value = stats.friedmanchisquare(usa_sample_filtered['co2'],
                                               russia_sample_filtered['co2'],
                                               china_sample_filtered['co2'],
                                               india_sample_filtered['co2'],
                                               europe_sample_filtered['co2'])


def post_hoc():
    post_hoc_data = np.array([usa_sample_filtered['co2'],
                              russia_sample_filtered['co2'],
                              china_sample_filtered['co2'],
                              india_sample_filtered['co2'],
                              europe_sample_filtered['co2']]).T

    # Perform Nemenyi test
    nemenyi_results = sp.posthoc_nemenyi_friedman(post_hoc_data)

    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 5)

    print()
    print('Nemenyi Friedman post-hoc result:\n')
    print(nemenyi_results)

    print('Indices: ')
    for index, region in enumerate(regions):
        print(f'[{index}] - {region}')

    plt.boxplot([usa_sample_filtered['co2'],
                russia_sample_filtered['co2'],
                china_sample_filtered['co2'],
                india_sample_filtered['co2'],
                europe_sample_filtered['co2']])

    plt.xticks([1, 2, 3, 4, 5], regions)
    plt.ylabel('C02 Emissions (millions of tons)')
    plt.show()


print('\nFriedman Proof value: ', p_value)
if p_value < alpha:
    print('Reject H0')
    print("There are significant differences in the total CO2 emissions between the regions.")
    post_hoc()
else:
    print('Do not reject H0')
    print("There are no significant differences in the total CO2 emissions between the regions.")

# 4.3
# Exercise 1

regions = ['Africa', 'Asia', 'South America', 'North America', 'Europe', 'Oceania']

co2_coal = df[(df['country'].isin(regions)) & (df['year'].between(2000, 2021)) & (df['coal_co2'].notna())]

co2_pivot = co2_coal.pivot_table(values='coal_co2', index='year', columns='country', aggfunc='sum')

# used pearson because it is the default. Values close to 1 or -1 indicate a strong correlation. Values close to 0
# indicate a weak correlation.
correlation_table = co2_pivot.corr(method='pearson')

plt.figure(figsize=(14, 8))
plt.matshow(correlation_table, fignum=1)

plt.xticks(range(len(correlation_table.columns)), correlation_table.columns, rotation=90)
plt.yticks(range(len(correlation_table.columns)), correlation_table.columns)

plt.colorbar()
plt.title('Correlation Matrix of CO2 Emissions from Coal (2000-2021)')
plt.show()

print('\nCorrelation Table:\n')
print(correlation_table)

# Exercise 2
df = df[(df['year'].between(2000, 2022)) & (df['year'] % 2 == 0)]

country = ['Germany', 'Russia', 'France', 'Portugal', 'Europe']

data_europe = df[df['country'].isin(country)]

pivot = data_europe.pivot_table(values='coal_co2', index='year', columns='country').dropna()

X = pd.DataFrame({
    'X1': pivot['Germany'],
    'X2': pivot['Russia'],
    'X3': pivot['France'],
    'X4': pivot['Portugal']
})
y = pivot['Europe']

Xc = sm.add_constant(X)

# OLS model, fit and summary
model = sm.OLS(y, Xc).fit()

print('Model Results: ', model.summary())

# extract the coefficients
print('\nModel Coefficients:')
print('Y = {:.4f} + {:.4f} * X1 + {:.4f} * X2 + {:.4f} * X3 + {:.4f} * X4'.format(model.params[0], model.params[1],
                                                                                  model.params[2], model.params[3],
                                                                                  model.params[4]))
print('\n\n')

# b
# shapiro test
print('Checking the normality of the residuals.')
resids = model.resid
_, p_value = stats.shapiro(resids)
print('Shapiro Test:', p_value)

alpha = 0.05
if p_value > alpha:
    print('The residuals are normally distributed.\n\n')
else:
    print('The residuals are not normally distributed., since the p-value is less than 0.05.\n\n')

# homoscedasticity test
print('Checking the homoscedasticity of the residuals.\n\n')
figure, axis = plt.subplots(2, 2)

axis[0, 0].scatter(pivot['Germany'], resids)
axis[0, 0].set_title('Germany vs Residuals')
axis[0, 1].scatter(pivot['Russia'], resids)
axis[0, 1].set_title('Russia vs Residuals')
axis[1, 0].scatter(pivot['France'], resids)
axis[1, 0].set_title('France vs Residuals')
axis[1, 1].scatter(pivot['Portugal'], resids)
axis[1, 1].set_title('Portugal vs Residuals')

for ax in axis.flat:
    ax.set(xlabel='X - Independent Variables', ylabel='Residuals')
    ax.axhline(y=0, color='black', linewidth=1)

plt.tight_layout()
plt.show()

# durbin watson test
print('Checking the autocorrelation of the residuals.\n')
dw = durbin_watson(resids)
print('Durbin Watson Test:', round(dw, 3))

if 1.5 < dw < 2.5:
    print('The residuals are not autocorrelated, so the model is valid.\n')
elif dw < 1.5:
    print('The residuals are positively autocorrelated.\n')
    print('The model is not valid, so, the residuals are not independent.\n')
else:
    print('The residuals are negatively autocorrelated.\n')
    print('The model is not valid, so, the residuals are not independent.\n')

# c
# multicollinearity test
print('\nChecking the multicollinearity of the independent variables.')
vif = pd.DataFrame()
vif['Variables'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print('VIF Results:')
print(vif)

# d
# analyse the model

print('\n\nComentários sobre o Modelo')

print(
    'O modelo apresentou um R-squared alto ajustado de 0.974, explicando uma grande porpoção da variância na emissão de CO2 proveniente do carvão na Europa.\n')
print(
    'Todos os coeficientes, exceto o constante e Portugal são estatisticamente significativos. Ou seja, as emissões provenientes do carvão da Alemanha, Rússia e França têm uma relação estatisticamente significativa com as emissões de CO2 da Europa.\n')

print('\n#1 Normalidade dos resíduos')
print(
    'Os resíduos seguem uma distribuição normal. Ou seja, a hipótese nula não foi rejeitada e as suposições do modelo de regressão linear são válidas.\n')

print('\n#2 Homocedasticidade dos resíduos')
print('Nenhum padrão foi identificado nos gráficos de dispersão dos resíduos em relação às variáveis independentes.\n')

print('\n#3 Autocorrelação dos resíduos')
print('O teste Durbin-Watson resultou em um valor de 1.57, indicando que não há autocorrelação nos resíduos.\n')

print('\n#4 Multicolinearidade das variáveis independentes')
print('O modelo apresenta multicolinearidade, o que sugere uma cautela maior na interpretação dos coeficientes.\n')

# e

data2 = pd.read_csv('CO_data.csv')

# filter data for Europe in 2015
df_europe_2015 = data2[(data2['country'] == 'Europe') & (data2['year'] == 2015) & (data2['coal_co2'].notna())]

# real CO2 emissions in Europe in 2015
real_emission_2015 = df_europe_2015['coal_co2'].values[0]

# filter data for Germany, Russia, France and Portugal in 2015
germany_coal = data2[(data2['country'] == 'Germany') & (data2['year'] == 2015)]
russia_coal = data2[(data2['country'] == 'Russia') & (data2['year'] == 2015)]
france_coal = data2[(data2['country'] == 'France') & (data2['year'] == 2015)]
portugal_coal = data2[(data2['country'] == 'Portugal') & (data2['year'] == 2015)]

# estimate the CO2 emissions in Europe in 2015
X1_2015 = germany_coal['coal_co2'].values[0]
X2_2015 = russia_coal['coal_co2'].values[0]
X3_2015 = france_coal['coal_co2'].values[0]
X4_2015 = portugal_coal['coal_co2'].values[0]

X_2015 = pd.DataFrame({
    'const': [1],
    'X1': [X1_2015],
    'X2': [X2_2015],
    'X3': [X3_2015],
    'X4': [X4_2015]
})

# model prediction to estimate the CO2 emissions in Europe in 2015
estimated_emission_2015 = model.predict(X_2015)

print('Real CO2 Emission in Europe in 2015:', round(real_emission_2015, 4))
print('Estimated CO2 Emission in Europe in 2015:', round(estimated_emission_2015[0], 4))

df = pd.DataFrame({
    'Emission Type': ['Real', 'Estimated'],
    'CO2 Emission (in million tons)': [real_emission_2015, estimated_emission_2015[0]]
})

# bars that compare the real and estimated CO2 emissions in Europe in 2015
bars = plt.bar(df['Emission Type'], df['CO2 Emission (in million tons)'], color=['blue', 'green'])
plt.title('Real vs Estimated CO2 Emission in Europe in 2015')
plt.ylabel('CO2 Emission (in million tons)')

# shows the exact value ontop of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), va='bottom')

plt.show()

difference = real_emission_2015 - estimated_emission_2015[0]
print('Difference: (Real - Estimated)', round(difference, 4))
