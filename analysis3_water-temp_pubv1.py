from os.path import abspath, dirname, join, pardir
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import warnings

from general_v2 import prep, prep3

warnings.filterwarnings('ignore')
f_path = "C:/windows/Fonts/malgun.ttf"
fm.FontProperties(fname=f_path).get_name()
plt.rc('font', family='Malgun Gothic')


def correlation(scaled_df, savePath, file_label):
    copy_df = scaled_df.copy()
    copy_df.drop(columns=['C0000'], inplace=True)
    corrmat = copy_df.corr(method="spearman") #or pearson
    corr_1 = corrmat.nlargest(n=15, columns='Water Temperature')
    corr_1 = corr_1[list(corr_1.index)]
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_1, square=True)
    plt.tight_layout()
    plt.savefig(savePath + f"{file_label}_corre_all.png")
    plt.show()
    corr_1.to_csv(savePath + f"{file_label}_corre_all.csv", encoding='CP949')

    #corrs = corr_1[abs(corr_1) >= 0.7]
    corrs = corr_1.copy()
    for col in corrs.columns:
        # print(corrs[col].values[0])
        if np.isnan(corrs[col].values[0]):
            corrs = corrs.drop([col])
            corrs = corrs.drop([col], axis=1)
    plt.figure(figsize=(12,10))
    # sns.set(font_scale=0.5)
    sns.heatmap(corrs, annot=True)
    plt.tight_layout()
    plt.savefig(savePath + f"{file_label}_corre_sub.png")
    plt.show()


def randomForestRegressor(scaled_df, savePath, file_label):
    #buoy_col = ['풍속(m/s)', '풍향(deg)', 'GUST풍속(m/s)', '현지기압(hPa)', '습도(%)', '기온(°C)', '수온(°C)', '최대파고(m)', '유의파고(m)', '평균파고(m)', '파주기(sec)', '파향(deg)']
    ########## RandomForestRegressor

    copy_df2 = scaled_df.copy()
    copy_df2.drop(columns=['Water Temperature'], inplace=True)
    copy_df2.drop(columns=['C0000'], inplace=True)
    x = copy_df2.values
    y = scaled_df['Water Temperature'].values

    model = RandomForestRegressor(criterion = 'squared_error').fit(x, y)
    reg_feat_import = model.feature_importances_
    print('SKlearn :', reg_feat_import)

    # plot
    df_fi = pd.DataFrame({'columns':copy_df2.columns, 'importances':reg_feat_import})
    df_fi = df_fi.sort_values(by=['importances'], ascending=False)[:10]
    #df_fi = df_fi[df_fi['importances'] > 0.01] # importance가 0이상인 것만


    fig = plt.figure(figsize=(15,7))
    ax = sns.barplot(x=df_fi['importances'].values, y=df_fi['columns'].values)
    ax.set_yticklabels(df_fi['columns'], fontsize=13)
    plt.tight_layout()
    plt.savefig(savePath + f"{file_label}_randomF_importV.png")
    plt.show()


if __name__ == '__main__':
    filePath = 'X:/Python/수온예보/20230807_buoy155102/'
    savePath = 'X:/Python/수온예보/20230807_buoy155102_correlation/'

    orig = pd.read_csv(join(filePath, 'combined_data.csv'), header=0, encoding='CP949')
    orig.set_index("일시", inplace=True)
    orig.index.name = '일시'
    orig.drop(columns=['지점'], inplace=True)
    orig = orig.dropna(axis=0, how='any')

    orig.rename(columns={'풍속(m/s)': 'Wind Speed'}, inplace=True)
    orig.rename(columns={'풍향(deg)': 'Wind Direction'}, inplace=True)
    orig.rename(columns={'GUST풍속(m/s)': 'GUST Wind Speed'}, inplace=True)
    orig.rename(columns={'현지기압(hPa)': 'Atmospheric Pressure'}, inplace=True)
    orig.rename(columns={'습도(%)': 'Humidity'}, inplace=True)
    orig.rename(columns={'기온(°C)': 'Temperature'}, inplace=True)
    orig.rename(columns={'수온(°C)': 'Water Temperature'}, inplace=True)
    orig.rename(columns={'최대파고(m)': 'Maximum Wave Height'}, inplace=True)
    orig.rename(columns={'유의파고(m)': 'Significant Wave Height'}, inplace=True)
    orig.rename(columns={'평균파고(m)': 'Average Wave Height'}, inplace=True)
    orig.rename(columns={'파주기(sec)': 'Wave Period'}, inplace=True)
    orig.rename(columns={'파향(deg)': 'Wave Direction'}, inplace=True)

    orig['C0000'] = np.arange(len(orig))
    orig = prep(orig, 'Water Temperature', 0.3)
    orig = prep3(orig, 'Temperature', 10)

    feature_sc = MinMaxScaler(feature_range=(0, 1))
    data_scaled = feature_sc.fit_transform(orig)
    scaled_df = pd.DataFrame(data_scaled, columns=orig.columns)

    correlation(scaled_df, savePath, 'combined_data')
    randomForestRegressor(scaled_df, savePath, 'combined_data')
