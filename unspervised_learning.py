# CUSTOMER SEGMENTATION UNSUPERVISED LEARNING

# İş Problemi
# E-ticaret sitesi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

# Veri Seti Hikayesi
# Veri seti E-ticaret sitesinden son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# Değişkenler
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.max_rows", None)
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore")

# Veri Seti Okutma
df = pd.read_csv("dataset/flo_data_20k.csv")

# EDA(Exploratory Data Analysis)
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include="number").quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# "dtype" hatası olan değişkenleri düzeltelim.
date_cols = df.columns[df.columns.str.contains("date")]
df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))

# Data Pre-Processing
def grab_col_names(dataframe, cat_th=9, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik ve Numerik Değişkenlerin veri setindeki durumlarına bakalım.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# Kategorik Değişkeneler için:
for col in cat_cols:
    cat_summary(df, col)

# Numerik Değişkenler için
num_cols = [col for col in num_cols if col not in date_cols]
for col in num_cols:
    num_summary(df, col)

# Aykırı Değerlere bakalım.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe.loc[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))  # AYKIRI DEĞERLER VAR.

# Eksik Değerlere Bakalım.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns
missing_values_table(df)  # Eksik Değer YOK.

# Korelasyon Bakalım.
def corr_map(df, width=14, height=6, annot_kws=15, corr_th=0.7):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize = (width,height))
    sns.heatmap(df.corr(),
                annot= True,
                fmt = ".2f",
                ax=ax,
                vmin = -1,
                vmax = 1,
                cmap = "RdBu",
                mask = mtx,
                linewidth = 0.4,
                linecolor = "black",
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0,size=15)
    plt.xticks(rotation=75,size=15)
    plt.title('\nCorrelation Map\n', size = 40)
    plt.show()
    return drop_list
corr_map(df[num_cols])

# FEATURE ENGINEERING

# Yeni feature üretelim.
df["customer_total_shopping"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# recency ve tenure değişkenleri üretelim.
analysis_date = df["last_order_date"].max()

df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (df["last_order_date"] - df["first_order_date"]).dt.days

# Değişkenleri tekrar kategorileştirelim.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in date_cols]

# Aykırı değerleri baskılayalım.
for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# K-MEANS ile Müşteri Segmantasyonu

# Veri setini sayısal değişkenlerden oluşturalım.
model_df = df[num_cols]

# Normal Dağılıma bakmak için çarpıklık testi yapalım.
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))

plt.figure(figsize=(9, 9))
plt.subplot(7, 1, 1)
check_skew(model_df, 'order_num_total_ever_online')
plt.subplot(7, 1, 2)
check_skew(model_df, 'order_num_total_ever_offline')
plt.subplot(7, 1, 3)
check_skew(model_df, 'customer_value_total_ever_offline')
plt.subplot(7, 1, 4)
check_skew(model_df, 'customer_value_total_ever_online')
plt.subplot(7, 1, 5)
check_skew(model_df, "customer_total_shopping")
plt.subplot(7, 1, 6)
check_skew(model_df, 'recency')
plt.subplot(7, 1, 7)
check_skew(model_df, 'tenure')
plt.tight_layout()
# plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)

# Değişkenlerde çok fazla çarpıklı olduğunu ve normal dağılım olmadığını görüyoruz.O yüzden Log Transformation uygulayalım.

# Normal dağılımın sağlanması için Log transformation uygulanması
model_df['order_num_total_ever_online'] = np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline'] = np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline'] = np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online'] = np.log1p(model_df['customer_value_total_ever_online'])
model_df['customer_total_shopping'] = np.log1p(model_df['customer_total_shopping'])
model_df['recency'] = np.log1p(model_df['recency'])
model_df['tenure'] = np.log1p(model_df['tenure'])
model_df.head()

# Scale İşlemleri
sc = StandardScaler()
model_df[num_cols] = sc.fit_transform(model_df[num_cols])

# K-Means ile Model Aşaması
# Optimum küme sayısı için elbow yöntemini kullanalım.
k_means = KMeans()
elbow = KElbowVisualizer(k_means, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# KElbowVisualizer tarafından bulunan ideal küme sayısı
elbow.elbow_value_

# Modeli oluşturup müşterileri segmentleyelim.
k_means = KMeans(n_clusters=elbow.elbow_value_, random_state=10).fit(model_df)

# Segmentlerin merkezlerini görelim.
k_means.cluster_centers_

# Segmentleri görelim.
segments = k_means.labels_

# müşterilerin segmentlerin olduğu bir dataframe oluşturalım.
final_df = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online",
               "customer_total_shopping", "recency", "tenure"]]

final_df["segment"] = segments
# Segmentler 0'dan başladığı için 1 arttıralım.
final_df["segment"] = final_df["segment"] + 1

# Şimdi her bir segmenti istatistiksel olarak inceleyelim.
final_df.groupby("segment").agg({"order_num_total_ever_online": ["count", "mean"],
                                 "order_num_total_ever_offline": ["count", "mean"],
                                 "customer_value_total_ever_online": ["count", "mean"],
                                 "customer_value_total_ever_offline": ["count", "mean"],
                                 "customer_total_shopping": ["count", "mean"],
                                 "recency": ["count", "mean"],
                                 "tenure": ["count", "mean"]})

# Hierarchical Clustering ile Müşteri Segmantasyonu

# model_df dataframe üzerinden ilerleyeceğim.
# Optimum küme sayısı için dendogram bakalım.
hc_complate = linkage(model_df, method="complete")

plt.figure(figsize=(10, 6))
plt.title("Dendrograms")
dend = dendrogram(hc_complate,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10,
           labels=model_df.index)
plt.axhline(y=1.24, color='r', linestyle='--')
plt.show()


# Model oluşturalım ve müşterileri segmentleyelim.
hc = AgglomerativeClustering(n_clusters=7, linkage="complete")
hc_segments = hc.fit_predict(model_df)

# Dataframe'e ekleyelim.
final_df["hc_segments"] = hc_segments
final_df["hc_segments"] = final_df["hc_segments"] + 1
final_df.head()

# Şimdi her bir segmenti istatistiksel olarak inceleyelim.
final_df.groupby("hc_segments").agg({"order_num_total_ever_online": ["count", "mean"],
                                    "order_num_total_ever_offline": ["count", "mean"],
                                    "customer_value_total_ever_online": ["count", "mean"],
                                    "customer_value_total_ever_offline": ["count", "mean"],
                                    "customer_total_shopping": ["count", "mean"],
                                    "recency": ["count", "mean"],
                                    "tenure": ["count", "mean"]})


