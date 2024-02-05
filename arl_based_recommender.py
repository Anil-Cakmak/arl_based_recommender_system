import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#####################
# Veriyi Hazırlama
#####################

# Online Retail II veri setinden 2010-2011 sheet’inin okutulması.
df_ = pd.read_excel("recommendation_systems/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Invoice sütununda nümerik olmayan değer barındıran gözlem birimleri fatura iptali gibi durumları ifade etmektedir.
# Ayrıca Quantity ve Price değerleri 0 veya daha küçük olan gözlem birimleri bir satın almayı ifade etmemektedir.
# Bunların veri setinden çıkarılması.
df = df[~df.Invoice.str.contains(r"[^0-9]", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# Invoice sütununda nümerik olmayan değer barındıran gözlem birimleri taşıma ücreti veya özel kampanya gibi durumları
# ifade etmektedir. Örneğin POST her faturaya eklenen posta bedelini ifade etmektedir, ürünü ifade etmemektedir.
# Bunların veri setinden çıkarılması.
df = df[~df.StockCode.str.contains(r"[^0-9]", na=False)]


#########################################################
# Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
#########################################################

# Fatura-ürün pivot table’ı oluşturulması ve Alman müşteriler için kuralların bulunması.


def create_invoice_product_df(dataframe, code=False):
    if code:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0).astype("bool")
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0).astype("bool")


def create_rules(dataframe, code=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, code)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rule = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
    return rule


rules = create_rules(df)

############################################################################
# Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
############################################################################

# Verilen ürünlerin isimlerinin bulması.
urun1 = 21987
urun2 = 23235
urun3 = 22747


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()[0]
    print(product_name)


check_id(df, urun1)  # PACK OF 6 SKULL PAPER CUPS
check_id(df, urun2)  # STORAGE TIN VINTAGE LEAF
check_id(df, urun3)  # POPPY'S PLAYHOUSE BATHROOM


# Kullanıcılar için ürün önerisinde bulunulması.


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                x = list(sorted_rules.iloc[i]["consequents"])[0]
                if x not in recommendation_list:
                    recommendation_list.append(x)
    return recommendation_list[0:rec_count]


arl_recommender(rules, urun1, 2)  # [21988, 21086]
arl_recommender(rules, urun2, 2)  # [23243, 23244]
arl_recommender(rules, urun3, 2)  # [22746, 22745]

# Önerilecek ürünlerin isimleri.
check_id(df, 21988)  # PACK OF 20 SKULL PAPER NAPKINS
check_id(df, 21086)  # SET/6 RED SPOTTY PAPER CUPS

check_id(df, 23243)  # SET OF TEA COFFEE SUGAR TINS PANTRY
check_id(df, 23244)  # ROUND STORAGE TIN VINTAGE LEAF

check_id(df, 22746)  # POPPY'S PLAYHOUSE LIVINGROOM
check_id(df, 22745)  # POPPY'S PLAYHOUSE BEDROOM
