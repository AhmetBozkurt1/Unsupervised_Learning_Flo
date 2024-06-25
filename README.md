# CUSTOMER SEGMENTATION UNSUPERVISED LEARNING

### İş Problemi (Business Problem)
Bir E-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak.

### Veri Seti Hikayesi
Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

### Değişkenler
* master_id: Eşsiz müşteri numarası
* order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
* last_order_channel: En son alışverişin yapıldığı kanal
* first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
* last_order_date: Müşterinin yaptığı son alışveriş tarihi
* last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
* last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
* order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
* order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
* customer_value_total_ever_online: Müşterinin offline alışverişlerinde ödediği toplam ücret
* customer_value_total_ever_offline: Müşterinin online alışverişlerinde ödediği toplam ücret
* interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

### MODEL OLUŞTURMA
- Veri seti keşfedilir ve özelliklerin analizi yapılır.
- Eksik veriler ve aykırı değerler işlenir.
- Özellik mühendisliği adımlarıyla yeni özellikler türetilir.
- Kategorik değişkenler sayısal formata dönüştürülür.
- Denetimsiz öğrenme model işlemleri yapılır.
- Müşterileri kümeleme işlemleri ile segmente edilir.


### Gereksinimler
☞ Bu proje çalıştırılmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- yellowbrick

### Kurulum
☞ Projeyi yerel makinenizde çalıştırmak için şu adımları izleyebilirsiniz:
- GitHub'dan projeyi klonlayın.
- Projeyi içeren dizine gidin ve terminalde `conda env create -f environment.yaml` komutunu çalıştırarak gerekli bağımlılıkları yükleyin.
- Derleyicinizi `conda` ortamına göre ayarlayın.
- Projeyi bir Python IDE'sinde veya Jupyter Notebook'ta açın.
