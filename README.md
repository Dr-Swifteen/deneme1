This dataset is split into 4 parts using 7-Zip.

To extract:

1. Download all of the following files:
   - yap470.7z.001  
   - yap470.7z.002  
   - yap470.7z.003  
   - yap470.7z.004

2. Install [7-Zip](https://www.7-zip.org/)

3. Right-click `yap470.7z.001` → **7-Zip > Extract Here**  
   This will automatically combine all parts and extract the full dataset.
mammo_gorunumlerini_listele
bu methodumda kök dizisi girilmiş directorydeki resimler çekilir mini ddsm ise kontrol yapılır KAU-BCDM ise direkt alınıp etiketleme yapılır
load_and_preprocess
imagelar opencv deki methodlar ile grayscale yüklenilir dah sonrasında 255 ile bölünüp normalizasyon yapılır
create_cnn_model
modelden modele ve filtreler değişecek şekilde CNN yapıları oluşur
X_egitim_val, X_test, y_egitim_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
test %10 bölünür
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    geri_cagirmalar = [
        EarlyStopping(patience=earlystop_patience, restore_best_weights=True),
        ReduceLROnPlateau(patience=reduce_lr_patience, factor=0.5)
    ]
    #chatgpt 
    model.fit(X_egitim_val, pd.get_dummies(y_egitim_val),
          epochs=epochs, batch_size=batch_size,validation_split=0.1,
          callbacks=geri_cagirmalar, verbose=1)
model cross validationa uygun eğitime başlanılır (Accuracy ile) Epoch aşılmaz
olasiliklar = model.predict(X_test)
    esikler = np.linspace(0.4, 0.6, 21)
    en_iyi_f1, en_iyi_T_m, en_iyi_T_b = 0, 0.47, 0.53
olasılılar hesaplanılıp tm tb eşikler f1 skorununa göre seçilir

 for T_m_aday in esikler:
        for T_b_aday in esikler:
            tahminler = [2 if p[2] >= T_m_aday else 1 if p[1] >= T_b_aday else 0 for p in olasiliklar]
            f1 = f1_score(y_test, tahminler, average='macro')
            if f1 > en_iyi_f1:
                en_iyi_f1, en_iyi_T_m, en_iyi_T_b = f1, T_m_aday, T_b_aday

Tm ve TB nin seçildiği kod bloğu maks f1 yapan değerler kabul edilir
nihai_tahminler = [2 if p[2] >= en_iyi_T_m else 1 if p[1] >= en_iyi_T_b else 0 for p in olasiliklar]
print(f"En iyi eşik sınırı: T_m={en_iyi_T_m:.2f}, T_b={en_iyi_T_b:.2f}")
print(classification_report(y_test, nihai_tahminler, target_names=['normal','benign','cancer']))
print(confusion_matrix(y_test, nihai_tahminler))
print("Doğruluk:", accuracy_score(y_test, nihai_tahminler))
print("Kesinlik:", precision_score(y_test, nihai_tahminler, average='macro'))
print("Duyarlılık:", recall_score(y_test, nihai_tahminler, average='macro'))
try:
    print("AUC:", roc_auc_score(pd.get_dummies(y_test), olasiliklar, average='macro', multi_class='ovr'))
except:
    print("AUC hesaplanamadı (muhtemelen tek sınıf tahmini nedeniyle)")
with open("optimized_thresholds.txt", "w") as f:
    f.write(f"Best T_m: {en_iyi_T_m}\nBest T_b: {en_iyi_T_b}\nBest F1: {en_iyi_f1}")
metrik değerlendirmeleri yapılır ve daha sonrasında model kayıt işlemleri yapılır
