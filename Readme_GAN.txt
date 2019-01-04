GAN.py:
    generator:
        輸入: 1026維度(前513為實數部分，後513為對應的虛數部分的"數值")
        輸出: 1539維度的"複數"
    discriminator:
        輸入: 3078維度(前1539為bgm, vocal, mixture的stft實數部分，後1539為對應的虛數部分的"數值")
        輸出: 0~1
    mask: 做frequency mask
    cast_complex: 將數字轉成複數
    combine: 將預測結果和原本的mixture data連接在一起(變成能丟入discriminator的維度)
    calc_complex: 計算複數絕對值大小
    prepare_train_data: 隨機選取batch_size個歌(可能重複)，這些歌各自在隨機選取一個time frame，將該time frame中的frequency資料存起來
    pre_train: supervised learning，若直接unsupervised learning會造成G和D完全不知誰對誰錯而結果怪異
    train: unsupervised learning, 根據ppt train 3000個epoch，每次D訓練10次、G訓練1次
    predict: 產生generator預測的bgm, vocal
