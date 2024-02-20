# from sklearn.model_selection import GroupKFold
# import tensorflow.keras.backend as K, gc

# # Output of the model
# all_oof = []
# # Actual output
# all_true = []

# # Grouping by patient id
# # So for a given patient_id all instances of it will either be in the test or training set.
# gkf = GroupKFold(n_splits=5)
# for i, (train_index, valid_index) in enumerate(gkf.split(eeg_metadata_df, eeg_metadata_df.target, eeg_metadata_df.patient_id)):

#     print("#"*25)
#     print(f"### Fold {i+1}")

#     # TODO: Pretty sure that PyTorch has some cross validation thing
#     # So we only need one dataset.
#     train_gen = PreloadedDataset(eeg_metadata_df.iloc[train_index], spectrograms, all_eegs,
#                               shuffle=True, batch_size=32, augment=False)
#     valid_gen = PreloadedDataset(eeg_metadata_df.iloc[valid_index], spectrograms, all_eegs,
#                               shuffle=False, batch_size=64, mode="valid")

#     print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
#     print("#"*25)

#     K.clear_session()
#     with strategy.scope():
#         model = build_model()
#     if LOAD_MODELS_FROM is None:
#         model.fit(train_gen, verbose=1,
#               validation_data = valid_gen,
#               epochs=EPOCHS, callbacks = [LR])
#         model.save_weights(f"EffNet_v{VER}_f{i}.h5")
#     else:
#         model.load_weights(f"{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5")

#     oof = model.predict(valid_gen, verbose=1)
#     all_oof.append(oof)
#     all_true.append(eeg_metadata_df.iloc[valid_index][TARGETS].values)

#     del model, oof
#     gc.collect()

# all_oof = np.concatenate(all_oof)
# all_true = np.concatenate(all_true)


# Compute the CV score
# import sys
# sys.path.append("/kaggle/input/kaggle-kl-div")
# from kaggle_kl_div import score

# oof = pd.DataFrame(all_oof.copy())
# oof["id"] = np.arange(len(oof))

# true = pd.DataFrame(all_true.copy())
# true["id"] = np.arange(len(true))

# cv = score(solution=true, submission=oof, row_id_column_name="id")
# print("CV Score KL-Div for EfficientNetB2 =",cv)
