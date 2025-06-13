# <YOUR_IMPORTS>
import datetime
import json
import os
import logging
import dill
import pandas as pd
from datetime import datetime

path = '/home/dev/airflow/plugins'


# def predict():
#     with open(f'{path}/data/models/cars_pipe_202506090037.pkl', 'rb') as file:
#         model = dill.load(file)
#
#     files_test_dir = os.listdir(f'{path}/data/test')
#     df_pred = pd.DataFrame(columns=['id', 'predict'])
#
#
#
#
#     for file_name in files_test_dir:
#         with open(f'{path}/data/test/{file_name}', 'rb') as file:
#             test_form = json.load(file)
#
#         df = pd.DataFrame.from_dict([test_form])
#         y = model.predict(df)
#         dict_pred = {'id': df.id, 'predict': y}
#         df_pred_values = pd.DataFrame(dict_pred)
#         df_pred = pd.concat([df_pred, df_pred_values], ignore_index=True)
#
#     df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv',index=False)
#     print(df_pred)


def predict():
    model_path = f'{path}/data/models'
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError('No model files found')

    latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    with open(f'{model_path}/{latest_model_file}', 'rb') as file:
        model = dill.load(file)

    logging.info(f'{latest_model_file} is loaded')

    test_data_path = f'{path}/data/test'
    test_files = [f for f in os.listdir(test_data_path) if f.endswith('.json')]
    if not test_files:
        raise FileNotFoundError('No test data found')

    logging.info(f'{len(test_files)} test data found')

    data = []
    file_ids = []

    for test_file in test_files:
        with open(f'{test_data_path}/{test_file}', 'rb') as file:
            content = json.load(file)
            data.append(content)
            file_ids.append(test_file.split('.')[0])

    df = pd.DataFrame(data)

    predictions = model.predict(df)

    result_df = pd.DataFrame({
        'car_id': file_ids,
        'pred': predictions
    })

    predictions_path = f'{path}/data/predictions'
    os.makedirs(predictions_path, exist_ok=True)
    output_file = f'{predictions_path}/preds_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    result_df.to_csv(output_file, index=False)

    logging.info(f'Predictions saved to {output_file}')


if __name__ == '__main__':
    predict()
