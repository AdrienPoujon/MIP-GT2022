import pandas as pd
from sklearn.model_selection import train_test_split

def contatenate_datasets():
    csv_path_1 = '/data/Datasets/COVID-19_Radiography_Dataset/COVID.metadata.xlsx'
    csv_path_2 = '/data/Datasets/COVID-19_Radiography_Dataset/Lung_Opacity.metadata.xlsx'
    csv_path_3 = '/data/Datasets/COVID-19_Radiography_Dataset/Viral Pneumonia.metadata.xlsx'
    csv_path_4 = '/data/Datasets/COVID-19_Radiography_Dataset/Normal.metadata.xlsx'
    df_1 = pd.read_excel(csv_path_1)
    df_1_test = df_1.sample(n=500)
    df_2 = pd.read_excel(csv_path_2)
    df_2_test = df_2.sample(n=500)
    df_3 = pd.read_excel(csv_path_3)
    df_3_test = df_3.sample(n=500)
    df_4 = pd.read_excel(csv_path_4)
    df_4_test = df_4.sample(n=500)
    df = pd.concat([df_1,df_2,df_3,df_4])
    df_test = pd.concat([df_1_test,df_2_test,df_3_test,df_4_test])
    df_train = df.drop(df_test.index)
    print(df_train.head())
    print(len(df))

    df_train = df_train.drop(columns = ['URL','SIZE'])
    df_train['State'] = ""
    df_train['Label'] = ""
    df_train = extract_state(df_train)

    df_test = df_test.drop(columns=['URL', 'SIZE'])
    df_test['State'] = ""
    df_test['Label'] = ""
    df_test = extract_state(df_test)


    df_train.to_csv('/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/train.csv',index=False)
    df_test.to_csv('/home/kiran/Desktop/Dev/ECE6780_MedEmbeddings01/csv_files/COVID_KAGGLE/test.csv', index=False)


def extract_state(df):
    for i in range(0,len(df)):
        file_name = df.iloc[i,0]
        disease = file_name.split('-')
        if(disease[0] =='NORMAL'):
            df.iloc[i,2] = 'Normal'
        else:
            df.iloc[i,2] = disease[0]
        if(df.iloc[i,2] == 'Normal'):
            df.iloc[i,3] = 0
        elif(df.iloc[i,2] == 'COVID'):
            df.iloc[i,3] = 1
        elif (df.iloc[i, 2] == 'Lung_Opacity'):
            df.iloc[i, 3] = 2
        else:
            df.iloc[i, 3] = 3

    return df

def process_dataframe(df_dir):
    df = pd.read_csv(df_dir)
    df = df.fillna(0)
    df = df.replace(-1,0)
    df.to_csv('/data/Datasets/CheXpert-v1.0-small/train_filtered.csv',index=False)
if __name__ == '__main__':
    df_dir = '/data/Datasets/CheXpert-v1.0-small/train.csv'
    process_dataframe(df_dir=df_dir)
    #contatenate_datasets()