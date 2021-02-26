import os 
import numpy as np 
import pandas as pd 
from PIL import Image 
from tqdm import tqdm
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics


#this is common function that transform the images in the file 
def create_dataset(training_df,image_dir):

    '''
    training_df: dataframe of the training data 
    dataframe : which has train_dataframe, target and image_id 
    output: (x,y) value which is array and labels 
    '''

    images=[]
    labels=[]

    for index, row in tqdm(training_df.iterows(),total=len(training_df),desc='processing image'):
        #get image id 
        image_id= training_df['ImageId']
        #join path of image 
        image_loc=os.path.join(image_dir,image_id)
        #image open the location 

        image=Image(image_loc , +'.png')

        #process each image 
        image = image.resize((256,256),resample=Image.BILINEAR)
        #convert image to array 
        image=np.array(image)

        #ravel 
        image=image.ravel()

        images.append(image)
        labels.append(int(row['targets']))

    images=np.array(images)
    print(images.shape)
    return images, labels 


if __name__ == '__main__':
    csv_path= '../dir/dir/iamge_dir.csv'
    image_path = '../dir/dir/image_Stror_png'
    #read csv_file into the programe
    df=pd.read_csv(csv_path)
    #create the fold
    df['kfold']=-1

    #create the randomize row 
    df=df.sample(frac=1).reset_index(Drop=True)

    #fetch labels 

    y=df.targets.values

    kf=model_selection.StratifiedKFold(n_splits=5)

    #fill the new k_fold column into dataframe 
    for f, (t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold']= f

    #now we go over the fold that we have created

    for fold_ in range(5):
        #create temporory dataframe for train and test 
        train_df=df[df.kfold!=fold_].reset_index(drop=True)
        test_df=df[df.kfold==fold_].reset_index(drop=True)

        #now create a train dataset 
        xtrain,ytrain=create_dataset(train_df,image_path)

        #now create the dataset 

        xtest,ytest=create_dataset(test_df,image_path)

        #fit the random forest algo and fit the model 

        clf=ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain,ytrain)


        preds=clf.predict_proba(xtest)[:,-1]

        print(f'FOLD: {fold_}')
        print(f'AUC: {metrics.roc.auc_score(y_test,preds)}')
        print(' ')


