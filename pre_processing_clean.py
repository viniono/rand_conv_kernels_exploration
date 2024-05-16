
## Libraries
from sktime.classification.kernel_based import RocketClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Borrowed from Professor Miller
### Function to check and trim data for samples that are too long/short
def check_and_trim(df):
    # Define the desired size per group
    desired_size = 3600

    # Create an empty DataFrame to hold the trimmed data
    trimmed_df = pd.DataFrame()

    # Group the dataframe by 'Full_Sample_ID' and iterate over the groups
    for full_sample_id, group in df.groupby('Full_Sample_ID'):
        # If the size of the group is larger than the desired size, take the first 'desired_size' rows
        if len(group) > desired_size:
            group = group.iloc[:desired_size]
        # Append the group to the trimmed dataframe
        trimmed_df = pd.concat([trimmed_df, group], ignore_index=True)

    return trimmed_df

def pre_processing(files,root='.'):
     ## Set up lists
    pred_vars = ['Full_Sample_ID', 'Brake','Accel', 'Lat_Pos', 'Speed', 'Heading', 'Wheel_Rate']
    x_vars = ['Brake','Accel', 'Lat_Pos', 'Speed', 'Heading', 'Wheel_Rate']
    y_vars = ['Target','Full_Sample_ID']
    # x_vars = ['Lat_Pos', 'Speed']  ## Reduced set of X-vars
    # pred_vars = ['Full_Sample_ID', 'Lat_Pos', 'Speed'] ## Reduced set of X-vars
    split_names = ['split1','split2','split3','split4','split5','split6','split7','split8','split9','split10']

    ###  Outer Loop through diff experiments
    for cur_file in files:
        data = pd.read_csv(root + '/' + cur_file)
        data = data.rename(columns={'CFS.Brake.Pedal.Force': 'Brake', 
                            'CFS.Accelerator.Pedal.Position': 'Accel', 
                            'SCC.Lane.Deviation.2': 'Lat_Pos',
                            'VDS.Veh.Heading.Fixed': 'Heading',
                            'VDS.Veh.Speed': 'Speed',
                            'CFS.Steering.Wheel.Angle': 'Wheel_Angle',
                            'CFS.Steering.Wheel.Angle.Rate': 'Wheel_Rate'})
        
        ## File where where results will be stored
        res_file = cur_file.strip('.csv') + '_rawpreds.csv'

        ### Inner loop through diff validation sets
        ict_res_df = pd.DataFrame(columns = ['val_split','sample_ID','pos_class_proba','target'])
        for sp in split_names:
            train_y = data.loc[data[sp] == 'train'][y_vars]
            train_X = check_and_trim(data.loc[data[sp] == 'train'][pred_vars]).iloc[1::6, :]
            # train_X = check_and_trim(data.loc[data[sp] == 'train'][pred_vars])
            test_y = data.loc[data[sp] == 'val'][y_vars]
            # test_X = check_and_trim(data.loc[data[sp] == 'val'][pred_vars])
            test_X = check_and_trim(data.loc[data[sp] == 'val'][pred_vars]).iloc[1::6, :]
            
            ## Reformat data in 3-d arrays suitable for inception time/rocket
            train_X_array = np.zeros(shape = (train_X['Full_Sample_ID'].nunique(), len(pred_vars) - 1, 
                                            int(train_X.shape[0]/train_X['Full_Sample_ID'].nunique())))
            train_y_small = np.zeros(shape = (train_X['Full_Sample_ID'].nunique()))

            test_X_array = np.zeros(shape = (test_X['Full_Sample_ID'].nunique(), len(pred_vars) - 1, 
                                            int(test_X.shape[0]/test_X['Full_Sample_ID'].nunique())))
            test_y_small = np.zeros(shape = (test_X['Full_Sample_ID'].nunique()))

            id_list = train_X['Full_Sample_ID'].unique()
            i = 0
            for cur_id in id_list:
                ji = 0
                for j in x_vars:
                    train_X_array[i,ji,:] = train_X.loc[train_X['Full_Sample_ID'] == cur_id][j]
                    ji=ji+1
                train_y_small[i] = train_y.loc[train_y['Full_Sample_ID'] == cur_id]['Target'].unique()
                i=i+1

            test_id_list = test_X['Full_Sample_ID'].unique()
            i = 0
            for cur_id in test_id_list:
                ji = 0
                for j in x_vars:
                    test_X_array[i,ji,:] = test_X.loc[test_X['Full_Sample_ID'] == cur_id][j]
                    ji=ji+1
                test_y_small[i] = test_y.loc[test_y['Full_Sample_ID'] == cur_id]['Target'].unique()
                i=i+1
    return [train_X_array,train_y_small, test_X_array, test_y_small]



if __name__ == '__main__':
    file_name = input("File to run: ")
    print("You entered: " + file_name)
    
    
    ## Get file names
    root = '.'
    # files = os.listdir(root) ## For everything
    files = [file_name] 
    df=pre_processing(files)
    
    param_grid = {
    'num_kernels': [10000, 20000, 30000],
    "random_state":["None","int"]
    }


    rocket = Rocket()

    grid_search = GridSearchCV(estimator=rocket, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(df[0])

    best_params = grid_search.best_params_
    best_rocket = Rocket(**best_params)
    best_rocket_fit=best_rocket.fit(df[0])
    X_train_transform = best_rocket_fit.transform(df[0])
    
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, df[1])

    X_test_transform = best_rocket.transform(df[2])
    classifier.score(X_test_transform, df[3])

    coeff=[]
    feature=[]

    for i in range(classifier.coef_.size):
        coeff.append(classifier.coef_[0,i])
        feature.append(i+1)
    
    # //dataset with all ccoefficeints from ridge
    ridge_coefficients=pd.DataFrame(data={"coeff":coeff,"Kernel":feature},index=range(1,20001))
    ridge_coefficients.index.names=['feature']
    ridge_coefficients["Kernel"]=np.ceil(ridge_coefficients["Kernel"]/2)
    #sort coefficeints by highest magnitude anbd then returns the kernels from the top 10 features
    # ridge_coefficients.sort_values(by="coeff",key=abs,ascending=False).head(n=10)
    top1=ridge_coefficients.sort_values(by="coeff",key=abs,ascending=False)["coeff"].iloc[0]
    ridge_coefficients["Importance"]=np.abs(ridge_coefficients["coeff"])/np.abs(top1)
    feature_index=ridge_coefficients.sort_values(by="coeff",key=abs,ascending=False).head(n=50)["Kernel"]

    # assign the max importance score of a givben feature to the entire kernel
    max_importance_per_kernel = ridge_coefficients.groupby('Kernel')['Importance'].max()
    max_importance_df = max_importance_per_kernel.reset_index()
    max_importance_df.columns = ['Kernel', 'Max_Importance']


   # Assuming kernelss is a tuple containing the generated kernels
    kernel_info = []

    start_index = 0
    end_index = 0
    for i in range(best_rocket_fit.num_kernels):
        
        lengths = best_rocket_fit.kernels[1][i]  
        end_index += lengths
        weights = best_rocket_fit.kernels[0][start_index:end_index]  
        biases = best_rocket_fit.kernels[2][i]  
        dilations = best_rocket_fit.kernels[3][i]  
        paddings = best_rocket_fit.kernels[4][i] 
        num_channel_indices = best_rocket_fit.kernels[5][i]  
        channel_indices = best_rocket_fit.kernels[6][sum(best_rocket_fit.kernels[5][:i]):
                                                    sum(best_rocket_fit.kernels[5][: i + 1])]  
        
        kernel_dict = {
            "Kernel": i + 1,
            "Weights": weights.tolist(),
            "Length": lengths,
            "Bias": biases,
            "Dilation": dilations,
            "Padding": paddings,
            "Num_Channel_Indices": num_channel_indices,
            "Channel_Indices": channel_indices.tolist()
        }
        
        kernel_info.append(kernel_dict)
        start_index = end_index
        
    kernel_info_df = pd.DataFrame(kernel_info)
    df_importance_all_kernels=pd.merge(kernel_info_df, max_importance_df, on="Kernel", how="inner")
    
    df_importance_all_kernels.to_pickle(file_name+".pkl")
    ridge_coefficients.to_pickle("Coefficients_ridge.pkl")



                
                
