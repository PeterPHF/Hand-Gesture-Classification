
# Recenter (x,y) landmarks using wrist point as origin
def recenter_using_wrist(X):
    step = 3  # each landmark has 3 columns: x, y, z
    df_processed = X.copy()
    
    # wrist coordinates (assume wrist is the first landmark)
    x_wrist = df_processed.iloc[:, 0]
    y_wrist = df_processed.iloc[:, 1]
    
    # loop through all landmarks except wrist
    for i in range(len(df_processed.columns)//step):
        x_idx = i*step      # x column of landmark i
        y_idx = i*step + 1  # y column of landmark i
        
        df_processed.iloc[:, x_idx] = df_processed.iloc[:, x_idx] - x_wrist
        df_processed.iloc[:, y_idx] = df_processed.iloc[:, y_idx] - y_wrist
    
    return df_processed


def normalization_mid_finger(X):
    
    step = 3  # each landmark has 3 columns: x, y, z
    df_processed = X.copy()
    x_mid = X['x12']
    y_mid = X['y12']
    
    for i in range(len(df_processed.columns)//step):
        x_idx = i*step      # x column of landmark i
        y_idx = i*step + 1  # y column of landmark i
        
        df_processed.iloc[:, x_idx] = df_processed.iloc[:, x_idx] / x_mid
        df_processed.iloc[:, y_idx] = df_processed.iloc[:, y_idx] / y_mid
    
    return df_processed

def preprocessing(X):
    df_recentered = recenter_using_wrist(X)
    df_preprocessed = normalization_mid_finger(df_recentered)
    
    return df_preprocessed