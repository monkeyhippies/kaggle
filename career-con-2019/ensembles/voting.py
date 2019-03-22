import os
import pandas as pd

def vote_submissions():
    dfs = []
    for file in os.listdir("../input/"):
        df = pd.read_csv(os.path.join("../input/", file))
        df.index = df['series_id']
        df = df.drop(["series_id"], axis=1)
        dfs.append(df)

    all_predictions = pd.concat(dfs, axis=1)
    voted_prediction = all_predictions.mode(axis=1)
    
    return pd.DataFrame({
        "series_id": voted_prediction.index,
        "surface": voted_prediction[0]
    })

from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

df = vote_submissions()
create_download_link(df)
