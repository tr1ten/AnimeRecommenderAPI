import uvicorn
from fastapi import FastAPI
from Anime import Preferences
import pickle
import numpy as np
import lightfm
# 2. Create the app object
app = FastAPI()
pickle_in = open("pipeline.pkl","rb")
model=pickle.load(pickle_in)
l = pickle.load(pickle_in)
n_users ,n_items = l[0],l[1]
anime=pickle.load(pickle_in)
dataset=pickle.load(pickle_in)

total_genres = Preferences.schema()['required']
def recommendME(model,anime,dataset,user_id=None,new_user_feature=None,k=5):
  nanime=anime.set_index('anime_id')
  if user_id is None:
    user_id = n_users +1
    dataset.fit_partial(users=[user_id],user_features=total_genres)
    # My Feature matrix
    # new_user_feature = [user_id,{' Adventure': 0.11764705882352941, ' Cars': 0.0, ' Comedy': 0.23529411764705882, ' Dementia': 0.0, ' Demons': 0.058823529411764705, ' Drama': 0.17647058823529413, ' Ecchi': 0.058823529411764705, ' Fantasy': 0.35294117647058826, ' Game': 0.058823529411764705, ' Harem': 0.0, ' Hentai': 0.0, ' Historical': 0.0, ' Horror': 0.058823529411764705, ' Josei': 0.0, ' Kids': 0.0, ' Magic': 0.11764705882352941, ' Martial Arts': 0.0, ' Mecha': 0.058823529411764705, ' Military': 0.11764705882352941, ' Music': 0.0, ' Mystery': 0.058823529411764705, ' Parody': 0.058823529411764705, ' Police': 0.17647058823529413, ' Psychological': 0.17647058823529413, ' Romance': 0.23529411764705882, ' Samurai': 0.0, ' School': 0.29411764705882354, ' Sci-Fi': 0.17647058823529413, ' Seinen': 0.11764705882352941, ' Shoujo': 0.058823529411764705, ' Shoujo Ai': 0.0, ' Shounen': 0.29411764705882354, ' Shounen Ai': 0.0, ' Slice of Life': 0.17647058823529413, ' Space': 0.0, ' Sports': 0.058823529411764705, ' Super Power': 0.17647058823529413, ' Supernatural': 0.47058823529411764, ' Thriller': 0.17647058823529413, ' Vampire': 0.0, ' Yaoi': 0.0, ' Yuri': 0.0, 'Action': 0.47058823529411764, 'Adventure': 0.058823529411764705, 'Cars': 0.0, 'Comedy': 0.11764705882352941, 'Dementia': 0.0, 'Demons': 0.0, 'Drama': 0.23529411764705882, 'Ecchi': 0.0, 'Fantasy': 0.0, 'Game': 0.0, 'Harem': 0.0, 'Hentai': 0.0, 'Historical': 0.0, 'Horror': 0.0, 'Josei': 0.0, 'Kids': 0.0, 'Magic': 0.0, 'Martial Arts': 0.0, 'Mecha': 0.0, 'Military': 0.0, 'Music': 0.0, 'Mystery': 0.058823529411764705, 'Parody': 0.0, 'Police': 0.0, 'Psychological': 0.0, 'Romance': 0.0, 'Samurai': 0.0, 'School': 0.0, 'Sci-Fi': 0.058823529411764705, 'Seinen': 0.0, 'Shoujo': 0.0, 'Shounen': 0.0, 'Slice of Life': 0.0, 'Space': 0.0, 'Sports': 0.0, 'Super Power': 0.0, 'Supernatural': 0.0, 'Thriller': 0.0, 'Vampire': 0.0, 'Yaoi': 0.0} ] 
    new_user_feature = [user_id,new_user_feature]
    new_user_feature = dataset.build_user_features([new_user_feature],normalize=False)
  user_id_map = dataset.mapping()[0][user_id] # just user_id -1 
  scores = model.predict(user_id_map, np.arange(n_items),user_features=new_user_feature)
  rank = np.argsort(-scores)
  selected_anime_id =np.array(list(dataset.mapping()[2].keys()))[rank]
  top_items = nanime.loc[selected_anime_id]

  return top_items['name'].values[:k]      


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:Preferences):
    
    user_feature = data.dict()
    k = user_feature['k']
    del user_feature['k']
    print(user_feature)
    rec = recommendME(model,anime,dataset,new_user_feature=user_feature,k=k)
    print(rec)

    return {
        'recommendation': set(rec),
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload