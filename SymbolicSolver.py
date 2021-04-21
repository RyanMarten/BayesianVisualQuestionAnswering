import pandas as pd

color_names = ["red", "green", "blue", "orange", "gray", "yellow"]
image_size = 75

def solve(question_latent, image_latent):
    
    color_i = -1
    for i in range(6):
        if question_latent[i]:
            color_i = i
            break
    color = color_names[color_i]
    
    if question_latent[6]:  # non-relational question
        
        if question_latent[8]:  # What shape is the {} object?
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    return row["Shape"]
        elif question_latent[9]:  # Is the {} object on the left?
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    return int(row["X"] < image_size / 2)
        elif question_latent[10]:  # Is the {} object on the top?
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    return int(row["Y"] < image_size / 2)
        else:
            raise RuntimeError("Invalid Question Latent")
        
    elif question_latent[7]:  # relational question
        
        if question_latent[8]:  # What shape is the object closest to the {} object?
            main_object = None
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    main_object = row
                    break
            image_latent["distance"] = image_latent.apply(lambda x: (x["X"] - main_object["X"])**2 + (x["Y"] - x["Y"])**2 if x["Color"] != main_object["Color"] else 9e10, axis=1)
            closest_object = image_latent.loc[image_latent["distance"].idxmin()]
            return closest_object["Shape"]
        elif question_latent[9]:  # What shape is the object furthest from the {} object?
            main_object = None
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    main_object = row
                    break
            image_latent["distance"] = image_latent.apply(lambda x: (x["X"] - main_object["X"])**2 + (x["Y"] - x["Y"])**2 if x["Color"] != main_object["Color"] else -9e10, axis=1)
            farthest_object = image_latent.loc[image_latent["distance"].idxmax()]
            return farthest_object["Shape"]
        elif question_latent[10]:  # How many objects are the same shape as the {} object?
            main_object = None
            for index, row in image_latent.iterrows():
                if row["Color"] == color:
                    main_object = row
                    break
            same_shape_count = -1
            for index, row in image_latent.iterrows():
                if row["Shape"] == main_object["Shape"]:
                    same_shape_count += 1
            return same_shape_count
        else:
            raise RuntimeError("Invalid Question Latent")
        
    else:
        raise RuntimeError("Invalid Question Latent")

