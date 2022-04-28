import os 


def create(name):
 
    main_dir = f"train/{name}"
    
    os.mkdir(main_dir) 
    print("Directory '% s' is built!" % main_dir)